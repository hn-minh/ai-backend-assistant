import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langsmith import traceable

from core.config import get_settings
from core.routing import ProviderConfig, RouteConfig, RoutingConfig, get_routing_config


RouteClassifier = Callable[[Sequence[dict[str, str]]], dict[str, str] | str]
RouteInvoker = Callable[[RouteConfig, ProviderConfig, Sequence[dict[str, str]]], str]


class RouteDecision(BaseModel):
    route: str
    reason: str


@dataclass
class RoutedLLMResult:
    content: str
    route: str
    provider: str
    model: str
    classifier_model: str
    reason: str


def _normalize_messages(messages: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def _hash_payload(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _to_langchain_messages(messages: Sequence[dict[str, str]]):
    converted = []
    for message in _normalize_messages(messages):
        role = message["role"]
        content = message["content"]
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def _flatten_messages(messages: Sequence[dict[str, str]]) -> str:
    lines = []
    for message in _normalize_messages(messages):
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)


def _strip_markdown_fence(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return cleaned


def _extract_route_from_output(raw_output: str, allowed_routes: set[str]) -> str | None:
    cleaned = _strip_markdown_fence(raw_output)
    if not cleaned:
        return None

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            route_value = parsed.get("route")
            if isinstance(route_value, str) and route_value in allowed_routes:
                return route_value
    except Exception:
        pass

    for route_name in sorted(allowed_routes, key=len, reverse=True):
        if cleaned == route_name:
            return route_name
        if re.search(rf"\b{re.escape(route_name)}\b", cleaned):
            return route_name
    return None


def _build_chat_model(
    api_base: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=api_base,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_seconds,
    )


class LLMGateway:
    def __init__(
        self,
        settings=None,
        routing_config: RoutingConfig | None = None,
        classifier: RouteClassifier | None = None,
        invoker: RouteInvoker | None = None,
    ):
        self.settings = settings or get_settings()
        self.routing_config = routing_config or get_routing_config()
        self._classifier_override = classifier
        self._invoker_override = invoker

    def _get_model(self, provider: ProviderConfig, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
        return _build_chat_model(
            api_base=provider.api_base,
            api_key=provider.api_key or self.settings.openai_api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=self.settings.request_timeout_seconds,
        )

    def _normalize_decision(self, decision: dict[str, str] | str) -> RouteDecision:
        if isinstance(decision, str):
            return RouteDecision(route=decision, reason="Provided by classifier override")
        return RouteDecision.model_validate(decision)

    def _eligible_routes(self) -> dict[str, RouteConfig]:
        eligible_routes = {
            route_name: route_config
            for route_name, route_config in self.routing_config.routes.items()
            if route_config.classifier_enabled
        }
        if eligible_routes:
            return eligible_routes
        return self.routing_config.routes

    @traceable(name="gateway_classify", run_type="chain")
    def classify(self, messages: Sequence[dict[str, str]]) -> RouteDecision:
        if self._classifier_override is not None:
            return self._normalize_decision(self._classifier_override(messages))

        parser = PydanticOutputParser(pydantic_object=RouteDecision)
        classifier_config = self.routing_config.classifier
        provider = self.routing_config.providers[classifier_config.provider]
        eligible_routes = self._eligible_routes()
        route_names = set(eligible_routes.keys())
        route_definitions = "\n".join(
            f"- {route_name}: {route_config.description}"
            for route_name, route_config in eligible_routes.items()
        )
        chat_model = self._get_model(
            provider=provider,
            model=classifier_config.model,
            temperature=classifier_config.temperature,
            max_tokens=classifier_config.max_tokens,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", classifier_config.instructions),
                (
                    "human",
                    "Available routes:\n{route_definitions}\n\nConversation:\n{conversation}\n\n{format_instructions}",
                ),
            ]
        )
        chain = prompt | chat_model | parser

        try:
            decision = chain.invoke(
                {
                    "route_definitions": route_definitions,
                    "conversation": _flatten_messages(messages),
                    "format_instructions": parser.get_format_instructions(),
                }
            )
        except Exception as exc:
            raw_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", classifier_config.instructions),
                    (
                        "human",
                        "Available routes:\n{route_definitions}\n\nConversation:\n{conversation}\n\n"
                        "Return ONLY one route name from this list: {route_list}. "
                        "No JSON, no explanation.",
                    ),
                ]
            )
            try:
                raw_response = (raw_prompt | chat_model).invoke(
                    {
                        "route_definitions": route_definitions,
                        "conversation": _flatten_messages(messages),
                        "route_list": ", ".join(sorted(route_names)),
                    }
                )
                parsed_route = _extract_route_from_output(str(raw_response.content), route_names)
                if parsed_route is not None:
                    return RouteDecision(
                        route=parsed_route,
                        reason=f"Classifier fallback recovered from {exc.__class__.__name__}",
                    )
            except Exception:
                pass

            return RouteDecision(
                route=self.routing_config.default_route,
                reason=f"Classifier fallback after error: {exc.__class__.__name__}",
            )

        normalized = RouteDecision.model_validate(decision)
        if normalized.route not in self.routing_config.routes:
            return RouteDecision(
                route=self.routing_config.default_route,
                reason=f"Classifier returned unknown route '{normalized.route}', default applied",
            )
        return normalized
        
    @traceable(name="gateway_invoke", run_type="llm")
    def invoke_route(
        self,
        route_config: RouteConfig,
        provider: ProviderConfig,
        messages: Sequence[dict[str, str]],
    ) -> str:
        if self._invoker_override is not None:
            return self._invoker_override(route_config, provider, messages)

        temperature = route_config.temperature if route_config.temperature is not None else provider.temperature
        max_tokens = route_config.max_tokens if route_config.max_tokens is not None else provider.max_tokens
        effective_messages = list(messages)
        if route_config.system_prompt:
            effective_messages = [{"role": "system", "content": route_config.system_prompt}, *effective_messages]

        response = self._get_model(
            provider=provider,
            model=route_config.model,
            temperature=temperature,
            max_tokens=max_tokens,
        ).invoke(_to_langchain_messages(effective_messages))

        if isinstance(response.content, str):
            return response.content
        return str(response.content)

    def route_chat(self, messages: Sequence[dict[str, str]], requested_route: str | None = None) -> RoutedLLMResult:
        if not messages:
            raise ValueError("At least one message is required")

        if requested_route:
            decision = RouteDecision(route=requested_route, reason="Route forced by request")
        else:
            decision = self.classify(messages)

        route_name = decision.route
        if route_name not in self.routing_config.routes:
            route_name = self.routing_config.default_route
            decision = RouteDecision(
                route=route_name,
                reason=f"Unknown requested route '{decision.route}', default applied",
            )

        route_config = self.routing_config.routes[route_name]
        provider = self.routing_config.providers[route_config.provider]
        content = self.invoke_route(route_config, provider, messages)
        
        return RoutedLLMResult(
            content=content,
            route=route_name,
            provider=route_config.provider,
            model=route_config.model,
            classifier_model=self.routing_config.classifier.model,
            reason=decision.reason,
        )


def get_gateway() -> LLMGateway:
    return LLMGateway()