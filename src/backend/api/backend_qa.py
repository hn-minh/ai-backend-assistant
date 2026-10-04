from fastapi import APIRouter, HTTPException
from langsmith import traceable

from api.llm import get_gateway
from core.template import BACKEND_TEMPLATE
from models.backend_qa import BackendQARequest, BackendQAResponse


router = APIRouter()


@traceable(name="backend_qa_endpoint", run_type="chain", metadata={"route": "backend_qa"})
@router.post("/backend-qa", response_model=BackendQAResponse)
def answer_backend_question(request: BackendQARequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    prompt = BACKEND_TEMPLATE.format(question=request.question)
    result = get_gateway().route_chat(
        messages=[{"role": "user", "content": prompt}],
        requested_route="backend_qa",
    )

    return BackendQAResponse(
        content=result.content,
        route=result.route,
        provider=result.provider,
        model=result.model,
    )
