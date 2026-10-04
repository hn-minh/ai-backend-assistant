from fastapi import APIRouter, HTTPException
from langsmith import traceable

from api.llm import get_gateway
from core.template import BASH_TEMPLATE
from models.bash_qa import BashQARequest, BashQAResponse

router = APIRouter()


@traceable(name="bash_qa_endpoint", run_type="chain", metadata={"route": "bash_qa"})
@router.post("/bash-qa", response_model=BashQAResponse)
def answer_bash_question(request: BashQARequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    prompt = BASH_TEMPLATE.format(question=request.question,)
    result = get_gateway().route_chat(
        messages=[{"role": "user", "content": prompt}],
        requested_route="bash_qa",
    )

    return BashQAResponse(
        content=result.content,
        route=result.route,
        provider=result.provider,
        model=result.model,
    )
