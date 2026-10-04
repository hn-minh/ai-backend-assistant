from pydantic import BaseModel


class BackendQARequest(BaseModel):
    question: str


class BackendQAResponse(BaseModel):
    content: str
    route: str
    provider: str
    model: str