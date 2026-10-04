from pydantic import BaseModel


class BashQARequest(BaseModel):
    question: str

class BashQAResponse(BaseModel):
    content: str
    route: str
    provider: str
    model: str