from fastapi import APIRouter
from .chat import router as chat_router
from .backend_qa import router as backend_qa_router
from .bash_qa import router as bash_qa_router

router = APIRouter()
router.include_router(chat_router, tags=["Routed Chat"])
router.include_router(backend_qa_router, tags=["Backend QA"])
router.include_router(bash_qa_router, tags=["Bash QA"])