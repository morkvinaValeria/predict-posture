from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    return {"message": "Service is available"}