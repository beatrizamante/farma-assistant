import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger("watchmeai")
router = APIRouter()

@router.post("/v1/chat")
async def prompt_route():
    """

    """
    try:
        return {}
    except Exception as e:
        logging.error("Error processing image: %s", str(e))
        raise HTTPException(status_code=422, detail={"Failed to process image: %s", str(e)}) from e
