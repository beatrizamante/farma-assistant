from pydantic import BaseModel

class SFTRecord(BaseModel):
    """Represents a supervised fine-tuning record with prompt and completion."""
    prompt: str
    completion: str
