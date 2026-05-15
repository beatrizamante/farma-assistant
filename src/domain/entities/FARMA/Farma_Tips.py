from pydantic import BaseModel


class FarmaTip(BaseModel):
    id: int
    description: str
    number_attempts: int
    position: int
    exercise_id: int
    created_at: str
    updated_at: str
