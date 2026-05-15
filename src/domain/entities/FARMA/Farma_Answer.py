from typing import Optional
from pydantic import BaseModel


class FarmaAnswer(BaseModel):
    """Represents a user's answer to a pharmacy solution step."""
    id: int
    solution_step_id: int
    user_id: Optional[int] = None
    team_id: Optional[int] = None
    response: str
    correct: bool
    attempt_number: int
    created_at: str
    updated_at: str
