from pydantic import BaseModel


class FarmaAnswer(BaseModel):
    """Represents a user's answer to a pharmacy solution step."""
    id: int
    solution_step_id: int
    user_id: int | None  = None
    team_id: int | None = None
    response: str
    correct: bool
    attempt_number: int
    created_at: str
    updated_at: str
