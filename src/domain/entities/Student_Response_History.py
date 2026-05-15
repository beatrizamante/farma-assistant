from pydantic import BaseModel

from domain.entities.FARMA.Farma_Answer import FarmaAnswer
from domain.entities.FARMA.Farma_Tips import FarmaTip


class StudentResponseHistory(BaseModel):
    """Model representing a student's response history including answers and tips viewed."""
    correct: bool
    response_history: list[FarmaAnswer]
    tips_viewed: list[FarmaTip]
    tip_available: bool
