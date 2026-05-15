from pydantic import BaseModel


class FarmaSteps(BaseModel):
    """Represents a Farma step entity."""
    id: int
    title: str
    description: str
    response: str
    decimal_digits: int
    public: bool
    tips_display_mode: str
    position: int
    tips_count: int
    exercise_id: int
    created_at: str
    updated_at: str
