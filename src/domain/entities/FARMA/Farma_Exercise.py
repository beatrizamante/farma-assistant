from pydantic import BaseModel


class FarmaExercise(BaseModel):
    """Represents a Farma exercise with metadata and solution information."""
    id: int
    title:	str
    description: str
    public: bool
    position: int
    solution_steps_count:	int
    lo_id: int
    created_at: str
    updated_at:	str
