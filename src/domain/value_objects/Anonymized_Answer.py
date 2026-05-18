from pydantic import BaseModel, field_validator


class AnonymizedAnswer(BaseModel):
    """FarmaAnswer with student identifiers replaced by a deterministic anonymous ID.

    ``user_id`` and ``team_id`` are removed — replaced by ``anon_id``,
    which is a SHA-256 hash of the original identifiers. The same student
    always produces the same ``anon_id`` across the dataset.
    """

    anon_id: str
    solution_step_id: int
    response: str
    correct: bool
    attempt_number: int
    created_at: str
    updated_at: str

    @field_validator("response")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("incomplete attempt: response is empty")
        return v
