from pydantic import BaseModel


class FarmaLO(BaseModel):
    id:	int
    title: str
    description: str
    image: bytes
    exercises_count:	int
    introductions_count: int
    created_at: str
    updated_at: str
