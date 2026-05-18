from pydantic import BaseModel

from domain.entities.FARMA.Farma_Exercise import FarmaExercise
from domain.entities.FARMA.Farma_Steps import FarmaSteps



class FarmaRawLog(BaseModel):
    """Represents raw log data for a Farma exercise session."""
    exercise: FarmaExercise
    steps: list[FarmaSteps]
