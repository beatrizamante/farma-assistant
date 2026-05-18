from typing import Generic, TypeVar

from pydantic import BaseModel

from domain.entities.FARMA.Farma_Tips import FarmaTip

T = TypeVar("T")


class HistoryResponse(BaseModel, Generic[T]):
    """Generic base for student response history shapes.

    ``T`` is the answer type — ``FarmaAnswer`` for raw logs,
    ``AnonymizedAnswer`` for processed/safe records.
    """

    correct: bool
    response_history: list[T]
    tips_viewed: list[FarmaTip]
    tip_available: bool
