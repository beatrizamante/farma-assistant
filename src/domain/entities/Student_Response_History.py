from domain.entities.FARMA.Farma_Answer import FarmaAnswer
from domain.value_objects.History_Response import HistoryResponse


class StudentResponseHistory(HistoryResponse[FarmaAnswer]):
    """Raw student response history as returned by the FARMA API.

    Contains ``FarmaAnswer`` entries which carry ``user_id`` and ``team_id``.
    Must be anonymized before any persistence or training step.
    """
