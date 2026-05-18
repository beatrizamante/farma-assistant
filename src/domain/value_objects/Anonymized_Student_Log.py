from domain.value_objects.Anonymized_Answer import AnonymizedAnswer
from domain.value_objects.History_Response import HistoryResponse


class AnonymizedStudentLog(HistoryResponse[AnonymizedAnswer]):
    """Student response history with all PII removed.

    ``user_id`` and ``team_id`` have been replaced by ``anon_id`` in every
    ``AnonymizedAnswer`` entry. Safe for persistence, training, and indexing.
    """
