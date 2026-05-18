import hashlib

from domain.entities.Student_Response_History import StudentResponseHistory
from domain.value_objects.Anonymized_Answer import AnonymizedAnswer
from domain.value_objects.Anonymized_Student_Log import AnonymizedStudentLog


def _make_anon_id(user_id: int | None, team_id: int | None) -> str:
    """Derive a deterministic, irreversible anonymous ID from student identifiers.

    SHA-256 is used so that:
    - The same (user_id, team_id) pair always produces the same anon_id.
    - The original IDs cannot be recovered from the hash.

    Both values are included so that a student who switches teams keeps a
    distinct identity per team context, matching FARMA's access model.
    """
    raw = f"{user_id}:{team_id}".encode()
    return hashlib.sha256(raw).hexdigest()


def anonymize_student_log(raw: StudentResponseHistory) -> AnonymizedStudentLog:
    """Replace PII in a StudentResponseHistory with anonymous identifiers.

    Each FarmaAnswer's ``user_id`` and ``team_id`` are hashed into a single
    ``anon_id``. The resulting AnonymizedStudentLog is safe for persistence,
    training pipelines, and RAG indexing.
    """
    anonymized_answers = [
        AnonymizedAnswer(
            anon_id=_make_anon_id(answer.user_id, answer.team_id),
            solution_step_id=answer.solution_step_id,
            response=answer.response,
            correct=answer.correct,
            attempt_number=answer.attempt_number,
            created_at=answer.created_at,
            updated_at=answer.updated_at,
        )
        for answer in raw.response_history
    ]

    return AnonymizedStudentLog(
        correct=raw.correct,
        response_history=anonymized_answers,
        tips_viewed=raw.tips_viewed,
        tip_available=raw.tip_available,
    )
