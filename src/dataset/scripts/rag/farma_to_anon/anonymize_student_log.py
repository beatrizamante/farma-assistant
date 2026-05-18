import hashlib
from dataclasses import dataclass

from pydantic import ValidationError

from dataset.scripts.normalize_math_notation import normalize_math_notation
from domain.entities.Student_Response_History import StudentResponseHistory
from domain.value_objects.Anonymized_Answer import AnonymizedAnswer
from domain.value_objects.Anonymized_Student_Log import AnonymizedStudentLog


@dataclass
class CleaningReport:
    skipped_empty: int = 0
    deduplicated: int = 0

    @property
    def total_dropped(self) -> int:
        return self.skipped_empty + self.deduplicated

    def __str__(self) -> str:
        return (
            f"CleaningReport(skipped_empty={self.skipped_empty}, "
            f"deduplicated={self.deduplicated}, "
            f"total_dropped={self.total_dropped})"
        )


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


def anonymize_student_log(
    raw: StudentResponseHistory,
) -> tuple[AnonymizedStudentLog, CleaningReport]:
    """Replace PII in a StudentResponseHistory with anonymous identifiers.

    Each FarmaAnswer's ``user_id`` and ``team_id`` are hashed into a single
    ``anon_id``. The resulting AnonymizedStudentLog is safe for persistence,
    training pipelines, and RAG indexing.

    Returns the anonymized log and a CleaningReport describing how many
    records were skipped (empty response) or deduplicated (last attempt kept).
    """
    seen: dict[tuple[str, int], AnonymizedAnswer] = {}
    report = CleaningReport()

    for answer in raw.response_history:
        try:
            anonymized = AnonymizedAnswer(
                anon_id=_make_anon_id(answer.user_id, answer.team_id),
                solution_step_id=answer.solution_step_id,
                response=normalize_math_notation(answer.response),
                correct=answer.correct,
                attempt_number=answer.attempt_number,
                created_at=answer.created_at,
                updated_at=answer.updated_at,
            )
        except ValidationError:
            report.skipped_empty += 1
            continue

        key = (anonymized.anon_id, anonymized.solution_step_id)
        if key in seen:
            report.deduplicated += 1
        seen[key] = anonymized

    log = AnonymizedStudentLog(
        correct=raw.correct,
        response_history=list(seen.values()),
        tips_viewed=raw.tips_viewed,
        tip_available=raw.tip_available,
    )
    return log, report
