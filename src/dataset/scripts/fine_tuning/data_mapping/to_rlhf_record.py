from dataset.scripts.normalize_math_notation import normalize_math_notation
from domain.entities.FARMA.Farma_Raw_Log import FarmaRawLog
from domain.value_objects.fine_tuning.RLHF_Record import RLHFRecord


def to_rlhf_record(raw: FarmaRawLog) -> RLHFRecord:
    """Transform a single FARMA raw log into an RLHFRecord value object.

    ``steps`` uses each SolutionStep's description as the chain-of-thought text.
    ``answer`` is the expected correct response from the last solution step.
    """
    ordered_steps = sorted(raw.steps, key=lambda s: s.position)
    return RLHFRecord(
        question=normalize_math_notation(raw.exercise.description),
        steps={str(step.position): normalize_math_notation(step.description) for step in ordered_steps},
        answer=normalize_math_notation(ordered_steps[-1].response),
    )
