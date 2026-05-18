from dataset.scripts.normalize_math_notation import normalize_math_notation
from domain.entities.Farma_Raw_Log import FarmaRawLog
from domain.value_objects.fine_tuning.GRPO_Record import GRPORecord


def to_grpo_record(raw: FarmaRawLog) -> GRPORecord:
    """Transform a single FARMA raw log into a GRPORecord value object.

    ``solution`` is built from the ordered step descriptions (reference CoT).
    ``answer`` is the correct response from the last solution step, which
    ``GRPORecord.normalized_answer`` will normalize before reward computation.
    """
    ordered_steps = sorted(raw.steps, key=lambda s: s.position)
    solution = "\n".join(
        f"Step {step.position}: {normalize_math_notation(step.description)}" for step in ordered_steps
    )
    return GRPORecord(
        problem=normalize_math_notation(raw.exercise.description),
        solution=solution,
        answer=normalize_math_notation(ordered_steps[-1].response),
    )
