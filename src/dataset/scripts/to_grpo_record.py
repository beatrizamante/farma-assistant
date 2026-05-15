from domain.value_objects.Farma_Raw_Log import FarmaRawLog
from domain.value_objects.GRPO_Record import GRPORecord


def to_grpo_record(raw: FarmaRawLog) -> GRPORecord:
    """Transform a single FARMA raw log into a GRPORecord value object.

    ``solution`` is built from the ordered step descriptions (reference CoT).
    ``answer`` is the correct response from the last solution step, which
    ``GRPORecord.normalized_answer`` will normalize before reward computation.
    """
    ordered_steps = sorted(raw.steps, key=lambda s: s.position)
    solution = "\n".join(
        f"Step {step.position}: {step.description}" for step in ordered_steps
    )
    return GRPORecord(
        problem=raw.exercise.description,
        solution=solution,
        answer=ordered_steps[-1].response,
    )
