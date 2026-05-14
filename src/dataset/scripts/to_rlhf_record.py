from domain.value_objects.RLHF_record import RLHFRecord
from domain.value_objects.Farma_raw_log import FarmaRawLog


def to_rlhf_record(raw: FarmaRawLog) -> RLHFRecord:
    """Transform a single FARMA raw log into an RLHFRecord value object.

    ``steps`` uses each SolutionStep's description as the chain-of-thought text.
    ``answer`` is the expected correct response from the last solution step.
    """
    return RLHFRecord(
        question=raw.exercise.description,
        steps={str(step.position): step.description for step in raw.steps},
        answer=sorted(raw.steps, key=lambda s: s.position)[-1].response,
    )
