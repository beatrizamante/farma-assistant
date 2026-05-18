import json
from pathlib import Path

from domain.value_objects.fine_tuning.GRPO_Record import GRPORecord

_OUTPUT_PATH = Path(__file__).resolve().parents[3] / "dataset" / "grpo" / "data.jsonl"


def grpo_to_json(grpo_records: list[GRPORecord]) -> None:
    """Serialize a list of GRPORecord value objects to a .jsonl file.

    Each line contains ``problem``, ``solution``, ``answer``, and
    ``normalized_answer`` — all validated by Pydantic before writing.
    The reward function should use ``normalized_answer`` for comparison.
    """
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in grpo_records:
            f.write(
                json.dumps(record.model_dump(), ensure_ascii=False) + "\n"
            )
