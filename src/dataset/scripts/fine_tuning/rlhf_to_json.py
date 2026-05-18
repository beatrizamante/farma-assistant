import json
from pathlib import Path

from domain.value_objects.fine_tuning.RLHF_Record import RLHFRecord

_OUTPUT_PATH = Path("src/dataset/sft/data.jsonl")

def rlhf_to_json(rlhf_records: list[RLHFRecord]) -> None:
    """Serialize a list of RLHFRecord value objects to a .jsonl file.

    Each line in the output file is a JSON object with ``prompt`` and
    ``completion`` keys, validated by Pydantic before writing.
    """
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in rlhf_records:
            f.write(json.dumps(record.prompt_structure.model_dump(), ensure_ascii=False) + "\n")
