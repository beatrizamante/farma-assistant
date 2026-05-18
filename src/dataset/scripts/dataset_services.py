import json
import random
from collections import defaultdict
from pathlib import Path

from domain.entities.Farma_Raw_Log import FarmaRawLog

SEED = 42
_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "dataset" / "processed" / "split_manifest.json"


def dataset_split(
    logs: list[FarmaRawLog],
    ratios: dict[str, float],
) -> tuple[list[FarmaRawLog], list[FarmaRawLog], list[FarmaRawLog]]:
    """Split a list of FarmaRawLog into train/val/test, stratified by LO.

    Exercises are grouped by ``lo_id`` before splitting so every topic is
    proportionally represented in all three splits (avoids topic leakage).

    Args:
        logs: All raw logs to split. Each log carries an exercise with ``lo_id``.
        ratios: Dict with keys ``train_size``, ``val_size``, ``test_size``
                as floats that sum to 1.0.

    Returns:
        Tuple of (train, val, test) lists of FarmaRawLog.
    """
    groups: dict[int, list[FarmaRawLog]] = defaultdict(list)
    for log in logs:
        groups[log.exercise.lo_id].append(log)

    rng = random.Random(SEED)
    train: list[FarmaRawLog] = []
    val: list[FarmaRawLog] = []
    test: list[FarmaRawLog] = []

    for lo_logs in groups.values():
        rng.shuffle(lo_logs)
        n = len(lo_logs)
        n_train = round(n * ratios["train_size"])
        n_val = round(n * ratios["val_size"])

        train.extend(lo_logs[:n_train])
        val.extend(lo_logs[n_train : n_train + n_val])
        test.extend(lo_logs[n_train + n_val :])

    return train, val, test


def save_split_manifest(
    train: list[FarmaRawLog],
    val: list[FarmaRawLog],
    test: list[FarmaRawLog],
) -> None:
    """Persist the split as a JSON manifest of exercise IDs grouped by split.

    The manifest records ``exercise_id`` and ``lo_id`` for each entry so the
    split can be reconstructed or audited without the full log objects.
    """
    def _to_entries(logs: list[FarmaRawLog]) -> list[dict]:
        return [{"exercise_id": log.exercise.id, "lo_id": log.exercise.lo_id} for log in logs]

    manifest = {
        "train": _to_entries(train),
        "val": _to_entries(val),
        "test": _to_entries(test),
    }

    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify_no_overlap(
    train: list[FarmaRawLog],
    val: list[FarmaRawLog],
    test: list[FarmaRawLog],
) -> None:
    """Assert that no exercise ID appears in more than one split.

    Raises ``AssertionError`` with the overlapping IDs if a duplicate is found.
    """
    train_ids = {log.exercise.id for log in train}
    val_ids = {log.exercise.id for log in val}
    test_ids = {log.exercise.id for log in test}

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids

    assert not overlap_tv, f"Overlap between train and val: {overlap_tv}"
    assert not overlap_tt, f"Overlap between train and test: {overlap_tt}"
    assert not overlap_vt, f"Overlap between val and test: {overlap_vt}"
