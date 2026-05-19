import re
from fractions import Fraction

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_boxed(text: str) -> str | None:
    """Return the content of the first \\boxed{} in text, or None if absent."""
    match = _BOXED_RE.search(text)
    return match.group(1).strip() if match else None


def _normalize(value: str) -> str:
    """Normalize a numeric string so 1/2 == 0.5 == 0.50.

    Mirrors GRPORecord.normalized_answer — both must stay in sync.
    Falls back to the stripped string for non-numeric answers.
    """
    stripped = value.strip()
    try:
        return str(float(Fraction(stripped)))
    except (ValueError, ZeroDivisionError):
        return stripped


def _has_correct_steps(completion: str, reference_solution: str) -> bool:
    """Return True if the completion reproduces most intermediate values.

    Heuristic: extract all numbers from the reference solution steps and check
    whether at least half of them appear somewhere in the completion. This is a
    rough proxy for "correct reasoning" that avoids needing a second model.
    """
    reference_numbers = set(_NUMBER_RE.findall(reference_solution))
    if not reference_numbers:
        return False
    found = sum(1 for n in reference_numbers if n in completion)
    return found / len(reference_numbers) >= 0.5


def grpo_reward(
    completions: list[str],
    answer: list[str],
    solution: list[str],
    **kwargs,
) -> list[float | None]:
    """Score each generated completion against the ground-truth answer.

    Scoring rubric (from issue #10):
    - 1.0: boxed answer matches ground truth (after normalization)
    - 0.5: no exact match, but completion reproduces most intermediate steps
    - 0.0: no boxed answer, or wrong answer with no recoverable reasoning

    Args:
        completions:  Model-generated texts, one per prompt in the batch.
        answer:       Ground-truth answers (from GRPORecord.normalized_answer).
        solution:     Reference chain-of-thought (from GRPORecord.solution),
                      used only for partial-credit step matching.

    Returns:
        List of floats in [0.0, 0.5, 1.0], one per completion.
    """
    rewards: list[float | None] = []

    for completion, ground_truth, reference_solution in zip(completions, answer, solution):
        extracted = _extract_boxed(completion)

        if extracted is None:
            rewards.append(0.0)
            continue

        if _normalize(extracted) == _normalize(ground_truth):
            rewards.append(1.0)
        elif _has_correct_steps(completion, reference_solution):
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards
