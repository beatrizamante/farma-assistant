from fractions import Fraction

from pydantic import BaseModel, computed_field


class GRPORecord(BaseModel):
    """Value object representing a single GRPO (Group Relative Policy Optimization) record.

    During GRPO training the model generates its own completions — this record
    supplies the ``problem`` as input and the ``normalized_answer`` as the
    ground-truth signal for the reward function (comparing against the model's
    ``\\boxed{...}`` output).

    ``solution`` stores the reference chain-of-thought from FARMA for
    logging and debugging; it is not fed to the model during RL training.
    """

    problem: str
    solution: str
    answer: str

    @computed_field
    @property
    def prompt(self) -> str:
        """Format the problem as the prompt fed to the model during GRPO training."""
        return (
            f"User: {self.problem}\n"
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            "A:"
        )

    @computed_field
    @property
    def normalized_answer(self) -> str:
        """Normalize the answer so equivalent forms compare as equal.

        Converts fractions to their decimal representation so that values
        like ``1/2`` and ``0.5`` match when the reward function compares
        model output against ground truth.
        """
        stripped = self.answer.strip()
        try:
            return str(float(Fraction(stripped)))
        except (ValueError, ZeroDivisionError):
            return stripped
