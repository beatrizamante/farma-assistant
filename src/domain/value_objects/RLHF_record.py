from pydantic import BaseModel, computed_field

from domain.value_objects.SFT_Record import SFTRecord

class RLHFRecord(BaseModel):
    """Value object representing a single SFT (Supervised Fine-Tuning) record.

    Holds the raw fields extracted from a FARMA exercise and exposes
    ``prompt_structure`` — the formatted prompt/completion pair expected
    by the DeepSeek instruction-tuning pipeline.
    """

    question: str
    steps: dict[str, str]
    answer: str

    @computed_field
    @property
    def prompt_structure(self) -> SFTRecord:
        """Build the prompt/completion pair in DeepSeek SFT format.

        The ``prompt`` contains everything up to and including ``Assistant:``
        so the model learns to complete from that point onward.
        The ``completion`` is the chain-of-thought followed by the boxed answer.

        Returns:
            dict with keys ``"prompt"`` and ``"completion"``.
        """
        chain_of_thought = "\n".join(self.steps.values())
        return SFTRecord(
            prompt=f"User: {self.question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nA:",
            completion=f"{chain_of_thought}\n\\boxed{{{self.answer}}}<｜end▁of▁sentence｜>",
        )
