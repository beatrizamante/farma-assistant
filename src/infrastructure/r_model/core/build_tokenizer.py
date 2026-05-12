
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.domain.entities.model_settings import ModelSettings


class BuildTokenizer:
    """Lazy loader for the DeepSeek-Math tokenizer."""

    def __init__(self, settings: ModelSettings, padding_side: str | None = None) -> None:
        """Args:
            settings: Model settings containing the path to the tokenizer files.
            padding_side: Padding side override — "right" for fine-tuning,
                "left" for inference. If None, keeps the tokenizer default.
        """
        self._settings = settings
        self._padding_side = padding_side
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def load(self) -> "BuildTokenizer":
        """Load the tokenizer from disk and apply pad token and padding side config.

        Returns:
            Self, to allow fluent chaining: BuildTokenizer(settings).load()
        """
        tokenizer = AutoTokenizer.from_pretrained(str(self._settings.model_dir))
        tokenizer.pad_token = tokenizer.eos_token
        if self._padding_side is not None:
            tokenizer.padding_side = self._padding_side
        self._tokenizer = tokenizer
        return self

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the loaded tokenizer.

        Returns:
            The loaded tokenizer instance.

        Raises:
            RuntimeError: If load() has not been called yet.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer
