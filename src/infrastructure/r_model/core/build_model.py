from transformers import AutoModelForCausalLM, PreTrainedModel

from src.domain.entities.model_settings import ModelSettings

class BuildModel:
    """Builds and manages a pre-trained language model for causal language modeling."""

    def __init__(self, settings: ModelSettings) -> None:
        self._settings = settings
        self._model: PreTrainedModel | None = None

    def load(self) -> "BuildModel":
        """Load the pre-trained model from the configured model directory."""
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._settings.model_dir),
            torch_dtype=self._settings.dtype,
            device_map=self._settings.device_map,
        )
        return self

    @property
    def model(self) -> PreTrainedModel:
        """Get the loaded pre-trained model.

        Returns:
            The loaded PreTrainedModel instance.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model
