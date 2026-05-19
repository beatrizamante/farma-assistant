from pathlib import Path
from typing import Literal

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings
from transformers import PreTrainedConfig, BitsAndBytesConfig

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

class ModelSettings(BaseSettings):
    """Settings for model configuration and loading."""

    model_dir: Path
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    quantization_type: Literal["none", "4bit", "8bit"] = "none"

    @field_validator("model_dir")
    @classmethod
    def must_exist(cls, v: Path) -> Path:
        """Validate that the model directory exists."""
        if not v.exists():
            raise ValueError(f"model_dir does not exist: {v}")
        return v

    @property
    def dtype(self) -> torch.dtype:
        """Get the torch dtype based on torch_dtype setting."""
        return _DTYPE_MAP[self.torch_dtype]

    @property
    def quantization_config(self) -> BitsAndBytesConfig | None:
        """Build the BitsAndBytesConfig for the chosen quantization type.

        ``bnb_4bit_compute_dtype`` is derived from ``torch_dtype`` so both
        settings always stay in sync — passing mismatched dtypes to
        ``from_pretrained`` and ``BitsAndBytesConfig`` is a common source of
        silent precision bugs.
        """
        if self.quantization_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
            )
        if self.quantization_type == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    @property
    def pretrained_config(self) -> PreTrainedConfig:
        """Get the pretrained model configuration."""
        return PreTrainedConfig.from_pretrained(str(self.model_dir))

    model_config = {"env_prefix": "MODEL_"}
