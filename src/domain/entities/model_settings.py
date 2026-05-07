from pathlib import Path

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings
from transformers import PreTrainedConfig

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
    def pretrained_config(self) -> PreTrainedConfig:
        """Get the pretrained model configuration."""
        return PreTrainedConfig.from_pretrained(str(self.model_dir))

    model_config = {"env_prefix": "MODEL_"}
