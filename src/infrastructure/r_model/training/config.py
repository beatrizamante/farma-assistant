from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfig):
    base_model_dir: Path
    adapter_dir: Path | None = None
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    quantization_type: Literal["none", "4bit", "8bit"] = "4bit"


class LoRAConfig(StrictConfig):
    r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class TrainingConfig(StrictConfig):
    per_device_train_batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    num_epochs: float = Field(default=3, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    max_seq_length: int = Field(default=1024, gt=0)
    save_steps: int = Field(default=100, gt=0)
    logging_steps: int = Field(default=10, gt=0)


class GRPOConfig(StrictConfig):
    num_generations: int = Field(default=8, gt=1)
    beta: float = Field(default=0.001, ge=0)
    max_completion_length: int = Field(default=512, gt=0)


class DatasetConfig(StrictConfig):
    sft_dataset_path: Path
    grpo_dataset_path: Path


class OutputConfig(StrictConfig):
    checkpoint_dir: Path
    sft_dir_name: str = "farma-sft"
    grpo_dir_name: str = "farma-grpo"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    data: DatasetConfig
    output: OutputConfig


def resolve_project_path(path: Path) -> Path:
    """Resolve config paths relative to the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_train_config(path: Path) -> TrainConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return TrainConfig.model_validate(raw)
