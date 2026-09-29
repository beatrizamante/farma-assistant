from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    base_model_dir: Path
    adapter_dir: Path | None = None
    output_dir: Path | None = None


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
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


@dataclass
class TrainConfig:
    model: ModelConfig
    lora: LoRAConfig
    training: dict[str, object] = field(default_factory=dict)


def load_train_config(path: Path) -> TrainConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    model = raw.get("model", {})
    lora = raw.get("lora", {})
    training = raw.get("training", {})

    return TrainConfig(
        model=ModelConfig(
            base_model_dir=Path(model["base_model_dir"]),
            adapter_dir=Path(model["adapter_dir"]) if model.get("adapter_dir") else None,
            output_dir=Path(model["output_dir"]) if model.get("output_dir") else None,
        ),
        lora=LoRAConfig(
            r=int(lora.get("r", 16)),
            lora_alpha=int(lora.get("lora_alpha", 32)),
            lora_dropout=float(lora.get("lora_dropout", 0.05)),
            target_modules=list(lora.get("target_modules", [
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj",
            ])),
        ),
        training=dict(training),
    )
