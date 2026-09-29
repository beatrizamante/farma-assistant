"""QLoRA SFT phase for the FARMA domain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from typing import cast

from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from src.domain.entities.model_settings import ModelSettings
from src.infrastructure.r_model.training.config import load_train_config

ModelType = PreTrainedModel | PeftModel


def _resolve_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path)
    if dataset_path.suffix == ".jsonl":
        return load_dataset("json", data_files=str(dataset_path), split="train")
    return load_dataset(str(dataset_path), split="train")


def build_model_and_tokenizer(
    base_model_dir: Path,
    torch_dtype: str = "bfloat16",
) -> tuple[ModelType, PreTrainedTokenizerBase]:
    settings = ModelSettings(
        model_dir=base_model_dir,
        torch_dtype=torch_dtype,
        quantization_type="4bit",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir))
    tokenizer.pad_token = tokenizer.eos_token

    model: ModelType = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        quantization_config=settings.quantization_config,
        torch_dtype=settings.dtype,
        device_map="auto",
    )
    return model, tokenizer


def _as_int(mapping: dict[str, object], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(value)
    return default


def _as_float(mapping: dict[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(value)
    return default


def _format_sft_example(example: dict[str, object]) -> str:
    prompt = example.get("prompt", "")
    completion = example.get("completion", "")
    return f"{prompt}{completion}"


def train_sft(config_path: Path) -> None:
    config = load_train_config(config_path)
    model, tokenizer = build_model_and_tokenizer(config.model.base_model_dir)

    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base_model = cast(PreTrainedModel, model)
    model = cast(ModelType, get_peft_model(base_model, peft_config))

    dataset = _resolve_dataset("src/dataset/sft/data.jsonl")
    if "prompt" not in dataset.column_names or "completion" not in dataset.column_names:
        raise ValueError("SFT dataset must contain 'prompt' and 'completion' columns.")

    output_dir = config.model.output_dir or Path("src/infrastructure/r_model/training/checkpoints/farma-sft")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=_as_int(config.training, "per_device_train_batch_size", 2),
        gradient_accumulation_steps=_as_int(config.training, "gradient_accumulation_steps", 8),
        learning_rate=_as_float(config.training, "learning_rate", 2e-4),
        max_steps=_as_int(config.training, "max_steps", 1000),
        save_steps=_as_int(config.training, "save_steps", 100),
        logging_steps=_as_int(config.training, "logging_steps", 10),
        bf16=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=_format_sft_example,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA SFT training for FARMA")
    parser.add_argument("--config", type=Path, default=Path("src/infrastructure/r_model/training/train_config.toml"))
    args = parser.parse_args()
    train_sft(args.config)


if __name__ == "__main__":
    main()
