"""QLoRA SFT phase for the FARMA domain."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from typing import Literal, cast

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
from src.infrastructure.r_model.training.config import (
    PROJECT_ROOT,
    load_train_config,
    resolve_project_path,
)

ModelType = PreTrainedModel | PeftModel


def _resolve_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path)
    if dataset_path.suffix == ".jsonl":
        return load_dataset("json", data_files=str(dataset_path), split="train")
    return load_dataset(str(dataset_path), split="train")


def build_model_and_tokenizer(
    base_model_dir: Path,
    torch_dtype: str = "bfloat16",
    quantization_type: Literal["none", "4bit", "8bit"] = "4bit",
) -> tuple[ModelType, PreTrainedTokenizerBase]:
    settings = ModelSettings(
        model_dir=base_model_dir,
        torch_dtype=torch_dtype,
        quantization_type=quantization_type,
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


def _format_sft_example(example: dict[str, object]) -> str:
    prompt = example.get("prompt", "")
    completion = example.get("completion", "")
    return f"{prompt}{completion}"


def train_sft(config_path: Path) -> None:
    config = load_train_config(config_path)
    model, tokenizer = build_model_and_tokenizer(
        resolve_project_path(config.model.base_model_dir),
        torch_dtype=config.model.torch_dtype,
        quantization_type=config.model.quantization_type,
    )

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

    dataset = _resolve_dataset(resolve_project_path(config.data.sft_dataset_path))
    if "prompt" not in dataset.column_names or "completion" not in dataset.column_names:
        raise ValueError("SFT dataset must contain 'prompt' and 'completion' columns.")

    output_dir = resolve_project_path(config.output.checkpoint_dir) / config.output.sft_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        max_length=config.training.max_seq_length,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        bf16=config.model.torch_dtype == "bfloat16",
        fp16=config.model.torch_dtype == "float16",
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
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/training/farma.toml",
    )
    args = parser.parse_args()
    train_sft(args.config)


if __name__ == "__main__":
    main()
