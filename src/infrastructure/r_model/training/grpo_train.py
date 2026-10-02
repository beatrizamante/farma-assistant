"""QLoRA + GRPO phase for mathematical reasoning reinforcement."""

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
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from src.domain.entities.model_settings import ModelSettings
from src.infrastructure.r_model.training.config import (
    PROJECT_ROOT,
    load_train_config,
    resolve_project_path,
)
from src.infrastructure.r_model.training.reward import grpo_reward

ModelType = PreTrainedModel | PeftModel


def _resolve_dataset(path: str | Path) -> Dataset:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix == ".jsonl":
        return load_dataset("json", data_files=str(dataset_path), split="train")
    return load_dataset(str(dataset_path), split="train")


def build_model_and_tokenizer(
    base_model_dir: Path,
    adapter_dir: Path | None = None,
    torch_dtype: str = "bfloat16",
    quantization_type: Literal["none", "4bit", "8bit"] = "4bit",
) -> tuple[ModelType, PreTrainedTokenizerBase]:
    settings = ModelSettings(
        model_dir=base_model_dir,
        torch_dtype=torch_dtype,
        quantization_type=quantization_type,
    )
    base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        quantization_config=settings.quantization_config,
        torch_dtype=settings.dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir or base_model_dir))
    tokenizer.pad_token = tokenizer.eos_token

    if adapter_dir is not None and adapter_dir.exists():
        return cast(ModelType, PeftModel.from_pretrained(base_model, str(adapter_dir))), tokenizer

    return cast(ModelType, base_model), tokenizer


def train_grpo(config_path: Path) -> None:
    config = load_train_config(config_path)
    checkpoint = resolve_project_path(config.model.adapter_dir) if config.model.adapter_dir else None
    dataset = _resolve_dataset(resolve_project_path(config.data.grpo_dataset_path))
    if "problem" not in dataset.column_names or "answer" not in dataset.column_names:
        raise ValueError("GRPO dataset must contain 'problem' and 'answer' columns.")

    model, tokenizer = build_model_and_tokenizer(
        resolve_project_path(config.model.base_model_dir),
        checkpoint,
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
    if not isinstance(model, PeftModel):
        model = cast(ModelType, get_peft_model(cast(PreTrainedModel, model), peft_config))

    mapped_dataset = dataset.map(
        lambda row: {
            "prompt": f"User: {row['problem']}\nPlease reason step by step and put the final answer inside \\boxed{{}}.\n\nA:",
            "answer": row["answer"],
            "solution": row.get("solution", ""),
        },
        remove_columns=list(dataset.column_names),
    )

    output_dir = resolve_project_path(config.output.checkpoint_dir) / config.output.grpo_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        num_generations=config.grpo.num_generations,
        beta=config.grpo.beta,
        max_completion_length=config.grpo.max_completion_length,
        save_steps=config.training.save_steps,
        logging_steps=config.training.logging_steps,
        bf16=config.model.torch_dtype == "bfloat16",
        fp16=config.model.torch_dtype == "float16",
    )

    trainer = GRPOTrainer(
        model=cast(PreTrainedModel | PeftModel, model),
        reward_funcs=grpo_reward,
        args=training_args,
        train_dataset=mapped_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA + GRPO training for FARMA")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/training/farma.toml",
    )
    args = parser.parse_args()
    train_grpo(args.config)


if __name__ == "__main__":
    main()
