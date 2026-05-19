
from pathlib import Path
from typing import cast

from datasets import Dataset
from peft import PeftModel, get_peft_model
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from domain.entities.GRPOTrainConfig import GRPOTrainConfig
from infrastructure.r_model.core.build_model import BuildModel
from infrastructure.r_model.core.build_tokenizer import BuildTokenizer
from infrastructure.r_model.training.reward import grpo_reward


def train(config: GRPOTrainConfig, train_dataset: Dataset, output_dir: Path) -> None:
    model = BuildModel(config.model_settings).load().model
    tokenizer = BuildTokenizer(config.model_settings, padding_side="right").load().tokenizer

    model = cast(PeftModel, get_peft_model(model, config.lora))
    model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        num_train_epochs=config.num_epochs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=grpo_reward,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()

    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(adapter_path)
