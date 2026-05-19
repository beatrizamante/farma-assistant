
from pathlib import Path

from peft import LoraConfig, TaskType
from pydantic import BaseModel

from domain.entities.Model_Settings import ModelSettings

_DEEPSEEK_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_LORA = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=_DEEPSEEK_TARGET_MODULES,
    bias="none",
)

class GRPOTrainConfig(BaseModel):
    model_settings: ModelSettings
    lora: LoraConfig = DEFAULT_LORA
    learning_rate: float = 1e-5
    per_device_batch_size: int = 4
    num_epochs: int = 3
    dataset_path: Path

    model_config = {"arbitrary_types_allowed": True}
