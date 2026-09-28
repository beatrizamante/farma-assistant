"""Merge a LoRA adapter into a quantized base model for deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.domain.entities.model_settings import ModelSettings


DEFAULT_OUTPUT_DIR = Path("src/infrastructure/r_model/core/deepseek-math-farma")


def merge_adapter(
    base_model_dir: Path,
    adapter_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tokenizer_dir: Path | None = None,
    device_map: str = "auto",
) -> None:
    """Merge ``adapter_dir`` into ``base_model_dir`` and verify the result."""
    _require_directory(base_model_dir, "base model")
    _require_directory(adapter_dir, "LoRA adapter")
    dtype = ModelSettings(model_dir=base_model_dir).dtype
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose another path or remove the existing artifact."
        )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=dtype,
    )
    model_with_adapter = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = model_with_adapter.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)

    source_tokenizer_dir = tokenizer_dir or adapter_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(source_tokenizer_dir))
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir))
    tokenizer.save_pretrained(str(output_dir))

    _verify_merged_model(output_dir, device_map, dtype)


def _verify_merged_model(output_dir: Path, device_map: str, dtype: torch.dtype) -> None:
    """Load the saved artifact with the same API used by inference."""
    verified_model = AutoModelForCausalLM.from_pretrained(
        str(output_dir),
        device_map=device_map,
        torch_dtype=dtype,
    )
    del verified_model
    AutoTokenizer.from_pretrained(str(output_dir))


def _require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description.capitalize()} directory not found: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Tokenizer directory. Defaults to the adapter, then the base model.",
    )
    parser.add_argument("--device-map", default="auto")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    merge_adapter(
        base_model_dir=args.base_model,
        adapter_dir=args.adapter,
        output_dir=args.output,
        tokenizer_dir=args.tokenizer,
        device_map=args.device_map,
    )
    print(f"Merged model verified at {args.output}")


if __name__ == "__main__":
    main()
