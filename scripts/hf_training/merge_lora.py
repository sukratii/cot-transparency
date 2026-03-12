"""
Merge a LoRA adapter into the base model weights.

Required before running evals with run_suite.py (Inspect AI's hf/ provider
loads plain HuggingFace models, not PEFT adapters).

Usage:
    python scripts/hf_training/merge_lora.py \
        --adapter outputs/bct-small \
        --output  outputs/bct-small-merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter (from train_bct_hf.py)")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model ID (defaults to adapter_config.json base_model_name_or_path)",
    )
    args = parser.parse_args()

    # Read base model from adapter config if not specified
    if args.base_model is None:
        import json
        from pathlib import Path
        adapter_config = json.loads((Path(args.adapter) / "adapter_config.json").read_text())
        base_model = adapter_config["base_model_name_or_path"]
    else:
        base_model = args.base_model

    print(f"Base model:  {base_model}")
    print(f"Adapter:     {args.adapter}")
    print(f"Output:      {args.output}")

    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # merge on CPU to avoid VRAM issues
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output} ...")
    model.save_pretrained(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(args.output)

    print(f"\nDone. Merged model saved to: {args.output}")
    print(f"Next: python -m sycophancy_eval_inspect.run_suite --experiment llama-bct-small --model hf/{args.output} --limit 100 --bias-types suggested_answer")


if __name__ == "__main__":
    main()
