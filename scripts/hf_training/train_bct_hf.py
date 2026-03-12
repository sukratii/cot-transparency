"""
BCT fine-tuning: SFTTrainer on (biased_prompt, llama_clean_response) pairs.

Loss is cross-entropy on response tokens only — identical to Tinker's
renderer.build_supervised_example() which sets weight=0 on prompt tokens.

Usage:
    # Small experiment (1000 BCT + 500 instruct):
    python scripts/hf_training/train_bct_hf.py \
        --train dataset_dumps/train_seed_42/llama-3-1-8b-instruct/small_mixed.jsonl \
        --output outputs/bct-small

    # Full run:
    python scripts/hf_training/train_bct_hf.py \
        --train dataset_dumps/train_seed_42/llama-3-1-8b-instruct/mixed.jsonl \
        --output outputs/bct-full
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training JSONL")
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument("--output", default="outputs/bct", help="Output dir for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # ── Load & format data ───────────────────────────────────────────────────
    samples = load_jsonl(Path(args.train))
    print(f"Loaded {len(samples)} training samples from {args.train}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Apply LLaMA-3 chat template.
    # SFTTrainer masks prompt tokens (weight=0) using the assistant header
    # boundary — equivalent to Tinker's renderer.build_supervised_example().
    def format_messages(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = Dataset.from_list(samples).map(
        format_messages, remove_columns=["messages"]
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    # Tinker targets attn + MLP + unembed by default (train_mlp=True, train_attn=True)
    # We match that here with all attention projections + gate/up/down proj
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none" if args.no_wandb else "wandb",
        run_name=Path(args.output).name,
        dataloader_num_workers=0,
    )

    # ── SFTTrainer ────────────────────────────────────────────────────────────
    # dataset_text_field="text" tells SFTTrainer to use the formatted string.
    # It automatically masks prompt tokens (-100) so CE loss is computed
    # only on response tokens — same as Tinker's weighted NLL with weight=0
    # on prompt tokens and weight=1 on response tokens.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"\nStarting BCT training for {args.epochs} epoch(s)...")

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nLoRA adapter saved to: {args.output}")
    print(f"Next: python scripts/hf_training/merge_lora.py --adapter {args.output}")


if __name__ == "__main__":
    main()
