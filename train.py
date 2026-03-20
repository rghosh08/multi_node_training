#!/usr/bin/env python3
"""Multi-node multi-GPU Supervised Fine-Tuning trainer using TRL.

Designed to run inside a Kubeflow PyTorchJob on Kubernetes.
Torchrun injects RANK, LOCAL_RANK, WORLD_SIZE and handles the
distributed process group initialisation.
"""

import argparse
import logging
import os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for SFT training."""
    parser = argparse.ArgumentParser(description="Multi-node SFT Trainer")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model identifier (e.g. meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset identifier (e.g. tatsu-lab/alpaca)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/output/sft-checkpoints",
        help="Directory for model checkpoints",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name for experiment tracking",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Optional Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity/team name",
    )
    return parser.parse_args()


def format_alpaca(example: Dict[str, str]) -> Dict[str, str]:
    """Convert an Alpaca-style example to a single text field."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp.strip():
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return {"text": text}


def main() -> None:
    """Entry point for SFT training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logger.info(
        "Distributed env — rank=%d  local_rank=%d  world_size=%d",
        rank,
        local_rank,
        world_size,
    )
    logger.info(
        "Training params — model=%s  dataset=%s  epochs=%d",
        args.model_name,
        args.dataset_name,
        args.epochs,
    )

    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    # ── Dataset ────────────────────────────────────────────────
    dataset = load_dataset(args.dataset_name, split="train")

    if "instruction" in dataset.column_names:
        logger.info("Detected Alpaca-style dataset — applying formatting")
        dataset = dataset.map(
            format_alpaca,
            remove_columns=dataset.column_names,
            desc="Formatting dataset",
        )
        text_field = "text"
    elif "text" in dataset.column_names:
        text_field = "text"
    else:
        raise ValueError(
            f"Cannot auto-detect text column. Available: {dataset.column_names}"
        )

    # ── Training config ────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        dataset_text_field=text_field,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        report_to="wandb",
        run_name=args.run_name,
        disable_tqdm=(rank != 0),
    )

    # ── Trainer ────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training …")
    result = trainer.train()
    logger.info("Training metrics: %s", result.metrics)

    if rank == 0:
        logger.info("Saving model to %s", args.output_dir)
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("Model saved successfully")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
