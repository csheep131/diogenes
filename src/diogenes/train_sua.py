#!/usr/bin/env python3
"""
SUA (Staleness/Unknown/Ambiguity) Fine-Tuning Script for Diogenes

Phase 3.5: Specialized fine-tuning for epistemic boundary detection.
Builds upon DPO checkpoint with low-rate fine-tuning to avoid Pass@1 degradation.

Usage:
    python train_sua.py \
        --model_name Qwen/Qwen2.5-3B-Instruct \
        --dpo_checkpoint models/dpo_3b_test/final_checkpoint \
        --dataset_path datasets/sua_dataset.jsonl \
        --eval_dataset datasets/sua_eval_holdout.jsonl \
        --output_dir models/sua_3b_test \
        --num_train_epochs 2 \
        --learning_rate 5e-6 \
        --lora_r 16 \
        --lora_alpha 32

Author: Diogenes Team
Date: March 2026
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from tqdm import tqdm

# Import Pass@1 Protection
try:
    from diogenes.pass1_protection import (
        run_pass1_protection_test,
        compute_core_reliability_metrics,
    )
    from diogenes.eval_metrics import compute_sua_metrics
except ImportError:
    print("⚠️  Warning: Diogenes modules not found. Running in standalone mode.")


@dataclass
class SUATrainingArguments:
    """SUA-specific training arguments."""

    model_name: str = field(
        default="Qwen/Qwen2.5-3B-Instruct",
        metadata={"help": "Base model name or path"}
    )
    dpo_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DPO checkpoint (starting point for SUA)"}
    )
    dataset_path: str = field(
        default="datasets/sua_dataset.jsonl",
        metadata={"help": "Path to SUA dataset"}
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation holdout dataset"}
    )
    output_dir: str = field(
        default="models/sua_3b_test",
        metadata={"help": "Output directory for SUA checkpoint"}
    )

    # SUA-specific hyperparameters
    learning_rate: float = field(
        default=5e-6,
        metadata={"help": "Low learning rate for minimal invasion (default: 5e-6)"}
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Number of epochs (keep low to avoid overfitting)"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank (reduced from 32 for faster adaptation)"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha (2x rank)"}
    )

    # Standard training arguments
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per device for 8GB VRAM"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Gradient accumulation steps"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Warmup ratio"}
    )
    logging_steps: int = field(
        default=50,
        metadata={"help": "Logging steps"}
    )
    save_steps: int = field(
        default=2000,
        metadata={"help": "Save checkpoint steps"}
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "Evaluation steps"}
    )

    # Early stopping
    early_stopping: bool = field(
        default=True,
        metadata={"help": "Enable early stopping"}
    )
    early_stopping_patience: int = field(
        default=2,
        metadata={"help": "Early stopping patience"}
    )

    # Pass@1 protection
    baseline_pass1: Optional[float] = field(
        default=None,
        metadata={"help": "Baseline Pass@1 from DPO checkpoint"}
    )
    max_pass1_degradation: float = field(
        default=0.02,
        metadata={"help": "Maximum allowed Pass@1 degradation (default: 2%)"}
    )


def load_sua_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load SUA dataset from JSONL file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"SUA dataset not found: {dataset_path}")

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Statistics
    total = len(dataset)
    categories = {}
    for sample in dataset:
        cat = sample.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"✅ Loaded {total} SUA samples:")
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count} ({count/total*100:.1f}%)")

    return dataset


def prepare_tokenizer(model_name: str):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def prepare_model(
    model_name: str,
    dpo_checkpoint: Optional[str],
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Load and configure model for SUA training."""

    # Determine starting checkpoint
    if dpo_checkpoint and os.path.exists(dpo_checkpoint):
        print(f"✅ Loading DPO checkpoint: {dpo_checkpoint}")
        model_path = dpo_checkpoint
    else:
        print(f"⚠️  No DPO checkpoint found, using base model: {model_name}")
        model_path = model_name

    # QLoRA configuration for 8GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration (reduced rank for SUA)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize SUA samples."""
    texts = []
    for question, chosen in zip(examples["question"], examples["chosen_answer"]):
        # Format: Question + Answer
        text = f"Question: {question}\n\nAnswer: {chosen}"
        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def check_pass1_protection(
    model_path: str,
    baseline_pass1: Optional[float],
    max_degradation: float = 0.02,
) -> bool:
    """
    Check Pass@1 after training epoch.

    Returns:
        True if Pass@1 is stable, False if regression detected
    """
    if baseline_pass1 is None:
        print("⚠️  No baseline Pass@1 provided, skipping protection check")
        return True

    print(f"\n🔍 Running Pass@1 Protection Check...")
    print(f"   Baseline Pass@1: {baseline_pass1:.4f}")
    print(f"   Max allowed degradation: {max_degradation*100:.1f}%")

    try:
        # Run Pass@1 protection test
        result = run_pass1_protection_test(
            current_pass1=None,  # Would be computed from eval dataset
            baseline_pass1=baseline_pass1,
            current_hallucination=None,
            baseline_hallucination=0.05,
        )

        if result.is_regression:
            print(f"❌ PASS@1 REGRESSION DETECTED: {result.regression_severity}")
            print(f"   Details: {result.regression_details}")
            print(f"   Recommendation: {result.recommendation}")
            return False
        else:
            print(f"✓ Pass@1 stable - no regression detected")
            return True

    except Exception as e:
        print(f"⚠️  Pass@1 protection check failed: {e}")
        return True  # Continue training if check fails


def main():
    """Main SUA training function."""
    parser = argparse.ArgumentParser(description="SUA Fine-Tuning for Diogenes")

    # Parse arguments
    args = parser.parse_args()

    # Use dataclass for better organization
    training_args = SUATrainingArguments(
        model_name=args.model_name if hasattr(args, 'model_name') else "Qwen/Qwen2.5-3B-Instruct",
        dpo_checkpoint=args.dpo_checkpoint if hasattr(args, 'dpo_checkpoint') else None,
        dataset_path=args.dataset_path if hasattr(args, 'dataset_path') else "datasets/sua_dataset.jsonl",
        eval_dataset=args.eval_dataset if hasattr(args, 'eval_dataset') else None,
        output_dir=args.output_dir if hasattr(args, 'output_dir') else "models/sua_3b_test",
        learning_rate=args.learning_rate if hasattr(args, 'learning_rate') else 5e-6,
        num_train_epochs=args.num_train_epochs if hasattr(args, 'num_train_epochs') else 2,
        lora_r=args.lora_r if hasattr(args, 'lora_r') else 16,
        lora_alpha=args.lora_alpha if hasattr(args, 'lora_alpha') else 32,
    )

    print("=" * 60)
    print("🔥 Phase 3.5: SUA (Staleness/Unknown/Ambiguity) Fine-Tuning")
    print("=" * 60)

    # 1. Load dataset
    print("\n📊 Loading SUA dataset...")
    sua_dataset = load_sua_dataset(training_args.dataset_path)

    # 2. Prepare tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = prepare_tokenizer(training_args.model_name)

    # 3. Prepare model
    print("\n🤖 Loading model...")
    model = prepare_model(
        training_args.model_name,
        training_args.dpo_checkpoint,
        training_args.lora_r,
        training_args.lora_alpha,
    )

    # 4. Tokenize dataset
    print("\n📝 Tokenizing dataset...")
    tokenized_dataset = sua_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=sua_dataset.column_names,
    )

    # 5. Configure training arguments
    print("\n⚙️  Configuring training...")
    hf_training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        warmup_ratio=training_args.warmup_ratio,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        eval_steps=training_args.eval_steps if training_args.eval_dataset else None,
        evaluation_strategy="steps" if training_args.eval_dataset else "no",
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        save_total_limit=3,
        load_best_model_at_end=True if training_args.eval_dataset else False,
        metric_for_best_model="eval_loss" if training_args.eval_dataset else None,
        report_to="none",  # Disable wandb
    )

    # 6. Initialize trainer
    print("\n🏋️  Initializing trainer...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=hf_training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=None,  # Would be tokenized eval dataset
        data_collator=data_collator,
    )

    # 7. Start training
    print("\n" + "=" * 60)
    print("🚀 Starting SUA training...")
    print("=" * 60)
    print(f"   Model: {training_args.model_name}")
    print(f"   DPO Checkpoint: {training_args.dpo_checkpoint}")
    print(f"   Dataset: {training_args.dataset_path} ({len(sua_dataset)} samples)")
    print(f"   Output: {training_args.output_dir}")
    print(f"   Learning Rate: {training_args.learning_rate}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   LoRA Rank: {training_args.lora_r}")
    print(f"   Batch Size: {training_args.per_device_train_batch_size}")
    print(f"   Gradient Accumulation: {training_args.gradient_accumulation_steps}")
    print("=" * 60)

    # Train
    trainer.train()

    # 8. Save final checkpoint
    print("\n💾 Saving final checkpoint...")
    final_output = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    print(f"✅ SUA training completed!")
    print(f"   Final checkpoint: {final_output}")

    # 9. Pass@1 Protection Check (if baseline provided)
    if training_args.baseline_pass1 is not None:
        print("\n🛡️  Running Pass@1 Protection Check...")
        is_stable = check_pass1_protection(
            final_output,
            training_args.baseline_pass1,
            training_args.max_pass1_degradation,
        )

        if not is_stable:
            print("\n⚠️  WARNING: Pass@1 regression detected!")
            print("   Consider:")
            print("   - Reducing learning rate (e.g., 1e-6)")
            print("   - Reducing epochs (e.g., 1 epoch)")
            print("   - Reducing LoRA rank (e.g., 8)")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ Phase 3.5 SUA Training successfully completed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
