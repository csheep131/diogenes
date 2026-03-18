#!/usr/bin/env python3
"""SFT Training Script for Diogenes.

Supervised Fine-Tuning with LoRA/QLoRA for epistemic mode training.
Trains on ~80k samples with QLoRA (4-bit) quantization.
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SFTTrainingArguments:
    """SFT Training configuration."""
    
    # Model
    model_name: str = field(default="Qwen/Qwen3-0.6B")
    trust_remote_code: bool = field(default=True)
    
    # Dataset
    dataset_path: str = field(default="./datasets/sft_dataset.jsonl")
    max_seq_length: int = field(default=2048)
    
    # LoRA Configuration
    lora_rank: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # QLoRA Configuration
    use_qlora: bool = field(default=True)
    load_in_4bit: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: str = field(default="float16")
    bnb_4bit_use_double_quant: bool = field(default=True)
    
    # Training
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    
    # Optimization
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    optim: str = field(default="paged_adamw_8bit")
    
    # Logging & Checkpointing
    output_dir: str = field(default="./models/sft_output")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    report_to: str = field(default="none")  # Disabled wandb by default
    
    # Misc
    seed: int = field(default=42)
    resume_from_checkpoint: Optional[str] = field(default=None)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_sft_sample(example: dict, tokenizer, max_length: int = 2048) -> dict:
    """Format SFT sample for training.
    
    Creates instruction-following format:
    <system>
    {system_prompt}
    </system>
    <user>
    {question}
    </user>
    <assistant>
    {answer}
    </assistant>
    """
    system_prompt = (
        "You are Diogenes, an epistemically reliable AI assistant. "
        "You recognize the limits of your knowledge and respond appropriately. "
        "You are honest about uncertainty and avoid hallucinations."
    )
    
    # Build conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    
    # Apply chat template if available
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback to manual formatting
        text = (
            f"<system>{system_prompt}</system>\n"
            f"<user>{example['question']}</user>\n"
            f"<assistant>{example['answer']}</assistant>"
        )
    
    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # Create labels (copy of input_ids for supervised learning)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def create_qlora_config(args: SFTTrainingArguments) -> tuple[BitsAndBytesConfig, LoraConfig]:
    """Create QLoRA configuration for memory-efficient training."""
    
    # BitsAndBytes configuration for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    logger.info(f"QLoRA Config: rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info(f"Target modules: {args.target_modules}")
    
    return bnb_config, lora_config


def create_lora_config(args: SFTTrainingArguments) -> LoraConfig:
    """Create standard LoRA configuration (without quantization)."""
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )
    
    logger.info(f"LoRA Config: rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info(f"Target modules: {args.target_modules}")
    
    return lora_config


def load_and_prepare_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 2048,
    split: str = "train",
):
    """Load and prepare SFT dataset."""
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Load JSONL dataset
    dataset = load_dataset("json", data_files=dataset_path, split=split)
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Format samples
    def format_fn(example):
        return format_sft_sample(example, tokenizer, max_length)
    
    dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
    
    # Log dataset statistics
    lengths = [len(x["input_ids"]) for x in dataset]
    logger.info(f"Average sequence length: {sum(lengths) / len(lengths):.1f}")
    logger.info(f"Max sequence length: {max(lengths)}")
    logger.info(f"Min sequence length: {min(lengths)}")
    
    return dataset


def train(args: SFTTrainingArguments):
    """Main SFT training function."""
    logger.info("=" * 60)
    logger.info("Starting SFT Training")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    
    if args.use_qlora:
        # QLoRA: 4-bit quantization + LoRA
        bnb_config, lora_config = create_qlora_config(args)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            torch_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        # Standard LoRA (full precision or mixed precision)
        lora_config = create_lora_config(args)
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
        )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_and_prepare_dataset(
        args.dataset_path,
        tokenizer,
        max_length=args.max_seq_length,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save model
    logger.info("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save final checkpoint
    final_checkpoint = Path(args.output_dir) / "final_checkpoint"
    trainer.save_model(str(final_checkpoint))
    tokenizer.save_pretrained(str(final_checkpoint))
    
    logger.info("=" * 60)
    logger.info("SFT Training Complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Final checkpoint: {final_checkpoint}")
    logger.info("=" * 60)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Diogenes")
    
    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for model loading",
    )
    
    # Dataset
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/sft_dataset.jsonl",
        help="Path to SFT dataset",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    
    # LoRA
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--no-qlora",
        action="store_true",
        help="Disable QLoRA (use standard LoRA)",
    )
    
    # Training
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/sft_output",
        help="Output directory",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Create training arguments
    training_args = SFTTrainingArguments(
        model_name=args.model_name,
        trust_remote_code=args.trust_remote_code,
        dataset_path=args.dataset_path,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_qlora=not args.no_qlora,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    # Run training
    train(training_args)


if __name__ == "__main__":
    main()
