#!/usr/bin/env python3
"""DPO Training Script for Diogenes.

Direct Preference Optimization for hallucination reduction and
epistemic honesty reinforcement. Trains on ~60k preference pairs.
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
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class DPOTrainingArguments:
    """DPO Training configuration."""
    
    # Model
    model_name: str = field(default="Qwen/Qwen3-0.6B")
    sft_model_path: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=True)
    
    # Dataset
    dataset_path: str = field(default="./datasets/dpo_dataset.jsonl")
    max_seq_length: int = field(default=2048)
    max_prompt_length: int = field(default=512)
    
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
    
    # DPO Specific
    beta: float = field(default=0.1)  # Temperature parameter for DPO
    label_smoothing: float = field(default=0.0)
    loss_type: str = field(default="sigmoid")  # sigmoid, hinge, ipo, kto_pair
    
    # Training
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-7)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    
    # Optimization
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    optim: str = field(default="paged_adamw_8bit")
    
    # Logging & Checkpointing
    output_dir: str = field(default="./models/dpo_output")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    report_to: str = field(default="wandb")
    
    # Misc
    seed: int = field(default=42)
    resume_from_checkpoint: Optional[str] = field(default=None)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_dpo_sample(
    example: dict,
    tokenizer,
    max_length: int = 2048,
    max_prompt_length: int = 512,
) -> dict:
    """Format DPO sample for training.
    
    DPO requires:
    - prompt: The user question
    - chosen: The preferred response
    - rejected: The less preferred response
    """
    system_prompt = (
        "You are Diogenes, an epistemically reliable AI assistant. "
        "You recognize the limits of your knowledge and respond appropriately. "
        "You are honest about uncertainty and avoid hallucinations."
    )
    
    # Format prompt (user question only for DPO)
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["question"]},
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = (
            f"<system>{system_prompt}</system>\n"
            f"<user>{example['question']}</user>\n"
            f"<assistant>"
        )
    
    # Format chosen response
    chosen_messages = prompt_messages + [
        {"role": "assistant", "content": example["chosen_answer"]}
    ]
    try:
        chosen = tokenizer.apply_chat_template(
            chosen_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        chosen = prompt + example["chosen_answer"]
    
    # Format rejected response
    rejected_messages = prompt_messages + [
        {"role": "assistant", "content": example["rejected_answer"]}
    ]
    try:
        rejected = tokenizer.apply_chat_template(
            rejected_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        rejected = prompt + example["rejected_answer"]
    
    # Tokenize
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_prompt_length,
        padding=False,
        return_tensors=None,
    )
    
    chosen_tokens = tokenizer(
        chosen,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    rejected_tokens = tokenizer(
        rejected,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    return {
        "prompt_input_ids": prompt_tokens["input_ids"],
        "prompt_attention_mask": prompt_tokens["attention_mask"],
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
    }


def create_qlora_config(args: DPOTrainingArguments) -> tuple[BitsAndBytesConfig, LoraConfig]:
    """Create QLoRA configuration for memory-efficient training."""
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    
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


def load_and_prepare_dataset(
    dataset_path: str,
    tokenizer,
    max_length: int = 2048,
    max_prompt_length: int = 512,
    split: str = "train",
):
    """Load and prepare DPO dataset."""
    logger.info(f"Loading dataset from: {dataset_path}")
    
    dataset = load_dataset("json", data_files=dataset_path, split=split)
    
    logger.info(f"Dataset loaded: {len(dataset)} preference pairs")
    
    def format_fn(example):
        return format_dpo_sample(example, tokenizer, max_length, max_prompt_length)
    
    dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
    
    # Log statistics
    prompt_lengths = [len(x["prompt_input_ids"]) for x in dataset]
    chosen_lengths = [len(x["chosen_input_ids"]) for x in dataset]
    rejected_lengths = [len(x["rejected_input_ids"]) for x in dataset]
    
    logger.info(f"Average prompt length: {sum(prompt_lengths) / len(prompt_lengths):.1f}")
    logger.info(f"Average chosen length: {sum(chosen_lengths) / len(chosen_lengths):.1f}")
    logger.info(f"Average rejected length: {sum(rejected_lengths) / len(rejected_lengths):.1f}")
    
    return dataset


class DPOTrainer:
    """Custom DPO Trainer implementing Direct Preference Optimization.
    
    DPO Loss:
    L_DPO = -log σ(β · [log(P(chosen)/P(rejected)) - log(π_ref(chosen)/π_ref(rejected))])
    
    Where:
    - σ: Sigmoid function
    - β: Temperature parameter (controls strength of preference)
    - π_ref: Reference model (frozen SFT model)
    """
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        args: DPOTrainingArguments,
        train_dataset,
        data_collator=None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator or self._default_data_collator
        
        # Beta parameter for DPO
        self.beta = args.beta
        
        logger.info(f"DPO Trainer initialized with beta={self.beta}")
        logger.info(f"Loss type: {args.loss_type}")
        logger.info(f"Label smoothing: {args.label_smoothing}")
    
    def _default_data_collator(self, features):
        """Simple data collator for DPO."""
        batch = {}
        for key in features[0].keys():
            batch[key] = [f[key] for f in features]
        return batch
    
    def concatenated_forward(self, model, batch):
        """Forward pass for both chosen and rejected responses."""
        # Concatenate chosen and rejected for efficient forward pass
        all_input_ids = batch["chosen_input_ids"] + batch["rejected_input_ids"]
        all_attention_mask = batch["chosen_attention_mask"] + batch["rejected_attention_mask"]
        
        # Pad sequences
        max_length = max(len(ids) for ids in all_input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(all_input_ids, all_attention_mask):
            padding_length = max_length - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
        
        # Convert to tensors
        input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(padded_attention_mask, dtype=torch.long)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Compute log probabilities
        logits = outputs.logits
        log_probs = self._get_log_probs(logits, input_ids, attention_mask)
        
        # Split into chosen and rejected
        batch_size = len(batch["chosen_input_ids"])
        chosen_log_probs = log_probs[:batch_size]
        rejected_log_probs = log_probs[batch_size:]
        
        return chosen_log_probs, rejected_log_probs
    
    def _get_log_probs(self, logits, input_ids, attention_mask):
        """Compute log probabilities for generated tokens."""
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        # Compute log softmax
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        batch_size = shift_logits.shape[0]
        log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Mask padding tokens
        log_probs = log_probs * shift_mask
        
        # Sum over sequence length and normalize
        log_probs = log_probs.sum(dim=1) / shift_mask.sum(dim=1)
        
        return log_probs
    
    def dpo_loss(self, chosen_log_probs, rejected_log_probs, ref_chosen_log_probs, ref_rejected_log_probs):
        """Compute DPO loss."""
        # Compute log probability ratios
        pi_logratios = chosen_log_probs - rejected_log_probs
        ref_logratios = ref_chosen_log_probs - ref_rejected_log_probs
        
        # Compute logits for sigmoid
        logits = pi_logratios - ref_logratios
        
        # Apply DPO loss based on type
        if self.args.loss_type == "sigmoid":
            # Standard DPO loss: -log(σ(β * logits))
            losses = -torch.log(torch.sigmoid(self.beta * logits) + 1e-7)
        elif self.args.loss_type == "hinge":
            # Hinge loss variant
            losses = torch.relu(1 - self.beta * logits)
        elif self.args.loss_type == "ipo":
            # Identity Preference Optimization
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(f"Unknown loss type: {self.args.loss_type}")
        
        # Apply label smoothing if specified
        if self.args.label_smoothing > 0:
            losses = losses * (1 - self.args.label_smoothing) + self.args.label_smoothing / 2
        
        return losses.mean()
    
    def train(self):
        """Main DPO training loop."""
        logger.info("Starting DPO training...")
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        
        # Setup learning rate scheduler
        num_training_steps = (
            len(self.train_dataset)
            // (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)
            * self.args.num_train_epochs
        )
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        
        if self.args.lr_scheduler_type == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        
        # Training loop
        self.model.train()
        self.ref_model.eval()
        
        global_step = 0
        total_loss = 0
        
        # Create data loader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )
        
        for epoch in range(int(self.args.num_train_epochs)):
            logger.info(f"Epoch {epoch + 1}/{int(self.args.num_train_epochs)}")
            
            for batch_idx, batch in enumerate(train_loader):
                # Forward pass with policy model
                chosen_log_probs, rejected_log_probs = self.concatenated_forward(
                    self.model, batch
                )
                
                # Forward pass with reference model (no gradients)
                with torch.no_grad():
                    ref_chosen_log_probs, ref_rejected_log_probs = self.concatenated_forward(
                        self.ref_model, batch
                    )
                
                # Compute DPO loss
                loss = self.dpo_loss(
                    chosen_log_probs,
                    rejected_log_probs,
                    ref_chosen_log_probs,
                    ref_rejected_log_probs,
                )
                
                # Backward pass
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item() * self.args.gradient_accumulation_steps
                
                # Update weights
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % self.args.logging_steps == 0:
                        avg_loss = total_loss / self.args.logging_steps
                        logger.info(f"Step {global_step}: loss={avg_loss:.4f}")
                        total_loss = 0
                    
                    # Save checkpoint
                    if global_step % self.args.save_steps == 0:
                        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{global_step}"
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(str(checkpoint_dir))
                        self.tokenizer.save_pretrained(str(checkpoint_dir))
                        logger.info(f"Saved checkpoint: {checkpoint_dir}")
        
        # Save final model
        final_dir = Path(self.args.output_dir) / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Saved final model: {final_dir}")
        
        logger.info("DPO training complete!")


def train(args: DPOTrainingArguments):
    """Main DPO training function."""
    logger.info("=" * 60)
    logger.info("Starting DPO Training")
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
    
    # Use SFT model if provided, otherwise load from base
    model_path = args.sft_model_path or args.model_name
    
    if args.use_qlora:
        bnb_config, lora_config = create_qlora_config(args)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            torch_dtype=getattr(torch, args.bnb_4bit_compute_dtype),
        )
        
        model = prepare_model_for_kbit_training(model)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
        )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load reference model (frozen SFT model)
    logger.info("Loading reference model (frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    )
    ref_model.eval()
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Load dataset
    dataset = load_and_prepare_dataset(
        args.dataset_path,
        tokenizer,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    # Initialize DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
    )
    
    # Train
    trainer.train()
    
    logger.info("=" * 60)
    logger.info("DPO Training Complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DPO Training for Diogenes")
    
    # Model
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--sft-model-path",
        type=str,
        default=None,
        help="Path to SFT-finetuned model (used as policy and reference)",
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
        default="./datasets/dpo_dataset.jsonl",
        help="Path to DPO dataset",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=512,
        help="Maximum prompt length",
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
    
    # DPO Specific
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature parameter",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo", "kto_pair"],
        help="DPO loss type",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor",
    )
    
    # Training
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=2,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
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
        default="./models/dpo_output",
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
    training_args = DPOTrainingArguments(
        model_name=args.model_name,
        sft_model_path=args.sft_model_path,
        trust_remote_code=args.trust_remote_code,
        dataset_path=args.dataset_path,
        max_seq_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_qlora=not args.no_qlora,
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
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
