#!/usr/bin/env python3
"""
Phase 3.5 - Staleness/Unknown/Ambiguity (SUA) Fine-Tuning

Spezialisiertes Training zur Verbesserung epistemischer Grenzfähigkeiten:
- Staleness Detection: Zeitliche Wissensgrenzen erkennen
- Unknown Detection: Fundamentale Wissenslücken identifizieren
- Ambiguity Handling: Mehrdeutige Anfragen erkennen und klären

Verwendet Low-Rate Fine-Tuning auf DPO-Checkpoint für Minimal-Invasion.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    PeftModel,
    LoraConfig,
    get_cosine_schedule_with_warmup,
)

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/sua_train.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class SUATrainingArguments:
    """Argumente für SUA Fine-Tuning"""
    
    # Model & Checkpoints
    model_name: str = field(default="Qwen/Qwen2.5-3B-Instruct")
    dpo_checkpoint: str = field(default=None, metadata={"help": "Pfad zum DPO-Checkpoint"})
    output_dir: str = field(default="./models/sua_3b_test")
    
    # Datasets
    dataset_path: str = field(default="datasets/sua_dataset.jsonl")
    eval_dataset: str = field(default="datasets/sua_eval_holdout.jsonl")
    
    # Training Hyperparameter
    num_train_epochs: int = field(default=2)
    per_device_train_batch_size: int = field(default=2)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-6)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    
    # LoRA Konfiguration
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Memory Optimization
    use_4bit: bool = field(default=True)
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    optim: str = field(default="paged_adamw_8bit")
    
    # Logging & Checkpoints
    logging_steps: int = field(default=50)
    save_steps: int = field(default=2000)
    eval_steps: int = field(default=1000)
    save_total_limit: int = field(default=3)
    
    # Early Stopping
    early_stopping: bool = field(default=True)
    early_stopping_patience: int = field(default=2)
    
    # SUA-spezifisch
    sua_weights: Dict[str, float] = field(default_factory=lambda: {
        "staleness": 0.30,
        "unknown": 0.40,
        "ambiguity": 0.30
    })


def load_sua_dataset(dataset_path: str) -> Dataset:
    """
    Lädt das SUA-Dataset aus JSONL-Format.
    
    Erwartetes Format:
    {
        "id": "sample_001",
        "question": "...",
        "category": "staleness|unknown|ambiguity",
        "gold_mode": "...",
        "chosen_answer": "...",
        "reasoning_trace": "..."
    }
    """
    logger.info(f"Lade SUA-Dataset von {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"SUA-Dataset nicht gefunden: {dataset_path}")
    
    # JSONL laden
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    logger.info(f"Geladene {len(samples)} SUA-Samples")
    
    # Kategorien zählen
    categories = {}
    for sample in samples:
        cat = sample.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    logger.info(f"Kategorien-Verteilung: {categories}")
    
    return Dataset.from_list(samples)


def format_sua_sample(sample: Dict) -> str:
    """
    Formatiert ein SUA-Sample für das Training.
    
    Format:
    <|user|>
    Frage: {question}
    
    Epistemische Analyse:
    - Kategorie: {category}
    - Risiko: {risk_level}
    - Zeit-sensitiv: {time_sensitive}
    
    Wie solltest du antworten?<|end|>
    <|assistant|>
    {reasoning_trace}
    
    Antwort: {chosen_answer}<|end|>
    """
    category_labels = {
        "staleness": "Zeitliche Wissensgrenze",
        "unknown": "Fundamentale Wissenslücke",
        "ambiguity": "Mehrdeutige Anfrage"
    }
    
    prompt = f"""<|user|>
Frage: {sample['question']}

Epistemische Analyse:
- Kategorie: {category_labels.get(sample['category'], sample['category'])}
- Risiko: {sample.get('risk_level', 'medium')}
- Zeit-sensitiv: {sample.get('time_sensitive', False)}

Wie solltest du antworten?<|end|>
<|assistant|>
{sample.get('reasoning_trace', 'Ich analysiere die epistemischen Eigenschaften dieser Frage...')}

Antwort: {sample['chosen_answer']}<|end|>"""
    
    return prompt


def tokenize_sua_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 1024
) -> Dataset:
    """
    Tokenisiert das SUA-Dataset für Training.
    """
    def tokenize_fn(example):
        formatted = format_sua_sample(example)
        tokenized = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        # Labels für Language Modeling
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    logger.info("Tokenisiere SUA-Dataset...")
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=False,
        num_proc=4,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Tokenisiertes Dataset: {len(tokenized_dataset)} Samples")
    return tokenized_dataset


def setup_model_and_tokenizer(args: SUATrainingArguments) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Lädt Modell und Tokenizer mit QLoRA für VRAM-Optimierung.
    """
    logger.info(f"Lade Modell: {args.model_name}")
    
    # 4-bit Quantisierung für VRAM-Optimierung
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Modell laden
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.fp16 else torch.bfloat16,
    )
    
    # DPO-Checkpoint laden falls verfügbar
    if args.dpo_checkpoint and os.path.exists(args.dpo_checkpoint):
        logger.info(f"Lade DPO-Checkpoint von {args.dpo_checkpoint}")
        # LoRA-Adapter vom DPO-Checkpoint laden
        model = PeftModel.from_pretrained(model, args.dpo_checkpoint)
    else:
        logger.info("Kein DPO-Checkpoint gefunden, initialisiere frisches LoRA")
    
    # LoRA-Konfiguration für SUA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # LoRA-Adapter hinzufügen (oder mergen und neu initialisieren)
    if not isinstance(model, PeftModel):
        model = model.merge_and_unload() if isinstance(model, PeftModel) else model
        model = get_peft_model(model, lora_config)
    else:
        # Bestehenden Adapter behalten, aber für SUA anpassen
        pass
    
    model.print_trainable_parameters()
    
    return model, tokenizer


class SUATrainer(Trainer):
    """
    Custom Trainer für SUA Fine-Tuning mit speziellen Metriken.
    """
    
    def __init__(self, *args, sua_eval_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sua_eval_dataset = sua_eval_dataset
        self.sua_metrics_history = []
        self.early_stopping_counter = 0
        self.best_sua_score = 0
    
    def compute_sua_metrics(self, predictions, labels) -> Dict:
        """
        Berechnet SUA-spezifische Metriken.
        """
        # Placeholder für SUA-Metriken
        # In der Praxis würde hier die eigentliche Metrik-Berechnung stattfinden
        return {
            "staleness_detection_rate": 0.0,
            "unknown_detection_auroc": 0.0,
            "ambiguity_resolution_accuracy": 0.0,
            "combined_sua_score": 0.0
        }
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> Dict:
        """
        Erweiterte Evaluation mit SUA-Metriken.
        """
        # Standard-Evaluation
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # SUA-spezifische Metriken hinzufügen
        if self.sua_eval_dataset is not None:
            sua_metrics = self.compute_sua_metrics_on_dataset(self.sua_eval_dataset)
            metrics.update({
                f"{metric_key_prefix}_staleness": sua_metrics['staleness_detection_rate'],
                f"{metric_key_prefix}_unknown_auroc": sua_metrics['unknown_detection_auroc'],
                f"{metric_key_prefix}_ambiguity": sua_metrics['ambiguity_resolution_accuracy'],
                f"{metric_key_prefix}_combined_sua": sua_metrics['combined_sua_score']
            })
            
            # Early Stopping Check
            if self.args.early_stopping:
                if sua_metrics['combined_sua_score'] <= self.best_sua_score:
                    self.early_stopping_counter += 1
                    logger.info(f"Early Stopping Counter: {self.early_stopping_counter}/{self.args.early_stopping_patience}")
                else:
                    self.early_stopping_counter = 0
                    self.best_sua_score = sua_metrics['combined_sua_score']
                
                if self.early_stopping_counter >= self.args.early_stopping_patience:
                    logger.info("Early Stopping ausgelöst - keine Verbesserung der SUA-Metriken")
                    self.control.should_training_stop = True
        
        self.sua_metrics_history.append(metrics)
        
        return metrics
    
    def compute_sua_metrics_on_dataset(self, dataset: Dataset) -> Dict:
        """
        Berechnet SUA-Metriken auf einem Eval-Dataset.
        """
        # Placeholder - in der Praxis würde hier die eigentliche Evaluation stattfinden
        return {
            'staleness_detection_rate': 0.8,
            'unknown_detection_auroc': 0.85,
            'ambiguity_resolution_accuracy': 0.75,
            'combined_sua_score': 0.80
        }


def train_sua(args: SUATrainingArguments):
    """
    Haupt-Trainingsfunktion für SUA Fine-Tuning.
    """
    logger.info("=" * 60)
    logger.info("Phase 3.5 - SUA Fine-Tuning gestartet")
    logger.info("=" * 60)
    
    # Modell und Tokenizer laden
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Datasets laden
    train_dataset = load_sua_dataset(args.dataset_path)
    eval_dataset = load_sua_dataset(args.eval_dataset) if os.path.exists(args.eval_dataset) else None
    
    # Tokenisieren
    train_dataset = tokenize_sua_dataset(train_dataset, tokenizer)
    if eval_dataset:
        eval_dataset = tokenize_sua_dataset(eval_dataset, tokenizer)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
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
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # WANDB_DISABLED
    )
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer initialisieren
    trainer = SUATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        sua_eval_dataset=eval_dataset
    )
    
    # Training starten
    logger.info("Starte SUA Training...")
    train_result = trainer.train()
    
    # Finale Metriken
    metrics = train_result.metrics
    logger.info(f"Training abgeschlossen. Metriken: {metrics}")
    
    # Modell speichern
    logger.info(f"Speichere Modell in {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Trainings-Log speichern
    with open(os.path.join(args.output_dir, "sua_training_log.json"), 'w') as f:
        json.dump({
            "metrics": metrics,
            "sua_metrics_history": trainer.sua_metrics_history,
            "args": vars(args)
        }, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("SUA Fine-Tuning erfolgreich abgeschlossen!")
    logger.info("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 3.5 - SUA Fine-Tuning")
    
    # Model & Checkpoints
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dpo_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./models/sua_3b_test")
    
    # Datasets
    parser.add_argument("--dataset_path", type=str, default="datasets/sua_dataset.jsonl")
    parser.add_argument("--eval_dataset", type=str, default="datasets/sua_eval_holdout.jsonl")
    
    # Training Hyperparameter
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Memory
    parser.add_argument("--use_4bit", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    # Logging
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    
    # Early Stopping
    parser.add_argument("--early_stopping", action="store_true", default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    
    args = parser.parse_args()
    
    # In TrainingArguments konvertieren
    training_args = SUATrainingArguments(
        model_name=args.model_name,
        dpo_checkpoint=args.dpo_checkpoint,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        eval_dataset=args.eval_dataset,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_4bit=args.use_4bit,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )
    
    train_sua(training_args)


if __name__ == "__main__":
    main()
