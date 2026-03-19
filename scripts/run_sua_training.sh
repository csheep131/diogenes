#!/bin/bash
# Phase 3.5 - SUA Training Script
# Staleness/Unknown/Ambiguity Fine-Tuning auf RTX 3050 (8GB)

set -e

# Standardwerte
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
DPO_CHECKPOINT=""
OUTPUT_DIR="./models/sua_3b_test"
DATASET_PATH="./datasets/sua_dataset.jsonl"
EVAL_DATASET="./datasets/sua_eval_holdout.jsonl"

# Hyperparameter
NUM_EPOCHS=2
BATCH_SIZE=2
GRADIENT_ACCUMULATION=8
LEARNING_RATE=5e-6
LORA_R=16
LORA_ALPHA=32

# Hilfe-Funktion
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Phase 3.5 - SUA Fine-Tuning Script"
    echo ""
    echo "Options:"
    echo "  --model_name NAME         Modell-Name (default: Qwen/Qwen2.5-3B-Instruct)"
    echo "  --dpo_checkpoint PATH     Pfad zum DPO-Checkpoint (required)"
    echo "  --output_dir PATH         Output-Verzeichnis (default: ./models/sua_3b_test)"
    echo "  --dataset_path PATH       SUA-Dataset Pfad (default: ./datasets/sua_dataset.jsonl)"
    echo "  --eval_dataset PATH       Eval-Dataset Pfad (default: ./datasets/sua_eval_holdout.jsonl)"
    echo "  --num_epochs N            Anzahl Epochen (default: 2)"
    echo "  --batch_size N            Batch Size (default: 2)"
    echo "  --learning_rate LR        Learning Rate (default: 5e-6)"
    echo "  --lora_r N                LoRA Rank (default: 16)"
    echo "  --help                    Diese Hilfe anzeigen"
    echo ""
    echo "Beispiel:"
    echo "  $0 --dpo_checkpoint models/dpo_3b_test/final_checkpoint"
    exit 1
}

# Argumente parsen
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dpo_checkpoint)
            DPO_CHECKPOINT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --eval_dataset)
            EVAL_DATASET="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unbekannte Option: $1"
            usage
            ;;
    esac
done

# Validierung
if [ -z "$DPO_CHECKPOINT" ]; then
    echo "Fehler: --dpo_checkpoint ist erforderlich"
    echo "Beispiel: --dpo_checkpoint models/dpo_3b_test/final_checkpoint"
    exit 1
fi

if [ ! -d "$DPO_CHECKPOINT" ]; then
    echo "Fehler: DPO-Checkpoint nicht gefunden: $DPO_CHECKPOINT"
    echo "Bitte stellen Sie sicher, dass Phase 3 (DPO Training) abgeschlossen ist."
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "Fehler: SUA-Dataset nicht gefunden: $DATASET_PATH"
    echo "Bitte generieren Sie zuerst das SUA-Dataset:"
    echo "  python src/diogenes/dataset_generator.py --split sua"
    exit 1
fi

# Umgebung vorbereiten
echo "=============================================="
echo "Phase 3.5 - SUA Fine-Tuning"
echo "=============================================="
echo ""
echo "Konfiguration:"
echo "  Modell: $MODEL_NAME"
echo "  DPO-Checkpoint: $DPO_CHECKPOINT"
echo "  Output: $OUTPUT_DIR"
echo "  Dataset: $DATASET_PATH"
echo "  Eval-Dataset: $EVAL_DATASET"
echo "  Epochen: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LoRA Rank: $LORA_R"
echo ""

# Virtuelle Umgebung aktivieren
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# WANDB deaktivieren
export WANDB_DISABLED=true

# Output-Verzeichnis erstellen
mkdir -p "$OUTPUT_DIR"

# Training starten
echo "Starte SUA Training..."
echo "Log wird nach /tmp/sua_train.log geschrieben"
echo ""

python3 src/diogenes/train_sua.py \
    --model_name "$MODEL_NAME" \
    --dpo_checkpoint "$DPO_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_path "$DATASET_PATH" \
    --eval_dataset "$EVAL_DATASET" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
    --learning_rate "$LEARNING_RATE" \
    --lora_r "$LORA_R" \
    --lora_alpha "$((LORA_R * 2))" \
    --logging_steps 50 \
    --save_steps 2000 \
    --eval_steps 1000 \
    --early_stopping \
    --early_stopping_patience 2

# Ergebnis prüfen
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✓ SUA Training erfolgreich abgeschlossen!"
    echo "=============================================="
    echo ""
    echo "Checkpoint gespeichert in: $OUTPUT_DIR"
    echo ""
    echo "Nächste Schritte:"
    echo "  1. Pass@1 Protection Check durchführen:"
    echo "     python3 scripts/pass1_protection_check.py --model-path $OUTPUT_DIR"
    echo ""
    echo "  2. SUA-Metriken evaluieren:"
    echo "     python3 src/diogenes/eval_metrics.py --model-path $OUTPUT_DIR --sua"
    echo ""
    echo "  3. Training-Log ansehen:"
    echo "     cat /tmp/sua_train.log"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "✗ SUA Training fehlgeschlagen!"
    echo "=============================================="
    echo ""
    echo "Bitte prüfen Sie das Log:"
    echo "  tail -f /tmp/sua_train.log"
    exit 1
fi
