#!/bin/bash
# DPO Training Script für Phase 3
# Startet DPO-Training auf Basis des SFT-Checkpoints
# Hardware: RTX 3050 (8GB VRAM)

set -e

# Konfiguration
SFT_CHECKPOINT="${1:-models/sft_3b_test/final_checkpoint}"
OUTPUT_DIR="${2:-models/dpo_3b_test}"
DATASET_PATH="${3:-datasets/dpo_dataset.jsonl}"
EPOCHS="${4:-2}"

echo "============================================================"
echo "Diogenes DPO Training - Phase 3"
echo "============================================================"
echo ""
echo "Konfiguration:"
echo "  SFT Checkpoint: $SFT_CHECKPOINT"
echo "  Output Dir:     $OUTPUT_DIR"
echo "  Dataset:        $DATASET_PATH"
echo "  Epochen:        $EPOCHS"
echo ""

# Prüfe ob SFT-Checkpoint existiert
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "❌ FEHLER: SFT-Checkpoint nicht gefunden: $SFT_CHECKPOINT"
    echo ""
    echo "Bitte warte bis das SFT-Training abgeschlossen ist."
    echo "SFT-Training Status prüfen:"
    echo "  ps aux | grep train_sft | grep -v grep"
    echo "  tail -f /tmp/sft_train.log"
    exit 1
fi

# Prüfe ob DPO-Dataset existiert
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ FEHLER: DPO-Dataset nicht gefunden: $DATASET_PATH"
    echo ""
    echo "DPO-Dataset generieren:"
    echo "  python3 src/diogenes/dataset_generator.py --dpo-pairs 60000"
    exit 1
fi

# DPO-Audit durchführen (optional, aber empfohlen)
echo "🔍 DPO-Audit durchführen..."
python3 src/diogenes/dpo_audit.py --dataset-path "$DATASET_PATH" --output-path dpo_audit_report.json

if [ $? -ne 0 ]; then
    echo "⚠️  DPO-Audit hat Warnungen gefunden. Siehe dpo_audit_report.json"
    echo "Fortsetzen? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Abgebrochen."
        exit 1
    fi
fi

echo ""
echo "🚀 Starte DPO-Training..."
echo ""

# Aktiviere virtuelle Umgebung und starte Training
source .venv/bin/activate
export WANDB_DISABLED=true

python3 src/diogenes/train_dpo.py \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --sft-model-path "$SFT_CHECKPOINT" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --num-train-epochs "$EPOCHS" \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 5e-7 \
    --beta 0.2 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --logging-steps 50 \
    --save-steps 1000 \
    --seed 42

echo ""
echo "============================================================"
echo "DPO Training abgeschlossen!"
echo "============================================================"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""
echo "Nächste Schritte:"
echo "  1. Evaluation durchführen:"
echo "     python3 src/diogenes/eval_metrics.py --model_path $OUTPUT_DIR"
echo ""
echo "  2. Pass@1 Protection Check:"
echo "     python3 src/diogenes/pass1_protection.py --model_path $OUTPUT_DIR"
echo ""
