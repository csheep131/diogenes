#!/bin/bash
#
# SUA (Staleness/Unknown/Ambiguity) Training Script for Diogenes
# Phase 3.5: Specialized fine-tuning for epistemic boundary detection
#
# Usage:
#   ./run_sua_training.sh [OPTIONS]
#
# Options:
#   --dpo_checkpoint    Path to DPO checkpoint (required)
#   --output_dir        Output directory (default: models/sua_3b_test)
#   --dataset           SUA dataset path (default: datasets/sua_dataset.jsonl)
#   --eval_dataset      Eval holdout dataset (default: datasets/sua_eval_holdout.jsonl)
#   --epochs            Number of epochs (default: 2)
#   --lr                Learning rate (default: 5e-6)
#   --lora_r            LoRA rank (default: 16)
#   --help              Show this help message
#
# Example:
#   ./run_sua_training.sh \
#     --dpo_checkpoint models/dpo_3b_test/final_checkpoint \
#     --output_dir models/sua_3b_test \
#     --epochs 2
#

set -e

# Default values
DPO_CHECKPOINT=""
OUTPUT_DIR="models/sua_3b_test"
DATASET="datasets/sua_dataset.jsonl"
EVAL_DATASET="datasets/sua_eval_holdout.jsonl"
EPOCHS=2
LR="5e-6"
LORA_R=16
LORA_ALPHA=32
BATCH_SIZE=2
GRADIENT_ACCUM=8
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}  ${GREEN}Diogenes Phase 3.5 - SUA Training Script${NC}              ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dpo_checkpoint    Path to DPO checkpoint (required)"
    echo "  --output_dir        Output directory (default: models/sua_3b_test)"
    echo "  --dataset           SUA dataset path (default: datasets/sua_dataset.jsonl)"
    echo "  --eval_dataset      Eval holdout dataset (default: datasets/sua_eval_holdout.jsonl)"
    echo "  --epochs            Number of epochs (default: 2)"
    echo "  --lr                Learning rate (default: 5e-6)"
    echo "  --lora_r            LoRA rank (default: 16)"
    echo "  --lora_alpha        LoRA alpha (default: 32)"
    echo "  --batch_size        Batch size (default: 2)"
    echo "  --gradient_accum    Gradient accumulation steps (default: 8)"
    echo "  --model_name        Base model name (default: Qwen/Qwen2.5-3B-Instruct)"
    echo "  --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --dpo_checkpoint models/dpo_3b_test/final_checkpoint"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dpo_checkpoint)
            DPO_CHECKPOINT="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --eval_dataset)
            EVAL_DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        --lora_alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accum)
            GRADIENT_ACCUM="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$DPO_CHECKPOINT" ]; then
    echo -e "${RED}❌ Error: --dpo_checkpoint is required${NC}"
    echo ""
    echo "Usage: $0 --dpo_checkpoint <path> [OPTIONS]"
    echo ""
    echo "Example:"
    echo "  $0 --dpo_checkpoint models/dpo_3b_test/final_checkpoint"
    exit 1
fi

# Check if DPO checkpoint exists
if [ ! -d "$DPO_CHECKPOINT" ]; then
    echo -e "${RED}❌ Error: DPO checkpoint not found: $DPO_CHECKPOINT${NC}"
    echo ""
    echo "Please ensure Phase 3 (DPO Training) is completed first."
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
    echo -e "${YELLOW}⚠️  Warning: SUA dataset not found: $DATASET${NC}"
    echo ""
    echo "Generating SUA dataset first..."
    echo ""

    # Generate SUA dataset
    python3 src/diogenes/dataset_generator.py \
        --split sua \
        --staleness 8000 \
        --unknown 10000 \
        --ambiguity 7000

    echo ""
    echo -e "${GREEN}✅ SUA dataset generated${NC}"
fi

# Check if eval dataset exists
if [ ! -f "$EVAL_DATASET" ] && [ "$EVAL_DATASET" != "None" ]; then
    echo -e "${YELLOW}⚠️  Warning: Eval dataset not found: $EVAL_DATASET${NC}"
    echo ""
    echo "Generating SUA eval holdout dataset..."
    echo ""

    # Generate eval holdout
    python3 src/diogenes/dataset_generator.py \
        --split sua_eval \
        --staleness 1000 \
        --unknown 1500 \
        --ambiguity 1000

    echo ""
    echo -e "${GREEN}✅ Eval dataset generated${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${GREEN}🚀 Diogenes Phase 3.5 - SUA Training${NC}                  ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  DPO Checkpoint:     $DPO_CHECKPOINT"
echo "  Output Directory:   $OUTPUT_DIR"
echo "  Dataset:            $DATASET"
echo "  Eval Dataset:       $EVAL_DATASET"
echo "  Epochs:             $EPOCHS"
echo "  Learning Rate:      $LR"
echo "  LoRA Rank:          $LORA_R"
echo "  LoRA Alpha:         $LORA_ALPHA"
echo "  Batch Size:         $BATCH_SIZE"
echo "  Gradient Accum:     $GRADIENT_ACCUM"
echo "  Model:              $MODEL_NAME"
echo ""

# Check GPU memory
echo -e "${YELLOW}Checking GPU memory...${NC}"
if command -v nvidia-smi &> /dev/null; then
    VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    VRAM_FREE=$((VRAM_TOTAL - VRAM_USED))

    echo "  VRAM Total:  ${VRAM_TOTAL} MB"
    echo "  VRAM Used:   ${VRAM_USED} MB"
    echo "  VRAM Free:   ${VRAM_FREE} MB"

    if [ $VRAM_FREE -lt 7000 ]; then
        echo ""
        echo -e "${RED}⚠️  Warning: Less than 7GB VRAM free. Training may fail.${NC}"
        echo "   Consider reducing batch size or closing other GPU processes."
        echo ""
    fi
else
    echo "  nvidia-smi not available, skipping GPU check"
fi

echo ""
echo -e "${GREEN}Starting SUA training...${NC}"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set environment variables
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0

# Run training
python3 src/diogenes/train_sua.py \
    --model_name "$MODEL_NAME" \
    --dpo_checkpoint "$DPO_CHECKPOINT" \
    --dataset_path "$DATASET" \
    --eval_dataset "$EVAL_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUM" \
    --logging_steps 50 \
    --save_steps 2000 \
    --eval_steps 1000 \
    --early_stopping true \
    --early_stopping_patience 2

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  ${GREEN}✅ SUA Training Completed Successfully!${NC}                ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Final checkpoint: $OUTPUT_DIR/final_checkpoint"
    echo ""

    # Run Pass@1 Protection Check
    echo -e "${YELLOW}Running Pass@1 Protection Check...${NC}"
    echo ""

    if [ -f "scripts/pass1_protection_check.py" ]; then
        python3 scripts/pass1_protection_check.py \
            --model-path "$OUTPUT_DIR/final_checkpoint" \
            --baseline-pass-at-1 0.75
    else
        echo "⚠️  Pass@1 protection script not found, skipping..."
    fi

    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "  1. Review Pass@1 Protection results"
    echo "  2. Evaluate SUA metrics: python3 src/diogenes/eval_metrics.py --sua"
    echo "  3. Proceed to Phase 4 (Calibration) if Pass@1 is stable"
else
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}  ${RED}❌ SUA Training Failed${NC}                                    ${RED}║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Please check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce batch size to 1, increase gradient_accum to 16"
    echo "  - Pass@1 regression: Reduce learning rate to 1e-6 or epochs to 1"
    echo "  - Dataset not found: Run dataset_generator.py --split sua first"
    exit 1
fi
