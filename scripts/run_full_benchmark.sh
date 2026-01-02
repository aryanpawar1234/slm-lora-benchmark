#!/bin/bash
set -e

# Full Benchmark Suite for SLM LoRA Framework
# Runs comprehensive experiments across models, datasets, and configurations

echo "========================================"
echo "SLM LoRA Benchmark Suite"
echo "========================================"
echo ""

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
SCRIPTS_DIR="$PROJECT_DIR/scripts"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"
OUTPUTS_DIR="$PROJECT_DIR/outputs"
LOG_FILE="$OUTPUTS_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$OUTPUTS_DIR"

# Setup
echo "[1/4] Setting up environment..."
echo "Project directory: $PROJECT_DIR"
echo "Log file: $LOG_FILE"

# Step 1: Prepare Datasets
echo "[2/4] Preparing datasets..."
python "$SCRIPTS_DIR/prepare_datasets.py" 2>&1 | tee -a "$LOG_FILE"

# Step 2: Run Main Experiments
echo "[3/4] Running main benchmark experiments..."
echo "" | tee -a "$LOG_FILE"
echo "=== Baseline Configuration ===" | tee -a "$LOG_FILE"
python "$SCRIPTS_DIR/train.py" --config "$CONFIGS_DIR/baseline.yaml" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Pythia-410M on OpenWebText ===" | tee -a "$LOG_FILE"
python "$SCRIPTS_DIR/train.py" \
  --config "$CONFIGS_DIR/baseline.yaml" \
  --model_name pythia-410m \
  --dataset openwebtext \
  --lora_rank 32 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Gemma-2B on TinyStories ===" | tee -a "$LOG_FILE"
python "$SCRIPTS_DIR/train.py" \
  --config "$CONFIGS_DIR/baseline.yaml" \
  --model_name gemma-2b \
  --dataset tinystories \
  --lora_rank 8 2>&1 | tee -a "$LOG_FILE"

# Step 3: Evaluate & Generate Results
echo "" | tee -a "$LOG_FILE"
echo "[4/4] Evaluating and generating results..."
python "$SCRIPTS_DIR/evaluate.py" --output_dir "$OUTPUTS_DIR" 2>&1 | tee -a "$LOG_FILE"

# Step 4: Generate Report
echo "" | tee -a "$LOG_FILE"
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo "Results saved to: $OUTPUTS_DIR/results/benchmark_results.csv"
echo "Full log: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. Review results: cat $OUTPUTS_DIR/results/benchmark_results.csv"
echo "  2. Analyze with notebook: jupyter notebook notebooks/01_quickstart.ipynb"
echo "  3. Compare ablations: bash scripts/run_full_benchmark.sh"
echo ""
