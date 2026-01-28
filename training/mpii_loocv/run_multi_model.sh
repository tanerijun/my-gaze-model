#!/bin/bash
#
# This script automates the full Leave-One-Person-Out Cross-Validation (LOOCV)
# protocol for the MPIIFaceGaze dataset across multiple backbone models.
# It timestamps the results, includes the model name in output directories,
# cleans up temporary training folders, and reports the MAE for each fold.
#
# Usage:
#   ./run_multi_model.sh           # Run full LOOCV for all models
#   ./run_multi_model.sh --test    # Run quick test with 1 epoch to verify script works

set -e

# Trap for Ctrl+C
trap "echo 'Script interrupted. Exiting.'; exit" INT

# --- CONFIGURATION ---
DATA_ROOT="/home/tanerijun/gaze-tracker-model/data/preprocessed/MPIIFaceGaze"
BASE_CONFIG="configs/mpiifacegaze_train.yaml"
EVAL_TEMPLATE="configs/mpiifacegaze_eval.yaml"

# Array of backbone models to run LOOCV on.
# You can add or remove backbones from this list.
# Make sure the backbone names are valid in model registry.
BACKBONES=(
    "mobileone_s1"
    # "swiftformer_xs"
    # "mobilenetv3_small_100"
    # "efficientformerv2_s0"
    # Add more backbones here...
)

# Test mode flag - set via command line argument
TEST_MODE=false
if [[ "$1" == "--test" ]]; then
    TEST_MODE=true
    echo "=== TEST MODE ENABLED: Using 1 epoch for quick validation ==="
fi
# ---------------------

# Create a master results directory for this run
MASTER_TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
MASTER_RESULTS_DIR="output/mpii_loocv_multi_model_${MASTER_TIMESTAMP}"
mkdir -p "$MASTER_RESULTS_DIR"
MASTER_LOG="$MASTER_RESULTS_DIR/master_summary.txt"
> "$MASTER_LOG"

echo "Starting Multi-Model MPIIFaceGaze LOOCV run on $(date)" | tee -a "$MASTER_LOG"
echo "Models to evaluate: ${BACKBONES[*]}" | tee -a "$MASTER_LOG"
if $TEST_MODE; then
    echo "*** TEST MODE: Training with 1 epoch only ***" | tee -a "$MASTER_LOG"
fi
echo "Master results directory: $MASTER_RESULTS_DIR" | tee -a "$MASTER_LOG"
echo "========================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

# Store final results for each model
declare -A MODEL_RESULTS

# Loop through each backbone model
for backbone in "${BACKBONES[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "########################################################" | tee -a "$MASTER_LOG"
    echo "### Starting LOOCV for backbone: $backbone" | tee -a "$MASTER_LOG"
    echo "########################################################" | tee -a "$MASTER_LOG"

    TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
    # Include model name in results directory
    RESULTS_DIR="$MASTER_RESULTS_DIR/${backbone}_loocv_${TIMESTAMP}"
    mkdir -p "$RESULTS_DIR"
    LOG_FILE="$RESULTS_DIR/full_loocv_log.txt"
    > "$LOG_FILE"

    echo "Starting MPIIFaceGaze LOOCV for $backbone on $(date)" | tee -a "$LOG_FILE"
    echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
    echo "--------------------------------------------------------" | tee -a "$LOG_FILE"

    # Create a temporary config file for this backbone (with test mode modifications if needed)
    TEMP_TRAIN_CONFIG="configs/temp_train_config_${backbone}.yaml"
    cp "$BASE_CONFIG" "$TEMP_TRAIN_CONFIG"

    # Modify config for test mode if enabled
    if $TEST_MODE; then
        # Use sed to modify epochs to 1 for quick testing
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS sed requires -i ''
            sed -i '' 's/^epochs:.*/epochs: 1/' "$TEMP_TRAIN_CONFIG"
        else
            # Linux sed
            sed -i 's/^epochs:.*/epochs: 1/' "$TEMP_TRAIN_CONFIG"
        fi
        echo "Modified config for test mode: epochs set to 1" | tee -a "$LOG_FILE"
    fi

    # Loop through all 15 participants (p00 to p14)
    for i in $(seq -f "%02g" 0 14); do
        PERSON_ID="p$i"
        echo "--- Starting Fold $(($i + 1))/15: Testing on $PERSON_ID ---" | tee -a "$LOG_FILE"

        # 1. Prepare Data
        echo "Step 1: Preparing training labels, holding out $PERSON_ID..."
        uv run mpii_loocv/prepare_mpii_labels.py --data_root "$DATA_ROOT" --test_person "$PERSON_ID"

        # 2. Train Model with backbone override
        echo "Step 2: Training model with backbone: $backbone..."
        TEMP_TRAIN_LOG="$RESULTS_DIR/temp_train_fold_${PERSON_ID}.log"
        # Use --backbone to override the backbone in config
        uv run train.py --config "$TEMP_TRAIN_CONFIG" --backbone "$backbone" 2>&1 | tee "$TEMP_TRAIN_LOG"
        TRAIN_DIR=$(tail -n 1 "$TEMP_TRAIN_LOG")

        if [ ! -d "$TRAIN_DIR" ]; then
            echo "Error: Could not determine training output directory. Aborting." >&2
            exit 1
        fi
        echo "Training output saved in: $TRAIN_DIR"

        WEIGHTS_PATH="$TRAIN_DIR/latest.pth"
        if [ ! -f "$WEIGHTS_PATH" ]; then
            echo "Error: Weights file not found at $WEIGHTS_PATH" >&2
            exit 1
        fi
        echo "Using weights: $WEIGHTS_PATH"

        # 3. Evaluate Model
        TEMP_EVAL_CONFIG="configs/temp_eval_config_for_${PERSON_ID}.yaml"
        sed "s/__PERSON_ID__/$PERSON_ID/" "$EVAL_TEMPLATE" > "$TEMP_EVAL_CONFIG"

        # Also update backbone in eval config
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/^backbone:.*/backbone: \"$backbone\"/" "$TEMP_EVAL_CONFIG"
        else
            sed -i "s/^backbone:.*/backbone: \"$backbone\"/" "$TEMP_EVAL_CONFIG"
        fi

        echo "Step 3: Evaluating on $PERSON_ID..."
        EVAL_OUTPUT=$(uv run eval.py --config "$TEMP_EVAL_CONFIG" --weights "$WEIGHTS_PATH" 2>&1)

        echo "$EVAL_OUTPUT" | tee -a "$LOG_FILE"

        CURRENT_MAE=$(echo "$EVAL_OUTPUT" | grep "Mean Angular Error" | awk '{print $(NF-1)}')

        if [ -z "$CURRENT_MAE" ]; then
            echo "Warning: Could not parse MAE for fold $PERSON_ID. Check logs." | tee -a "$LOG_FILE"
        else
            echo "MAE for fold $(($i + 1))/15 ($PERSON_ID): $CURRENT_MAE degrees" | tee -a "$LOG_FILE"
        fi

        # 4. Cleanup
        echo "Step 4: Cleaning up temporary files..."
        rm "$TEMP_EVAL_CONFIG" "$TEMP_TRAIN_LOG"
        if [ -d "$TRAIN_DIR" ]; then
            rm -rf "$TRAIN_DIR"
            echo "Removed temporary training directory: $TRAIN_DIR"
        fi

        echo "--- Fold $(($i + 1))/15 for $PERSON_ID Complete ---" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done

    # Clean up the temporary training config
    rm -f "$TEMP_TRAIN_CONFIG"

    echo "Cross-validation for $backbone complete." | tee -a "$LOG_FILE"
    echo "Calculating final average MAE for $backbone..."

    AVG_MAE=$(grep "Mean Angular Error" "$LOG_FILE" | awk '{print $(NF-1)}' | awk '{s+=$1} END {print s/NR}')

    echo "========================================================" | tee -a "$LOG_FILE"
    echo "FINAL LEAVE-ONE-OUT CROSS-VALIDATION RESULT FOR $backbone" | tee -a "$LOG_FILE"
    echo "Run completed on: $(date)" | tee -a "$LOG_FILE"
    printf "Average Mean Angular Error across 15 folds: %.2f degrees\n" "$AVG_MAE" | tee -a "$LOG_FILE"
    echo "========================================================" | tee -a "$LOG_FILE"

    # Store result for master summary
    MODEL_RESULTS["$backbone"]="$AVG_MAE"
    echo "$backbone: $AVG_MAE degrees" >> "$MASTER_LOG"

done

# Print master summary
echo "" | tee -a "$MASTER_LOG"
echo "########################################################" | tee -a "$MASTER_LOG"
echo "### MASTER SUMMARY - ALL MODELS COMPLETED" | tee -a "$MASTER_LOG"
echo "### Completed on: $(date)" | tee -a "$MASTER_LOG"
echo "########################################################" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Model                          | Average MAE (degrees)" | tee -a "$MASTER_LOG"
echo "-------------------------------|----------------------" | tee -a "$MASTER_LOG"
for backbone in "${BACKBONES[@]}"; do
    printf "%-30s | %s\n" "$backbone" "${MODEL_RESULTS[$backbone]}" | tee -a "$MASTER_LOG"
done
echo "" | tee -a "$MASTER_LOG"
echo "All results saved in: $MASTER_RESULTS_DIR" | tee -a "$MASTER_LOG"
echo "########################################################" | tee -a "$MASTER_LOG"
