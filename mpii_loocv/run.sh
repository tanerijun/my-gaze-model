#!/bin/bash
#
# This script automates the full Leave-One-Person-Out Cross-Validation (LOOCV)
# protocol for the MPIIFaceGaze dataset. It timestamps the results, cleans
# up temporary training folders, and reports the MAE for each fold.

set -e

# --- CONFIGURATION ---
DATA_ROOT="/home/tanerijun/gaze-tracker-model/data/preprocessed/MPIIFaceGaze"
BASE_CONFIG="configs/mpiifacegaze_train.yaml"
EVAL_TEMPLATE="configs/mpiifacegaze_eval.yaml"
TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
RESULTS_DIR="output/mpii_loocv_results_${TIMESTAMP}"
# ---------------------

mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/full_loocv_log.txt"
> "$LOG_FILE"

echo "Starting MPIIFaceGaze LOOCV run on $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "--------------------------------------------------------" | tee -a "$LOG_FILE"

# Loop through all 15 participants (p00 to p14)
for i in $(seq -f "%02g" 0 14); do
    PERSON_ID="p$i"
    echo "--- Starting Fold $(($i + 1))/15: Testing on $PERSON_ID ---" | tee -a "$LOG_FILE"

    # 1. Prepare Data
    echo "Step 1: Preparing training labels, holding out $PERSON_ID..."
    uv run scripts/prepare_mpii_labels.py --data_root "$DATA_ROOT" --test_person "$PERSON_ID"

    # 2. Train Model
    echo "Step 2: Training model..."
    TEMP_TRAIN_LOG="$RESULTS_DIR/temp_train_fold_${PERSON_ID}.log"
    # Redirect stderr to stdout (2>&1) here as well to be safe
    uv run train.py --config "$BASE_CONFIG" 2>&1 | tee "$TEMP_TRAIN_LOG"
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

    echo "Step 3: Evaluating on $PERSON_ID..."
    # --- THE DEFINITIVE FIX ---
    # Redirect stderr (2) to stdout (1) so the logger output is captured.
    EVAL_OUTPUT=$(uv run eval.py --config "$TEMP_EVAL_CONFIG" --weights "$WEIGHTS_PATH" 2>&1)

    echo "$EVAL_OUTPUT" | tee -a "$LOG_FILE"

    # This parse will now work because EVAL_OUTPUT contains the logger's messages.
    CURRENT_MAE=$(echo "$EVAL_OUTPUT" | grep "Mean Angular Error" | awk '{print $(NF-1)}')
    # ---------------------------------

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

echo "Cross-validation complete." | tee -a "$LOG_FILE"
echo "Calculating final average MAE..."

AVG_MAE=$(grep "Mean Angular Error" "$LOG_FILE" | awk '{print $(NF-1)}' | awk '{s+=$1} END {print s/NR}')

echo "========================================================" | tee -a "$LOG_FILE"
echo "FINAL LEAVE-ONE-OUT CROSS-VALIDATION RESULT" | tee -a "$LOG_FILE"
echo "Run completed on: $(date)" | tee -a "$LOG_FILE"
printf "Average Mean Angular Error across 15 folds: %.2f\n" "$AVG_MAE" | tee -a "$LOG_FILE"
echo "========================================================" | tee -a "$LOG_FILE"
