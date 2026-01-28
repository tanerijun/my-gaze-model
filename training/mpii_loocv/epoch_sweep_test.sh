#!/bin/bash
#
# Epoch Sweep Test for MPIIFaceGaze LOOCV
#
# This script runs a quick test to find the optimal number of epochs
# by training on a subset of folds with different epoch counts.
#
# Usage:
#   ./epoch_sweep_test.sh
#
# This will help determine if more epochs actually improves performance
# or if overfitting occurs.

set -e

# Trap for Ctrl+C
trap "echo 'Script interrupted. Exiting.'; exit" INT

# --- CONFIGURATION ---
# IMPORTANT: DATA_ROOT must match the data_root in mpiifacegaze_train.yaml
DATA_ROOT="/home/tanerijun/gaze-tracker-model/training/data/preprocessed/MPIIFaceGaze"
BASE_CONFIG="configs/mpiifacegaze_train.yaml"
EVAL_TEMPLATE="configs/mpiifacegaze_eval.yaml"

# Backbone to test with
BACKBONE="swiftformer_xs"

# Epoch counts to test
EPOCHS_TO_TEST=(1 3 5 10 15)

# Test on a subset of folds for speed (e.g., 3 folds instead of 15)
# Using p00, p07, p14 to get a representative sample
TEST_PERSONS=("p00" "p07" "p14")
# ---------------------

# Create results directory
TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
RESULTS_DIR="output/epoch_sweep_${BACKBONE}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

echo "======================================================" | tee "$SUMMARY_FILE"
echo "Epoch Sweep Test for $BACKBONE" | tee -a "$SUMMARY_FILE"
echo "Started: $(date)" | tee -a "$SUMMARY_FILE"
echo "Testing epochs: ${EPOCHS_TO_TEST[*]}" | tee -a "$SUMMARY_FILE"
echo "Test persons: ${TEST_PERSONS[*]}" | tee -a "$SUMMARY_FILE"
echo "======================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Store results
declare -A EPOCH_RESULTS

for epochs in "${EPOCHS_TO_TEST[@]}"; do
    echo "" | tee -a "$SUMMARY_FILE"
    echo "------------------------------------------------------" | tee -a "$SUMMARY_FILE"
    echo "Testing with $epochs epoch(s)..." | tee -a "$SUMMARY_FILE"
    echo "------------------------------------------------------" | tee -a "$SUMMARY_FILE"

    # Create temporary config with this epoch count
    TEMP_TRAIN_CONFIG="configs/temp_epoch_sweep_${epochs}.yaml"
    cp "$BASE_CONFIG" "$TEMP_TRAIN_CONFIG"

    # Modify epochs in config
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/^epochs:.*/epochs: $epochs/" "$TEMP_TRAIN_CONFIG"
    else
        sed -i "s/^epochs:.*/epochs: $epochs/" "$TEMP_TRAIN_CONFIG"
    fi

    # Collect MAEs for this epoch count
    MAE_SUM=0
    FOLD_COUNT=0

    for PERSON_ID in "${TEST_PERSONS[@]}"; do
        echo "  Fold: $PERSON_ID" | tee -a "$SUMMARY_FILE"

        # 1. Prepare Data
        uv run mpii_loocv/prepare_mpii_labels.py --data_root "$DATA_ROOT" --test_person "$PERSON_ID" > /dev/null

        # 2. Train Model
        TEMP_TRAIN_LOG="$RESULTS_DIR/train_${epochs}ep_${PERSON_ID}.log"
        uv run train.py --config "$TEMP_TRAIN_CONFIG" --backbone "$BACKBONE" > "$TEMP_TRAIN_LOG" 2>&1
        TRAIN_DIR=$(tail -n 1 "$TEMP_TRAIN_LOG")

        if [ ! -d "$TRAIN_DIR" ]; then
            echo "    Error: Training failed. Check $TEMP_TRAIN_LOG" | tee -a "$SUMMARY_FILE"
            continue
        fi

        WEIGHTS_PATH="$TRAIN_DIR/latest.pth"

        # 3. Evaluate
        TEMP_EVAL_CONFIG="configs/temp_eval_sweep_${PERSON_ID}.yaml"
        sed "s/__PERSON_ID__/$PERSON_ID/" "$EVAL_TEMPLATE" > "$TEMP_EVAL_CONFIG"
        echo "backbone: \"$BACKBONE\"" >> "$TEMP_EVAL_CONFIG"

        EVAL_OUTPUT=$(uv run eval.py --config "$TEMP_EVAL_CONFIG" --weights "$WEIGHTS_PATH" 2>&1)
        CURRENT_MAE=$(echo "$EVAL_OUTPUT" | grep "Mean Angular Error" | awk '{print $(NF-1)}')

        if [ -n "$CURRENT_MAE" ]; then
            echo "    MAE: $CURRENT_MAE degrees" | tee -a "$SUMMARY_FILE"
            MAE_SUM=$(echo "$MAE_SUM + $CURRENT_MAE" | bc)
            FOLD_COUNT=$((FOLD_COUNT + 1))
        else
            echo "    Error: Could not parse MAE" | tee -a "$SUMMARY_FILE"
        fi

        # Cleanup
        rm -f "$TEMP_EVAL_CONFIG"
        rm -rf "$TRAIN_DIR"
    done

    # Calculate average MAE for this epoch count
    if [ $FOLD_COUNT -gt 0 ]; then
        AVG_MAE=$(echo "scale=4; $MAE_SUM / $FOLD_COUNT" | bc)
        EPOCH_RESULTS[$epochs]="$AVG_MAE"
        echo "" | tee -a "$SUMMARY_FILE"
        echo "  Average MAE for $epochs epoch(s): $AVG_MAE degrees" | tee -a "$SUMMARY_FILE"
    fi

    # Cleanup temp config
    rm -f "$TEMP_TRAIN_CONFIG"
done

# Print final summary
echo "" | tee -a "$SUMMARY_FILE"
echo "======================================================" | tee -a "$SUMMARY_FILE"
echo "EPOCH SWEEP RESULTS SUMMARY" | tee -a "$SUMMARY_FILE"
echo "Backbone: $BACKBONE" | tee -a "$SUMMARY_FILE"
echo "Completed: $(date)" | tee -a "$SUMMARY_FILE"
echo "======================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Epochs | Average MAE (degrees)" | tee -a "$SUMMARY_FILE"
echo "-------|----------------------" | tee -a "$SUMMARY_FILE"

# Find best epoch count
BEST_EPOCHS=""
BEST_MAE="999"

for epochs in "${EPOCHS_TO_TEST[@]}"; do
    mae="${EPOCH_RESULTS[$epochs]}"
    if [ -n "$mae" ]; then
        printf "%-6s | %s\n" "$epochs" "$mae" | tee -a "$SUMMARY_FILE"
        # Check if this is the best
        is_better=$(echo "$mae < $BEST_MAE" | bc)
        if [ "$is_better" -eq 1 ]; then
            BEST_MAE="$mae"
            BEST_EPOCHS="$epochs"
        fi
    fi
done

echo "" | tee -a "$SUMMARY_FILE"
echo "RECOMMENDATION: Use $BEST_EPOCHS epoch(s) (MAE: $BEST_MAE degrees)" | tee -a "$SUMMARY_FILE"
echo "======================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"
echo "Full results saved to: $RESULTS_DIR" | tee -a "$SUMMARY_FILE"
