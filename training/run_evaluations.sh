#!/bin/bash

# This script evaluates trained models by running 'get_model_stats.py' and 'eval.py'.
#
# Usage:
#   ./run_evaluations.sh [path_to_model_dir_1] [path_to_model_dir_2] ...
#
# Example:
#   ./run_evaluations.sh output/gaze360_mobileone_s0_*/ output/gaze360_resnet18_*
#
# It will create a file named 'eval_results.txt' inside each of the specified
# directories, containing the stats and evaluation results.

# --- Configuration ---
BASE_EVAL_CONFIG="configs/gaze360_eval.yaml"

# Check if at least one directory path is provided
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [path_to_model_dir_1] [path_to_model_dir_2] ..."
    echo "Example: $0 output/gaze360_mobileone_s0_*"
    exit 1
fi

# --- Main Loop ---
for model_dir in "$@"; do
    # Remove trailing slash if it exists
    model_dir=${model_dir%/}

    echo "--- Processing directory: $model_dir ---"

    # --- 1. Setup and Backbone Extraction ---
    RESULTS_FILE="${model_dir}/eval_results.txt"
    > "$RESULTS_FILE" # Create or clear the results file

    # Robustly extract backbone name from directory name.
    # Assumes format: <dataset>_<backbone_name>_<timestamp>
    dir_basename=$(basename "$model_dir")
    prefix="gaze360_"
    # Remove prefix
    name_no_prefix=${dir_basename#$prefix}
    # Remove suffix (timestamp)
    backbone=$(echo "$name_no_prefix" | sed 's/_[0-9]\{8\}-[0-9]\{6\}$//')

    if [ -z "$backbone" ]; then
        echo "Could not extract backbone name from '$model_dir'. Skipping."
        continue
    fi
    echo "Extracted backbone: $backbone"

    # --- 2. Create Temporary Eval Config ---
    TEMP_EVAL_CONFIG="configs/temp_eval_${backbone}.yaml"
    sed "s/backbone: .*/backbone: $backbone/" "$BASE_EVAL_CONFIG" > "$TEMP_EVAL_CONFIG"
    if [ $? -ne 0 ]; then
        echo "Failed to create temporary config for $backbone. Skipping."
        continue
    fi

    # --- 3. Get Model Stats (Size and FLOPs) ---
    echo "Running get_model_stats.py for $backbone..."
    stats_output=$(uv run get_model_stats.py "$backbone")

    flops=$(echo "$stats_output" | grep "FLOPs:" | awk '{print $2}')
    params=$(echo "$stats_output" | grep "Parameters:" | awk '{print $2}')

    if [ -z "$flops" ] || [ -z "$params" ]; then
        echo "Could not parse stats for $backbone."
    else
        echo "SIZE = ${params}" >> "$RESULTS_FILE"
        echo "FLOP = ${flops}" >> "$RESULTS_FILE"
    fi

    inference_time=""

    # --- 4. Evaluate Model Checkpoints ---
    echo "Finding model checkpoints (.pth files)..."
    checkpoints=$(find "$model_dir" -name "*.pth" | sort)

    for ckpt_path in $checkpoints; do
        epoch_name=$(basename "$ckpt_path" .pth | cut -d'_' -f2)
        if [[ "$(basename "$ckpt_path")" == "best.pth" ]]; then
            epoch_name="best"
        elif [[ "$(basename "$ckpt_path")" == "latest.pth" ]]; then
            epoch_name="latest"
        fi

        echo "Evaluating checkpoint: $ckpt_path (Epoch: $epoch_name)..."

        eval_output=$(uv run eval.py --config "$TEMP_EVAL_CONFIG" --weights "$ckpt_path" 2>&1)
        error=$(echo "$eval_output" | grep "Mean Angular Error" | awk '{print $11}')

        if [ -n "$error" ]; then
            echo "${epoch_name} = ${error}" >> "$RESULTS_FILE"
        else
            echo "Could not parse evaluation error for $ckpt_path."
        fi

        if [[ "$epoch_name" == "best" ]]; then
            inf_time_s=$(echo "$eval_output" | grep "Total evaluation time:" | awk '{print $5}')
            if [ -n "$inf_time_s" ]; then
                inference_time=$inf_time_s
            fi
        fi
    done

    # --- 5. Add Inference Time and Cleanup ---
    if [ -n "$inference_time" ]; then
        sed -i "1s;^;INFERENCE_TIME = ${inference_time}\n;" "$RESULTS_FILE"
    fi

    rm "$TEMP_EVAL_CONFIG"

    echo "--- Finished processing $model_dir. Results saved to $RESULTS_FILE ---"
    echo ""
done

echo "All evaluations complete."
