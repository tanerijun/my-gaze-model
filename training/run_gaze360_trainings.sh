#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Trap for Ctrl+C
trap "echo 'Script interrupted. Exiting.'; exit" INT

# Array of backbone models to train.
# You can add or remove backbones from this list.
# Make sure the backbone names are valid in model registry.
BACKBONES=("resnet18" "resnet34" "resnet50" "efficientnet_b0" "efficientnet_b1" "efficientnet_b2" "efficientnet_b3" "efficientnet_b4" "efficientnet_b5" "efficientnet_b6" "efficientnet_b7" "efficientnet_b8" "efficientformer_l1" "efficientformer_l3" "efficientformer_l7" "efficientformerv2_s0" "efficientformerv2_s1" "efficientformerv2_s2" "mobilevit_s" "mobilevit_xs" "mobilevit_xxs" "mobilevitv2_050" "mobilevitv2_075" "mobilevitv2_100" "mobilevitv2_125" "mobilevitv2_150" "mobilevitv2_175" "mobilevitv2_200" "mobilenetv4_conv_small" "mobilenetv4_conv_medium" "mobilenetv4_conv_large" "mobilenetv3_small_050" "mobilenetv3_small_075" "mobilenetv3_small_100" "fasternet_s" "fasternet_m" "fasternet_l" "tinynet_a" "tinynet_b" "tinynet_c" "tinynet_d" "tinynet_e")

# Path to the original configuration file
BASE_CONFIG="configs/gaze360_train.yaml"

# Ensure the output directory for logs exists
mkdir -p output

# Loop through each backbone and start a training process
for backbone in "${BACKBONES[@]}"
do
  echo "--- Preparing training for backbone: $backbone ---"

  # Define a log file for the training output
  LOG_FILE="output/train_${backbone}_$(date +%Y%m%d-%H%M%S).log"

  # Start the training
  echo "Starting training for $backbone. Log file: $LOG_FILE"
  nohup uv run train.py --config "$BASE_CONFIG" --backbone "$backbone" > "$LOG_FILE" 2>&1
done

echo "--- All training processes have completed. ---"
