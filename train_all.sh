CONFIG_DIR="configs"
CONFIGS=(
  g360_lowformer_b15.yaml
  g360_lowformer_b3.yaml
  g360_mobileone_s0.yaml
  g360_mobileone_s1.yaml
  g360_mobileone_s2.yaml
  g360_resnet18.yaml
  g360_resnet34.yaml
  g360_resnet50.yaml
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "Running training with $CONFIG"
  uv run train.py --config "$CONFIG_DIR/$CONFIG"
done
