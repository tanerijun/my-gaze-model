CONFIG_DIR="configs"
CONFIGS=(
  train1.yaml
  train2.yaml
  train3.yaml
  train4.yaml
  train5.yaml
  train6.yaml
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "Running training with $CONFIG"
  uv run train.py --config "$CONFIG_DIR/$CONFIG"
done
