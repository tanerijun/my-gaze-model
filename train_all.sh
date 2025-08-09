CONFIG_DIR="configs"
CONFIGS=(
  train01.yaml
  train02.yaml
  train03.yaml
  train04.yaml
  train05.yaml
)

for CONFIG in "${CONFIGS[@]}"; do
  echo "Running training with $CONFIG"
  uv run train.py --config "$CONFIG_DIR/$CONFIG"
done
