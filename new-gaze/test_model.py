import yaml
import argparse
import torch
from src.models import build_model

def test_model_build(cfg_path):
    """
    Tests the model building process and a single forward pass.
    Args:
        cfg_path (str): Path to YAML config file.
    """
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print("--- Configuration ---")
    print(f"Backbone: {cfg['backbone']}")
    print(f"Num Bins: {cfg['num_bins']}")
    print("-----------------------")

    # 1. Build model from config
    try:
        model = build_model(cfg)
        print("\n--- Model Architecture ---")
        print(model)
        print("--------------------------\n")
        print("Model built successfully!")
    except Exception as e:
        print("\n---! Failed to build model !---")
        print(f"Error: {e}")
        return

    # 2. Test a forward pass
    try:
        # Create a dummy input tensor with the expected shape (Batch, Channels, Height, Width)
        # Using a batch size of 4 for this test
        dummy_input = torch.randn(4, 3, 224, 224)

        print("--- Testing Forward Pass ---")
        print(f"Input shape: {dummy_input.shape}")

        # Pass the input through the model
        pitch_pred, yaw_pred = model(dummy_input)

        print(f"Output Pitch shape: {pitch_pred.shape}")
        print(f"Output Yaw shape: {yaw_pred.shape}")
        print("----------------------------\n")

        # 3. Check output shapes
        expected_shape = (4, cfg['num_bins'])
        assert pitch_pred.shape == expected_shape, f"Pitch shape mismatch! Got {pitch_pred.shape}, expected {expected_shape}"
        assert yaw_pred.shape == expected_shape, f"Yaw shape mismatch! Got {yaw_pred.shape}, expected {expected_shape}"

        print("Model forward pass test PASSED!")

    except Exception as e:
        print("\n---! Failed during forward pass !---")
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    test_model_build(args.config)
