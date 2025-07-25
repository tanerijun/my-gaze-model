import yaml
import argparse
from torch.utils.data import DataLoader
from src.datasets import get_dataset

def test_dataset(cfg_path):
    """
    Tests the dataset loading and batching process.
    Args:
        cfg_path (str): Path to YAML config file.
    """
    # Load config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print("--- Configuration ---")
    print(cfg)
    print("-----------------------")

    # Get dataset using factory
    try:
        dataset = get_dataset(cfg)
        print(f"\nSuccessfully created dataset '{cfg['dataset_name']}'.")
        print(f"Total samples in '{cfg['split']}' split: {len(dataset)}")
    except Exception as e:
        print("\n---! Failed to create dataset !---")
        print(f"Error: {e}")
        print("Please check the 'data_root' path in your config file and the dataset structure.")
        return

    loader = DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 2), # Use a small batch for testing
        shuffle=True
    )

    # Get one batch from the loader
    try:
        image_batch, binned_labels, cont_labels = next(iter(loader))
        print("\n--- Fetched one batch ---")
        print(f"Image batch shape: {image_batch.shape}, dtype: {image_batch.dtype}")
        print(f"Binned labels shape: {binned_labels.shape}, dtype: {binned_labels.dtype}")
        print(f"Continuous labels shape: {cont_labels.shape}, dtype: {cont_labels.dtype}")
        print("-------------------------\n")
        print("Dataset test PASSED!")

    except Exception as e:
        print("\n---! Failed to fetch a batch !---")
        print(f"Error: {e}")
        print("There might be an issue with file paths or an error in the __getitem__ method.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    test_dataset(args.config)
