from .gaze360 import Gaze360Dataset
from .mpiigaze import MPIIGazeDataset

def get_dataset(config):
    dataset_name = config['dataset_name']
    image_size = config.get('image_size', 224)

    if dataset_name == 'gaze360':
        return Gaze360Dataset(
            data_root=config['data_root'],
            split=config['split'],
            num_bins=config['num_bins'],
            angle_range=config['angle_range'],
            image_size=image_size
        )
    elif dataset_name == 'mpiigaze':
        return MPIIGazeDataset(
            data_root=config['data_root'],
            # For MPII, 'split' refers to the label file name (e.g., 'train' or 'p14')
            split=config['split'],
            num_bins=config['num_bins'],
            angle_range=config['angle_range'],
            image_size=image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset_name']}")
