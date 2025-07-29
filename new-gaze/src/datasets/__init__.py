from .gaze360 import Gaze360Dataset
from .mpiigaze import MPIIGazeDataset

def get_dataset(config):
    if config['dataset_name'] == 'gaze360':
        return Gaze360Dataset(
            data_root=config['data_root'],
            split=config['split'],
            num_bins=config['num_bins'],
            angle_range=config['angle_range'],
            bin_width=config['bin_width'],
            image_size=config.get('image_size', 224)
        )
    elif config['dataset_name'] == 'mpiigaze':
        return MPIIGazeDataset(
            data_root=config['data_root'],
            split=config['split'], # For MPII, split will be the person ID like 'p00'
            num_bins=config['num_bins'],
            angle_range=config['angle_range'],
            bin_width=config['bin_width']
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset_name']}")
