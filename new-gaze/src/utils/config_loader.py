import yaml
import os
from typing import Dict, Any

def load_config(cfg_path: str) -> Dict[str, Any]:
    """
    Loads a YAML config file, recursively handling an 'include' key.

    Args:
        cfg_path (str): The path to the YAML configuration file.

    Returns:
        Dict[str, Any]: The fully resolved configuration dictionary.
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # If an 'include' key exists, load the base config first
    if 'include' in cfg:
        # Get the path of the included file relative to the current file's directory
        base_cfg_path = os.path.join(os.path.dirname(cfg_path), cfg['include'])

        # Load the base config recursively
        base_cfg = load_config(base_cfg_path)

        # Remove the 'include' directive from the current config
        del cfg['include']

        # Merge the base config with the current config.
        # The current config's values will override the base config's values.
        base_cfg.update(cfg)
        return base_cfg

    return cfg
