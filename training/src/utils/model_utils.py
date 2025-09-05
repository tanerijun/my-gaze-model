import logging

def load_filtered_state_dict(model, state_dict):
    """
    Loads a state dictionary into a model, but only for layers that match.
    This is useful for loading pre-trained weights into a modified architecture.

    Args:
        model (torch.nn.Module): The model to load weights into.
        state_dict (dict): The dictionary of weights to load.
    """
    current_model_dict = model.state_dict()

    # Filter out unnecessary keys and keys that don't match in shape
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if k in current_model_dict and current_model_dict[k].shape == v.shape
    }

    # Overwrite the model's state dictionary with the filtered weights
    current_model_dict.update(filtered_state_dict)
    model.load_state_dict(current_model_dict)

    # Log what was loaded
    logger = logging.getLogger('new_gaze_logger')
    logger.info(f"Loaded {len(filtered_state_dict)}/{len(current_model_dict)} layers from pretrained weights.")
    if len(filtered_state_dict) == 0:
        logger.warning("Warning: No layers were loaded from the pretrained weights.")
