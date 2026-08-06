import os
import sys
import torch
import logging

def load_checkpoint(model, checkpoint_path, device):
    """
    Load a model checkpoint into the given model.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to map the checkpoint to.
    """
    if not os.path.exists(checkpoint_path):
        logging.error(f"Model checkpoint '{checkpoint_path}' not found.")
        sys.exit(1)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Accept both formats: {"model": ...} or {"model_state_dict": ...}
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise KeyError(f"Unrecognized checkpoint format. Keys: {list(checkpoint.keys())[:20]}")

        # Strip DDP prefix if present
        state_dict = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        logging.info(f"Loaded checkpoint from '{checkpoint_path}'. "
                     f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing:
            logging.debug(f"Missing (first 10): {missing[:10]}")
        if unexpected:
            logging.debug(f"Unexpected (first 10): {unexpected[:10]}")

    except Exception as e:
        logging.error(f"Error loading the model from '{checkpoint_path}': {e}")
        sys.exit(1)