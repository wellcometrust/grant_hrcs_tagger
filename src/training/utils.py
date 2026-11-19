import yaml
import torch


def load_yaml_config(config_path: str):
    """Load yaml configuration file.

    Args:
        config_path (str): path to the yaml configuration file

    Returns:
        dict: configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def init_device():
    """Initialize device to use for training.

    Returns:
        str: device to use for training
    """
    torch.cuda.empty_cache()
    if torch.backends.mps.is_built():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
