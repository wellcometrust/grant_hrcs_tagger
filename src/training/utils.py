import yaml


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
