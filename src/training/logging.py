import wandb


def init_wandb(project_name, config_settings, report_to):
    """Initialize wandb if reporting is enabled.

    Args:
        project_name (str): Name of the wandb project
        config_settings (dict): Configuration settings to log
        report_to (str): Reporting destination ('wandb' or None)
    """
    if report_to == "wandb":
        wandb.init(project=project_name, config=config_settings)


def log_to_wandb(data, report_to):
    """Log data to wandb if reporting is enabled.

    Args:
        data (dict): Data to log
        report_to (str): Reporting destination ('wandb' or None)
    """
    if report_to == "wandb":
        wandb.log(data)
