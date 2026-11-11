"""Load config.yaml and pick the settings of the chosen run mode."""
import yaml


def load_config(path="config.yaml", mode=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["run_mode"] = mode or cfg["run_mode"]
    if cfg["run_mode"] not in cfg["run_modes"]:
        raise ValueError(f"run_mode must be one of {list(cfg['run_modes'])}, got {cfg['run_mode']!r}")
    cfg["run"] = cfg["run_modes"][cfg["run_mode"]]     # shortcut: cfg["run"]["budgets"] etc.
    return cfg
