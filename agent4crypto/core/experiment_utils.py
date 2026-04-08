import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional in some environments.
    torch = None


DEFAULT_RUN_SEEDS = [42, 43, 44, 45, 46]


def _normalize_seed_list(raw_seeds):
    if raw_seeds in (None, ""):
        return []
    if isinstance(raw_seeds, str):
        parts = [part.strip() for part in raw_seeds.split(",") if part.strip()]
        return [int(part) for part in parts]
    return [int(seed) for seed in raw_seeds]


def get_run_seed_list(config, default=None):
    """Read the configured fixed seed list used for repeated evaluation."""
    backtest_cfg = config.get("backtest", {})

    raw_seed_list = backtest_cfg.get("run_seeds")
    if raw_seed_list not in (None, ""):
        seeds = _normalize_seed_list(raw_seed_list)
        if seeds:
            return seeds

    raw_seed = backtest_cfg.get("run_seed")
    if raw_seed not in (None, ""):
        return [int(raw_seed)]

    return _normalize_seed_list(default) if default is not None else []


def get_run_seed(config, default=None):
    """Read the configured experiment seed from the shared config."""
    raw_seed = config.get("backtest", {}).get("run_seed")
    if raw_seed in (None, ""):
        seed_list = get_run_seed_list(config)
        if seed_list:
            return seed_list[0]
        return default
    return int(raw_seed)


def set_global_seed(seed):
    """Align Python, NumPy, and PyTorch RNG state for reproducible runs."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
