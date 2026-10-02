"""Internal backend version checks and isolation of legacy global random state."""

from contextlib import contextmanager
from importlib import import_module, metadata
import random

import numpy as np

__all__ = []


def _require_backend(name, expected_version):
    if not isinstance(expected_version, str) or not expected_version.strip():
        raise ValueError("Optional backend requires an explicit version")
    try:
        version = metadata.version(name)
        module = import_module(name)
    except (metadata.PackageNotFoundError, ImportError) as exc:
        raise ImportError(f"Enabled method requires optional backend {name}=={expected_version}; it is not installed") from exc
    if version != expected_version:
        raise ValueError(f"Backend version mismatch: requested {name}=={expected_version}, installed {version}")
    return module


@contextmanager
def _isolated_random_state(seed):
    import torch
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
