"""Atomic run artifacts; portable source and private run directories are separate."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _replace(temporary, destination):
    # Windows can briefly deny replacement while another worker renames the
    # same status file. Retry bounded contention; real permission failures raise.
    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.005 * (attempt + 1))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Workers share progress.json; each writer needs its own staging file.
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.partial')
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    try:
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def arrays(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    try:
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
