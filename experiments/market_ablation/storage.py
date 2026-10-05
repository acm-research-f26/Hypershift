"""Atomic, resumable artifacts and a shared disk reserve."""
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import shutil
from time import sleep
from uuid import uuid4
import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preparation and training publish status concurrently; each writer needs
    # its own staging file even when they replace the same destination.
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid4().hex}.partial")
    temporary.write_text(json.dumps(value, default=json_default, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            sleep(.05)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def reserve(config, estimated_bytes=0):
    config.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(config.root).free
    if free - estimated_bytes < config.reserve_gib * 2**30:
        raise OSError(f"Disk reserve: {free / 2**30:.2f} GiB free; {config.reserve_gib} GiB required")
    return free


def event(config, phase, **fields):
    value = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase, **fields}
    config.root.mkdir(parents=True, exist_ok=True)
    with (config.root / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, default=json_default, allow_nan=False) + "\n")
    atomic_json(config.root / "progress.json", value)
    print(json.dumps(value, default=json_default), flush=True)


def digest_file(path):
    with Path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def archive_execution_sources(config):
    """Preserve executable research recipes and the locked dependency versions."""
    from importlib.metadata import version,PackageNotFoundError
    from dataclasses import asdict
    from zipfile import ZipFile,ZIP_DEFLATED
    import platform
    import torch
    repository=Path(__file__).resolve().parents[2]
    files=[]
    for directory in ("experiments","hyperedges","hyperbolicity","src","data"):
        files.extend(sorted((repository/directory).rglob("*.py")))
    files.extend(repository/name for name in ("pyproject.toml","uv.lock","GOAL_PROMPT.txt","notebooks/main_ablation.ipynb",
                                              "README.md","experiments/market_ablation/README.md"))
    contents={p.relative_to(repository).as_posix():p.read_bytes() for p in files if p.is_file()}
    hashes={name:sha256(data).hexdigest() for name,data in contents.items()}
    identity=sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    path=config.root/"execution_sources"/f"{identity}.zip"
    if not path.exists():
        reserve(config,sum(p.stat().st_size for p in files if p.is_file()))
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+f".{os.getpid()}.{uuid4().hex}.partial")
        with ZipFile(temporary,"w",compression=ZIP_DEFLATED) as archive:
            for name,data in contents.items(): archive.writestr(name,data)
        os.replace(temporary,path)
    packages={}
    for name in ("torch","torch-geometric","optuna","ripser","numpy","pandas","scipy","numba","shapediscover","arch","scikit-fuzzy"):
        try: packages[name]=version(name)
        except PackageNotFoundError: packages[name]=None
    atomic_json(config.root/"execution_sources/latest.json",{"identity":identity,"archive":str(path),"files":hashes,
        "packages":packages,"python":platform.python_version(),"platform":platform.platform(),"cuda_runtime":torch.version.cuda,
        "GPU":torch.cuda.get_device_name() if torch.cuda.is_available() else None,"configuration":asdict(config),
        "saved_at":datetime.now(timezone.utc).isoformat()})
