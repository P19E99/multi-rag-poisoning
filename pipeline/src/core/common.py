"""Small shared configuration, reproducibility and atomic I/O helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import random
import tempfile


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=handlers, force=True)


def load_config(path):
    import yaml
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    root = (config_path.parent / config.get("project_root", "..")).resolve()
    def resolve(value):
        if value is None:
            return None
        p = Path(os.path.expandvars(str(value))).expanduser()
        return str(p.resolve() if p.is_absolute() else (root / p).resolve())
    config["project_root"] = str(root)
    config["paths"] = {key: resolve(value) for key, value in config["paths"].items()}
    for section, keys in {"dataset": ("cache_dir", "local_path"),
                          "expansion": ("model_path",), "retrieval": ("model_path",)}.items():
        for key in keys:
            if key in config.get(section, {}):
                config[section][key] = resolve(config[section][key])
    if config["expansion"]["num_rewrites"] != 4:
        raise ValueError("This pipeline requires exactly 4 rewrites")
    if config["expansion"]["temperature"] != 0:
        raise ValueError("This pipeline requires temperature=0 (greedy decoding)")
    if config["retrieval"]["top_k_per_view"] != 10 or config["retrieval"]["final_top_k"] != 5:
        raise ValueError("This pipeline requires view Top-10 and final Top-5")
    threads = int(config.get("runtime", {}).get("num_threads", 8))
    if threads < 1:
        raise ValueError("runtime.num_threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return config


def seed_everything(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    # Do not import torch in the offline corpus/cache/analysis stages.
    import sys
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fingerprint(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path}:{number}")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=f".{destination.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
