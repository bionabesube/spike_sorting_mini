from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config():
    path = PROJECT_ROOT / "config" / "settings.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(value):
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def read_spikeglx_meta(meta_path):
    meta = {}
    with Path(meta_path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                meta[key.strip()] = value.strip()
    return meta


def parse_positive_int(meta, key):
    try:
        value = int(meta[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"No se pudo leer {key} como entero en el .meta")
    if value <= 0:
        raise ValueError(f"{key} debe ser > 0; valor: {value}")
    return value


def parse_positive_float(meta, key):
    try:
        value = float(meta[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"No se pudo leer {key} como número en el .meta")
    if value <= 0:
        raise ValueError(f"{key} debe ser > 0; valor: {value}")
    return value
