import csv
import json
from pathlib import Path
from common import PROJECT_ROOT, load_config, resolve_project_path, read_spikeglx_meta, parse_positive_int

cfg = load_config()
bin_path = resolve_project_path(cfg["data_file"])
meta_path = bin_path.with_suffix(".meta")
meta = read_spikeglx_meta(meta_path)
n_chan_bin = parse_positive_int(meta, "nSavedChans")
csv_path = PROJECT_ROOT / "config" / "probe_contacts.csv"
probe_path = resolve_project_path(cfg["probe_file"])

rows = []
with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    required = {"binary_channel", "x_um", "y_um", "shank"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"El CSV debe tener columnas: {sorted(required)}")
    for row in reader:
        if not row.get("binary_channel", "").strip():
            continue
        rows.append({
            "binary_channel": int(row["binary_channel"]),
            "x_um": float(row["x_um"]),
            "y_um": float(row["y_um"]),
            "shank": int(row["shank"]),
        })

if not rows:
    raise ValueError("probe_contacts.csv está vacío. Completalo con el mapa REAL de la sonda antes de crear el probe.")

channels = [r["binary_channel"] for r in rows]
if len(channels) != len(set(channels)):
    raise ValueError("Hay binary_channel repetidos en probe_contacts.csv")
if min(channels) < 0 or max(channels) >= n_chan_bin:
    raise ValueError(f"Los índices de canal deben estar entre 0 y {n_chan_bin - 1}")

probe = {
    "chanMap": channels,
    "xc": [r["x_um"] for r in rows],
    "yc": [r["y_um"] for r in rows],
    "kcoords": [r["shank"] for r in rows],
    "n_chan": len(rows),
    "notes": "Neuropixels 2.0; completar/verificar con el mapa de canales y geometría reales de esta adquisición."
}

probe_path.parent.mkdir(parents=True, exist_ok=True)
with probe_path.open("w", encoding="utf-8") as f:
    json.dump(probe, f, indent=2)
print(f"Probe guardado: {probe_path}")
print(f"Contactos ephys en mapa: {len(rows)}; canales totales del binario: {n_chan_bin}")
