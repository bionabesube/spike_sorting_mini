from pathlib import Path
import numpy as np
import pandas as pd
from common import load_config, resolve_project_path, read_spikeglx_meta, parse_positive_float

cfg = load_config()
bin_path = resolve_project_path(cfg["data_file"])
results_dir = resolve_project_path(cfg["results_root"]) / bin_path.stem

spike_times_path = results_dir / "spike_times.npy"
spike_clusters_path = results_dir / "spike_clusters.npy"
if not spike_times_path.exists() or not spike_clusters_path.exists():
    raise FileNotFoundError(f"No encuentro spike_times.npy y spike_clusters.npy en {results_dir}")

spike_times = np.load(spike_times_path).reshape(-1)
spike_clusters = np.load(spike_clusters_path).reshape(-1)
if len(spike_times) != len(spike_clusters):
    raise ValueError("Los arrays de spike_times y spike_clusters no tienen igual longitud")

meta = read_spikeglx_meta(bin_path.with_suffix('.meta'))
fs = parse_positive_float(meta, 'imSampRate')
spike_seconds = spike_times / fs
unique, counts = np.unique(spike_clusters, return_counts=True)
summary = pd.DataFrame({"cluster_id": unique.astype(int), "spike_count": counts.astype(int)})
summary["firing_rate_hz_segment"] = summary["spike_count"] / max(float(spike_seconds.max() - spike_seconds.min()), 1e-9)
summary.to_csv(results_dir / "cluster_summary_initial.csv", index=False)
print(summary.sort_values("spike_count", ascending=False).to_string(index=False))
print(f"\nResumen guardado: {results_dir / 'cluster_summary_initial.csv'}")
print("Este resumen no clasifica unidades good/MUA/noise: eso requiere QC posterior, por ejemplo con Phy.")
