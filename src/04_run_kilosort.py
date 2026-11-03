import json
import torch
from kilosort import run_kilosort
from kilosort.io import load_probe
from common import PROJECT_ROOT, load_config, resolve_project_path, read_spikeglx_meta, parse_positive_int, parse_positive_float

cfg = load_config()
bin_path = resolve_project_path(cfg["data_file"])
meta_path = bin_path.with_suffix(".meta")
probe_path = resolve_project_path(cfg["probe_file"])
results_root = resolve_project_path(cfg["results_root"])

meta = read_spikeglx_meta(meta_path)
n_chan_bin = parse_positive_int(meta, "nSavedChans")
fs = parse_positive_float(meta, "imSampRate")
probe = load_probe(probe_path)

start = float(cfg.get("start_time_seconds", 0))
test_duration = cfg.get("test_duration_seconds")
settings = {"fs": fs, "n_chan_bin": n_chan_bin, "tmin": start}
if test_duration is not None:
    settings["tmax"] = start + float(test_duration)

results_dir = results_root / bin_path.stem
results_dir.mkdir(parents=True, exist_ok=True)

threads = int(cfg.get("torch_threads", 4))
torch.set_num_threads(threads)
print("Kilosort 4 en CPU")
print(f"Entrada en servidor: {bin_path}")
print(f"Resultados locales:  {results_dir}")
print(f"Ventana: {start}s a {settings.get('tmax', 'fin de grabación')}s")
print("La lectura desde Z: puede ser lenta. No desconectes la unidad de red.")

run_kilosort(
    settings=settings,
    probe=probe,
    filename=bin_path,
    results_dir=results_dir,
    data_dtype="int16",
    do_CAR=bool(cfg.get("do_CAR", True)),
    invert_sign=bool(cfg.get("invert_sign", False)),
    device=torch.device("cpu"),
    save_extra_vars=bool(cfg.get("save_extra_vars", False)),
    torch_thread_lim=threads,
)
print(f"\nSorting finalizado. Resultados: {results_dir}")
