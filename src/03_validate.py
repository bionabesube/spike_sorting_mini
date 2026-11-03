from common import load_config, resolve_project_path, read_spikeglx_meta, parse_positive_int, parse_positive_float
import json

cfg = load_config()
bin_path = resolve_project_path(cfg["data_file"])
meta_path = bin_path.with_suffix(".meta")
probe_path = resolve_project_path(cfg["probe_file"])

if not bin_path.is_file() or not meta_path.is_file():
    raise FileNotFoundError("No se encuentran el .bin y/o su .meta")
meta = read_spikeglx_meta(meta_path)
n_chan_bin = parse_positive_int(meta, "nSavedChans")
fs = parse_positive_float(meta, "imSampRate")
if bin_path.stat().st_size % (n_chan_bin * 2):
    raise ValueError("El tamaño del binario no cuadra con nSavedChans e int16")

if not probe_path.is_file():
    raise FileNotFoundError(f"Falta el probe: {probe_path}. Completá probe_contacts.csv y ejecutá 02_make_probe.py")
with probe_path.open("r", encoding="utf-8") as f:
    probe = json.load(f)

required = ("chanMap", "xc", "yc", "kcoords", "n_chan")
for key in required:
    if key not in probe:
        raise ValueError(f"Falta '{key}' en el probe")
lengths = [len(probe[k]) for k in ("chanMap", "xc", "yc", "kcoords")]
if len(set(lengths)) != 1 or lengths[0] == 0:
    raise ValueError("chanMap, xc, yc y kcoords deben tener la misma cantidad de elementos y no estar vacíos")
if int(probe["n_chan"]) != lengths[0]:
    raise ValueError("n_chan debe coincidir con la cantidad de contactos incluidos en el probe")
chan_map = [int(x) for x in probe["chanMap"]]
if min(chan_map) < 0 or max(chan_map) >= n_chan_bin:
    raise ValueError("chanMap contiene índices fuera del binario")
if len(chan_map) != len(set(chan_map)):
    raise ValueError("chanMap contiene canales duplicados")

# Confirmar que el segmento solicitado cabe dentro de la grabación.
frames = bin_path.stat().st_size // (n_chan_bin * 2)
duration = frames / fs
start = float(cfg.get("start_time_seconds", 0))
test_duration = cfg.get("test_duration_seconds")
end = duration if test_duration is None else start + float(test_duration)
if start < 0 or end <= start or end > duration:
    raise ValueError(f"Ventana inválida: {start}-{end}s; duración grabación {duration:.2f}s")

print("Validación OK")
print(f"Canales del binario: {n_chan_bin}; contactos en probe: {len(chan_map)}")
print(f"Fs: {fs:g} Hz; duración grabación: {duration:.2f} s")
print(f"Ventana a ordenar: {start:.2f}–{end:.2f} s")
print("IMPORTANTE: verificá visualmente el mapa de probe antes de considerar científico el sorting.")
