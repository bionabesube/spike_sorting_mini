from pathlib import Path
from common import load_config, resolve_project_path, read_spikeglx_meta, parse_positive_int, parse_positive_float

cfg = load_config()
bin_path = resolve_project_path(cfg["data_file"])
if not bin_path.is_file():
    raise FileNotFoundError(f"No encuentro el binario o no está montada la unidad de red: {bin_path}")

# imec0.ap.bin -> imec0.ap.meta
meta_path = bin_path.with_suffix(".meta")
if not meta_path.is_file():
    raise FileNotFoundError(f"No encuentro el .meta correspondiente: {meta_path}")

meta = read_spikeglx_meta(meta_path)
n_saved = parse_positive_int(meta, "nSavedChans")
fs = parse_positive_float(meta, "imSampRate")
size_bytes = bin_path.stat().st_size
bytes_per_sample_frame = n_saved * 2  # SpikeGLX AP suele almacenarse como int16
remainder = size_bytes % bytes_per_sample_frame
n_frames = size_bytes // bytes_per_sample_frame
duration = n_frames / fs

print("\n=== SPIKEGLX AP INSPECTION ===")
print(f"BIN:          {bin_path}")
print(f"META:         {meta_path}")
print(f"Tamaño:       {size_bytes / 1e9:.3f} GB")
print(f"Canales guardados (nSavedChans): {n_saved}")
print(f"Frecuencia AP (imSampRate):      {fs:g} Hz")
print(f"Duración estimada:               {duration / 60:.2f} min ({duration:.1f} s)")
print(f"Bytes por frame (int16):         {bytes_per_sample_frame}")
print(f"Binario alineado:                {'SÍ' if remainder == 0 else 'NO'}")

for key in ("snsApLfSy", "imDatPrb_type", "imDatPrb_sn", "imroTbl", "snsGeomMap", "snsShankMap"):
    if key in meta:
        value = meta[key]
        print(f"{key}: {value[:500]}{' ...[recortado]' if len(value) > 500 else ''}")

if remainder:
    raise RuntimeError("El tamaño del .bin no es múltiplo de nSavedChans * 2. Revisá archivo/meta antes de seguir.")

print("\nInspección básica OK. Falta confirmar el mapa real de contactos antes de ejecutar Kilosort.")
