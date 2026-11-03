"""Resumen de sorting Kilosort 4: metricas, MUA/SU/good, waveforms y shanks.

Usa SOLO los archivos exportados por Kilosort 4. No requiere .bin, .meta ni Phy.

Salidas en <results_dir>/sorting_review/:
  - cluster_metrics.csv
  - unit_counts.txt
  - waveforms_by_cluster.png
  - waveforms/cluster_<id>.png
  - shank_assignments.png (si hay geometria disponible)

Definiciones usadas:
  MUA: cluster_KSLabel == 'mua'.
  SU (etiqueta KS): cluster_KSLabel == 'good'. Es el criterio de
      refractariedad de Kilosort; no es una confirmacion biologica.
  Good para analisis: KSLabel == 'good', ContamPct < 20% y al menos
      MIN_SPIKES_FOR_GOOD spikes. Es un filtro conservador adicional,
      no una etiqueta oficial de Kilosort.

El shank se estima desde channel_shanks.npy y la energia de cada template.
Si el mapa guardado contiene un solo shank, el informe lo dice claramente:
no inventa asignaciones a los otros tres shanks.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "prueba_g0_t0.imec0.ap"
EXPECTED_SHANKS = 4
MIN_SPIKES_FOR_GOOD = 100
MAX_CONTAM_PCT_FOR_GOOD = 20.0
SHANK_ENERGY_THRESHOLD = 0.60
REFRACTORY_MS = 1.5


def load_npy(folder: Path, name: str):
    path = folder / name
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=True)
    except Exception as exc:
        warnings.warn(f"No se pudo leer {name}: {exc}")
        return None


def load_tsv(folder: Path, name: str):
    path = folder / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception as exc:
        warnings.warn(f"No se pudo leer {name}: {exc}")
        return None


def find_col(df: pd.DataFrame | None, names: list[str]):
    if df is None:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def tsv_map(df: pd.DataFrame | None, value_names: list[str]) -> dict[int, object]:
    if df is None or df.empty:
        return {}
    id_col = find_col(df, ["cluster_id", "clusterid", "id"])
    val_col = find_col(df, value_names)
    if id_col is None or val_col is None:
        return {}
    out = {}
    for _, row in df.iterrows():
        try:
            out[int(row[id_col])] = row[val_col]
        except (TypeError, ValueError):
            pass
    return out


def get_ops(folder: Path):
    fs = 30000.0
    duration = None
    tmin = 0.0
    ops = load_npy(folder, "ops.npy")
    if isinstance(ops, np.ndarray) and ops.shape == ():
        ops = ops.item()
    if isinstance(ops, dict):
        settings = ops.get("settings", {}) or {}
        try:
            fs = float(ops.get("fs", settings.get("fs", fs)))
        except (TypeError, ValueError):
            pass
        try:
            tmin = float(settings.get("tmin", 0.0))
            tmax = settings.get("tmax")
            if tmax is not None:
                duration = float(tmax) - tmin
        except (TypeError, ValueError):
            duration = None
    return fs, duration, tmin


def template_row_for_cluster(cluster_id: int, cluster_ids: np.ndarray,
                              n_templates: int, spike_templates: np.ndarray | None,
                              spike_clusters: np.ndarray) -> int | None:
    # En la exportacion normal de KS4 los IDs de cluster indexan templates.
    if 0 <= cluster_id < n_templates:
        return int(cluster_id)
    # Fallback: cuando hay un template por cluster y los IDs no son contiguos.
    if n_templates == len(cluster_ids):
        return int(np.where(cluster_ids == cluster_id)[0][0])
    # Fallback adicional si spike_templates conserva el indice del template.
    if spike_templates is not None and len(spike_templates) == len(spike_clusters):
        st = np.asarray(spike_templates)
        if st.ndim == 2:
            st = st[:, 0]
        st = st.ravel()
        vals = st[spike_clusters == cluster_id]
        vals = vals[(vals >= 0) & (vals < n_templates)]
        if vals.size:
            return int(np.bincount(vals.astype(int)).argmax())
    return None


def get_template_channels(templates_ind: np.ndarray | None, row: int,
                          n_template_channels: int, n_probe_channels: int):
    if templates_ind is None:
        if n_template_channels == n_probe_channels:
            return np.arange(n_template_channels, dtype=int)
        return None
    ti = np.asarray(templates_ind)
    if ti.ndim == 2 and row < ti.shape[0]:
        channels = np.asarray(ti[row]).ravel().astype(int)
    elif ti.ndim == 1 and ti.size == n_template_channels:
        channels = ti.astype(int)
    else:
        return None
    if len(channels) != n_template_channels:
        return None
    return channels


def human_shank_label(shank_id: int, all_shanks: np.ndarray) -> str:
    vals = np.asarray(all_shanks).astype(int)
    # Kilosort/probe maps commonly store 0-based kcoords; display 1-based.
    if np.min(vals) == 0:
        return f"Shank {shank_id + 1} (mapa={shank_id})"
    return f"Shank {shank_id} (mapa={shank_id})"


def infer_shank(template: np.ndarray | None, template_channels: np.ndarray | None,
                channel_shanks: np.ndarray | None):
    if template is None or template_channels is None or channel_shanks is None:
        return "No disponible", np.nan, {}
    shanks = np.asarray(channel_shanks).ravel()
    good = (template_channels >= 0) & (template_channels < len(shanks))
    if not np.any(good):
        return "No disponible", np.nan, {}
    chans = template_channels[good]
    wave = template[:, good]
    channel_shanks_for_template = shanks[chans]
    # Energy per channel based on peak-to-peak template amplitude.
    p2p = np.ptp(wave, axis=0)
    energy = p2p ** 2
    totals = {}
    for sh in np.unique(channel_shanks_for_template):
        totals[int(sh)] = float(np.sum(energy[channel_shanks_for_template == sh]))
    total_energy = sum(totals.values())
    if total_energy <= 0:
        return "Indeterminado", np.nan, totals
    dominant = max(totals, key=totals.get)
    fraction = totals[dominant] / total_energy
    if fraction < SHANK_ENERGY_THRESHOLD:
        return "Ambiguo / energia en varios shanks", fraction, totals
    return human_shank_label(int(dominant), shanks), fraction, totals


def main():
    parser = argparse.ArgumentParser(
        description="Resume metricas, etiquetas, waveforms y shanks de Kilosort 4."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help="Carpeta de resultados de Kilosort 4.")
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {results_dir}")

    out_dir = results_dir / "sorting_review"
    wave_dir = out_dir / "waveforms"
    wave_dir.mkdir(parents=True, exist_ok=True)

    spike_times = load_npy(results_dir, "spike_times.npy")
    spike_clusters = load_npy(results_dir, "spike_clusters.npy")
    templates = load_npy(results_dir, "templates.npy")
    templates_ind = load_npy(results_dir, "templates_ind.npy")
    spike_templates = load_npy(results_dir, "spike_templates.npy")
    channel_shanks = load_npy(results_dir, "channel_shanks.npy")
    channel_positions = load_npy(results_dir, "channel_positions.npy")
    spike_positions = load_npy(results_dir, "spike_positions.npy")
    amplitudes = load_npy(results_dir, "amplitudes.npy")

    if spike_times is None or spike_clusters is None:
        raise FileNotFoundError("Faltan spike_times.npy o spike_clusters.npy.")
    spike_times = np.asarray(spike_times).ravel()
    spike_clusters = np.asarray(spike_clusters).ravel()
    if len(spike_times) != len(spike_clusters):
        raise ValueError("spike_times.npy y spike_clusters.npy no coinciden en longitud.")

    fs, duration, tmin = get_ops(results_dir)
    times_s = spike_times.astype(float) / fs
    if duration is None or not np.isfinite(duration) or duration <= 0:
        duration = float(times_s.max() - times_s.min()) if times_s.size > 1 else 0.0
        warnings.warn("No se recupero la duracion desde ops.npy; se aproxima usando spike_times.")
    if duration <= 0:
        raise ValueError("No se pudo determinar una duracion valida.")

    clusters = np.sort(np.unique(spike_clusters)).astype(int)
    n_templates = int(templates.shape[0]) if templates is not None and templates.ndim == 3 else 0

    # Kilosort's native exported labels and metrics.
    ks_df = load_tsv(results_dir, "cluster_KSLabel.tsv")
    if ks_df is None:
        ks_df = load_tsv(results_dir, "cluster_group.tsv")
    contam_map = tsv_map(load_tsv(results_dir, "cluster_ContamPct.tsv"),
                         ["ContamPct", "contam_pct", "contamination"])
    amp_template_map = tsv_map(load_tsv(results_dir, "cluster_Amplitude.tsv"),
                               ["Amplitude", "amplitude"])
    label_map = tsv_map(ks_df, ["KSLabel", "group", "label"])

    if templates is not None and templates.ndim != 3:
        warnings.warn(f"templates.npy tiene forma inesperada: {templates.shape}; no se graficaran waveforms.")
        templates = None
        n_templates = 0

    if channel_shanks is not None:
        channel_shanks = np.asarray(channel_shanks).ravel().astype(int)
        unique_shanks = np.unique(channel_shanks)
    else:
        unique_shanks = np.array([], dtype=int)

    rows = []
    template_rows = {}
    waveform_records = []
    for cluster in clusters:
        mask = spike_clusters == cluster
        st = np.sort(times_s[mask])
        n_spikes = int(mask.sum())
        fr = n_spikes / duration
        isi_ms = np.diff(st) * 1000.0
        isi_pct = float(np.mean(isi_ms < REFRACTORY_MS) * 100) if isi_ms.size else np.nan

        raw_label = str(label_map.get(int(cluster), "unknown")).strip().lower()
        label = raw_label if raw_label in {"good", "mua", "noise", "unsorted"} else "unknown"
        try:
            contam = float(contam_map.get(int(cluster), np.nan))
        except (TypeError, ValueError):
            contam = np.nan
        if np.isfinite(contam) and 0 <= contam <= 1:
            contam *= 100.0
        try:
            ks_amp = float(amp_template_map.get(int(cluster), np.nan))
        except (TypeError, ValueError):
            ks_amp = np.nan

        amp_stats = {}
        if amplitudes is not None and np.asarray(amplitudes).size == len(spike_clusters):
            a = np.asarray(amplitudes)[mask].astype(float)
            a = a[np.isfinite(a)]
            amp_stats = {
                "amplitude_spike_median": float(np.median(a)) if a.size else np.nan,
                "amplitude_spike_mean": float(np.mean(a)) if a.size else np.nan,
                "amplitude_spike_cv_pct": float(np.std(a) / abs(np.mean(a)) * 100)
                    if a.size and np.mean(a) != 0 else np.nan,
            }
        else:
            amp_stats = {"amplitude_spike_median": np.nan,
                         "amplitude_spike_mean": np.nan,
                         "amplitude_spike_cv_pct": np.nan}

        active_seconds = np.unique(np.floor(st)).size if st.size else 0
        presence_1s = active_seconds / duration

        row_idx = template_row_for_cluster(int(cluster), clusters, n_templates,
                                           spike_templates, spike_clusters)
        template_rows[int(cluster)] = row_idx
        template = None
        channels = None
        shank_label = "No disponible"
        shank_conf = np.nan
        shank_energy = {}
        waveform_peak_channel = np.nan
        template_ptp = np.nan
        if templates is not None and row_idx is not None and 0 <= row_idx < n_templates:
            template = np.asarray(templates[row_idx], dtype=float)
            channels = get_template_channels(templates_ind, row_idx,
                                             template.shape[1],
                                             len(channel_shanks) if channel_shanks is not None else template.shape[1])
            template_ptp_by_ch = np.ptp(template, axis=0)
            peak_local = int(np.argmax(template_ptp_by_ch))
            template_ptp = float(template_ptp_by_ch[peak_local])
            if channels is not None and peak_local < len(channels):
                waveform_peak_channel = int(channels[peak_local])
            shank_label, shank_conf, shank_energy = infer_shank(
                template, channels, channel_shanks
            )

            # Plot the three strongest template channels for this cluster.
            x_ms = np.arange(template.shape[0]) / fs * 1000.0
            order = np.argsort(template_ptp_by_ch)[::-1]
            top = order[:min(3, len(order))]
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for rank, ch_local in enumerate(top):
                ch_global = int(channels[ch_local]) if channels is not None else int(ch_local)
                ax.plot(x_ms, template[:, ch_local],
                        label=f"Canal {ch_global}" + (" (pico)" if rank == 0 else ""),
                        linewidth=2.0 if rank == 0 else 1.2)
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
            ax.set_title(f"Cluster {cluster} | KS: {label.upper()} | {shank_label}")
            ax.set_xlabel("Tiempo desde el pico (ms)")
            ax.set_ylabel("Amplitud de template (unidades blanqueadas)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(wave_dir / f"cluster_{cluster}.png", dpi=180)
            plt.close(fig)
            waveform_records.append((int(cluster), label, shank_label, template, top, channels))

        # Shank estimate from spike_positions x-coordinate as a secondary check.
        spike_x_median = np.nan
        shank_from_position = "No disponible"
        if spike_positions is not None:
            sp = np.asarray(spike_positions)
            if sp.ndim == 2 and sp.shape[0] == len(spike_clusters) and sp.shape[1] >= 2:
                xs = sp[mask, 0].astype(float)
                xs = xs[np.isfinite(xs)]
                if xs.size:
                    spike_x_median = float(np.median(xs))
                    if channel_positions is not None and channel_shanks is not None:
                        cp = np.asarray(channel_positions)
                        if cp.ndim == 2 and cp.shape[0] == len(channel_shanks) and cp.shape[1] >= 1:
                            centers = {}
                            for sh in unique_shanks:
                                sx = cp[channel_shanks == sh, 0]
                                sx = sx[np.isfinite(sx)]
                                if sx.size:
                                    centers[int(sh)] = float(np.median(sx))
                            if centers:
                                nearest = min(centers, key=lambda k: abs(centers[k] - spike_x_median))
                                shank_from_position = human_shank_label(int(nearest), channel_shanks)

        # Native classifications vs added conservative reporting screen.
        is_mua = label == "mua"
        is_su_ks = label == "good"
        enough_spikes = n_spikes >= MIN_SPIKES_FOR_GOOD
        low_contam = np.isfinite(contam) and contam < MAX_CONTAM_PCT_FOR_GOOD
        is_good_screen = is_su_ks and enough_spikes and low_contam

        if is_good_screen:
            status = "GOOD_candidate"
        elif is_mua:
            status = "MUA"
        elif is_su_ks and not enough_spikes:
            status = "SU_KS_but_few_spikes"
        elif is_su_ks:
            status = "SU_KS_needs_review"
        else:
            status = label.upper() if label != "unknown" else "UNCLASSIFIED"

        rows.append({
            "cluster_id": int(cluster),
            "n_spikes": n_spikes,
            "firing_rate_hz": fr,
            "KSLabel": label,
            "ContamPct_Kilosort": contam,
            "cluster_template_amplitude": ks_amp,
            "ISI_below_1p5ms_pct_descriptive": isi_pct,
            "presence_seconds_1s_bins_descriptive": presence_1s,
            **amp_stats,
            "template_row": row_idx if row_idx is not None else np.nan,
            "template_peak_to_peak": template_ptp,
            "template_peak_channel": waveform_peak_channel,
            "shank_from_template": shank_label,
            "shank_energy_fraction": shank_conf,
            "median_spike_x_um": spike_x_median,
            "shank_from_spike_position": shank_from_position,
            "MUA_KS": is_mua,
            "SU_KS_candidate": is_su_ks,
            "GOOD_extra_screen": is_good_screen,
            "report_status": status,
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "cluster_metrics.csv", index=False)

    # Combined waveform overview.
    if waveform_records:
        ncols = 3
        nrows = int(np.ceil(len(waveform_records) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.4 * nrows), squeeze=False)
        for ax, (cluster, label, shank, template, top, channels) in zip(axes.flat, waveform_records):
            x_ms = np.arange(template.shape[0]) / fs * 1000.0
            ch_local = int(top[0])
            ax.plot(x_ms, template[:, ch_local], linewidth=1.8)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.set_title(f"Cluster {cluster} | {label.upper()}\n{shank}", fontsize=9)
            ax.set_xlabel("Tiempo (ms)")
            ax.set_ylabel("Template (unidades blanqueadas)")
            ax.grid(alpha=0.2)
        for ax in axes.flat[len(waveform_records):]:
            ax.axis("off")
        fig.suptitle("Formas de onda promedio de Kilosort 4 (template, no waveform cruda)", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "waveforms_by_cluster.png", dpi=220)
        plt.close(fig)
    else:
        warnings.warn("No se pudo crear figura de waveforms: falta templates.npy o no se mapearon los clusters.")

    # Plot probe channels colored by stored shank labels.
    if channel_positions is not None and channel_shanks is not None:
        cp = np.asarray(channel_positions)
        sh = np.asarray(channel_shanks).ravel()
        if cp.ndim == 2 and cp.shape[0] == len(sh) and cp.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 9))
            for sid in np.unique(sh):
                m = sh == sid
                ax.scatter(cp[m, 0], cp[m, 1], s=14,
                           label=human_shank_label(int(sid), sh))
            ax.set_xlabel("X (µm)")
            ax.set_ylabel("Y (µm)")
            ax.set_title("Geometría de canales guardada en Kilosort")
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "shank_assignments.png", dpi=220)
            plt.close(fig)

    # Counts summary.
    n_mua = int((metrics["KSLabel"] == "mua").sum())
    n_su_ks = int((metrics["KSLabel"] == "good").sum())
    n_good = int(metrics["GOOD_extra_screen"].sum())
    n_noise = int((metrics["KSLabel"] == "noise").sum())
    n_unknown = int((metrics["KSLabel"] == "unknown").sum())

    shank_note = "No se encontró channel_shanks.npy. No se puede identificar shank."
    if channel_shanks is not None:
        if len(unique_shanks) == 1:
            only = int(unique_shanks[0])
            shank_note = (
                f"El mapa guardado en Kilosort contiene 1 solo shank: "
                f"{human_shank_label(only, channel_shanks)}. Aunque la sonda física tenga "
                f"4 shanks, estos resultados solo permiten asignar unidades al shank "
                f"representado en el mapa; no se deben inventar asignaciones a los otros tres."
            )
        else:
            present = ", ".join(human_shank_label(int(s), channel_shanks) for s in unique_shanks)
            shank_note = f"Shanks representados en channel_shanks.npy: {present}."
            if len(unique_shanks) < EXPECTED_SHANKS:
                shank_note += f" La sonda indicada tiene {EXPECTED_SHANKS} shanks, pero el sorting contiene {len(unique_shanks)}."

    lines = [
        "KILOSORT 4 - RESUMEN DE SORTING",
        "================================",
        f"Carpeta: {results_dir}",
        f"Duracion: {duration:.3f} s | Frecuencia de muestreo: {fs:.1f} Hz",
        f"Clusters: {len(clusters)} | Spikes exportados: {len(spike_times)}",
        "",
        "CONTEO DE UNIDADES",
        f"MUA (KSLabel=mua): {n_mua}",
        f"SU candidatos segun Kilosort (KSLabel=good): {n_su_ks}",
        f"GOOD para analisis (KS good + ContamPct<20% + >= {MIN_SPIKES_FOR_GOOD} spikes): {n_good}",
        f"Noise: {n_noise}",
        f"Sin etiqueta reconocida: {n_unknown}",
        "",
        "IMPORTANTE: SU candidatos y GOOD para analisis no son una confirmacion biologica.",
        "KSLabel 'good' es la etiqueta automatica basada en refractariedad/correlograma.",
        "GOOD para analisis agrega un filtro conservador de cantidad de spikes y ContamPct.",
        "Las categorias no sustituyen la inspeccion experimental/manual.",
        "",
        "SHANKS",
        shank_note,
        "La asignacion por template se basa en la energia de la forma de onda promedio por shank.",
        "La geometria solo es tan correcta como el probe map usado al ejecutar Kilosort.",
        "",
        "FORMAS DE ONDA",
        "Las figuras usan templates.npy: formas promedio guardadas por Kilosort despues del procesamiento.",
        "No son waveforms crudas individuales; para extraerlas desde la senal original se necesita el .bin.",
        "",
        f"Tabla: {out_dir / 'cluster_metrics.csv'}",
        f"Waveforms: {out_dir / 'waveforms_by_cluster.png'}",
    ]
    (out_dir / "unit_counts.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nTABLA POR CLUSTER")
    cols = ["cluster_id", "n_spikes", "firing_rate_hz", "KSLabel",
            "ContamPct_Kilosort", "shank_from_template", "report_status"]
    print(metrics[cols].to_string(index=False))
    print(f"\nSalidas guardadas en: {out_dir}")


if __name__ == "__main__":
    main()
