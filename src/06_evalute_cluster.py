"""Evaluación cuantitativa inicial de clusters exportados por Kilosort 4.

Lee los archivos ya generados por Kilosort. NO vuelve a ejecutar sorting.

Colocación recomendada: <proyecto>/src/06_evaluate_clusters.py
Ejecución: con el entorno donde está instalado Kilosort:
    python src/06_evaluate_clusters.py

Genera en <results_dir>/cluster_evaluation/:
    cluster_metrics.csv
    cluster_metrics.txt
    firing_rates.png
    quality_overview.png
    isi_distributions.png
    templates_by_cluster.png (si existe templates.npy)
    spike_positions_by_cluster.png (si existe spike_positions.npy)

Las etiquetas propuestas son filtros de revisión, no una identificación
biológica definitiva de neuronas individuales.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURACIÓN: si guardás este archivo dentro de src/, la ruta se resuelve
# automáticamente hacia <proyecto>/results/prueba_g0_t0.imec0.ap.
# Cambiá RESULTS_DIR si querés evaluar otra sesión.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "prueba_g0_t0.imec0.ap"

# Si Kilosort no guardó tmax y el binario original no está accesible, usá aquí
# la duración REAL del segmento procesado. Tus resultados compartidos antes
# correspondían a 60 s. Si esta sesión tiene otra duración, cambialo.
FALLBACK_SEGMENT_DURATION_S = 60.0

# Parámetros transparentes y editables de evaluación.
PRESENCE_BIN_SECONDS = 1.0
ISI_THRESHOLDS_MS = (1.0, 1.5, 2.0)
MIN_SPIKES_FOR_ASSESSMENT = 100
GOOD_MAX_CONTAMINATION_PCT = 10.0
BAD_MIN_CONTAMINATION_PCT = 20.0
GOOD_MIN_PRESENCE_RATIO = 0.50

# ---------------------------------------------------------------------------
# Estilo de figuras
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#555555",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#D9DDE3",
    "font.size": 9,
    "axes.titlesize": 11,
})


def as_float(value: Any, default: float | None = None) -> float | None:
    """Convierte valores NumPy/escalares a float de forma segura."""
    try:
        if value is None:
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def load_ops(results_dir: Path) -> dict:
    """Carga ops.npy con el helper oficial de Kilosort cuando está disponible."""
    ops_path = results_dir / "ops.npy"
    if not ops_path.exists():
        return {}
    try:
        from kilosort.io import load_ops as kilosort_load_ops
        ops = kilosort_load_ops(ops_path)
        return ops if isinstance(ops, dict) else {}
    except Exception:
        loaded = np.load(ops_path, allow_pickle=True)
        if getattr(loaded, "shape", None) == ():
            loaded = loaded.item()
        return loaded if isinstance(loaded, dict) else {}


def load_npy(results_dir: Path, filename: str, required: bool = True):
    path = results_dir / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Falta el archivo requerido: {path}")
        return None
    return np.load(path, mmap_mode="r" if path.stat().st_size > 100_000_000 else None)


def read_cluster_tsv(results_dir: Path, filename: str, value_columns: tuple[str, ...]) -> dict[int, Any]:
    """Lee TSV de Kilosort y devuelve {cluster_id: valor}."""
    path = results_dir / filename
    if not path.exists():
        print(f"Aviso: no está {filename}; esa métrica quedará vacía.")
        return {}

    result: dict[int, Any] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames:
            return result
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        id_col = columns.get("cluster_id") or columns.get("clusterid")
        value_col = next((columns.get(name.lower()) for name in value_columns
                          if columns.get(name.lower())), None)
        if id_col is None or value_col is None:
            print(f"Aviso: columnas no reconocidas en {filename}: {reader.fieldnames}")
            return result
        for row in reader:
            try:
                cid = int(float(row[id_col]))
                result[cid] = row[value_col]
            except (TypeError, ValueError):
                continue
    return result


def resolve_sampling_rate(ops: dict) -> float:
    settings = ops.get("settings", {}) if isinstance(ops, dict) else {}
    fs = as_float(ops.get("fs")) if isinstance(ops, dict) else None
    if fs is None:
        fs = as_float(settings.get("fs"))
    if fs is None or fs <= 0:
        raise ValueError("No pude leer la frecuencia de muestreo de ops.npy. Revisá ops['fs'].")
    return fs


def get_segment_bounds(ops: dict, fs: float) -> tuple[float, float, float, str]:
    """Devuelve tmin, tmax, duración y explicación de cómo se obtuvo."""
    settings = ops.get("settings", {}) if isinstance(ops, dict) else {}
    tmin = as_float(settings.get("tmin"), 0.0)
    if tmin is None:
        tmin = 0.0
    tmax = as_float(settings.get("tmax"))
    if tmax is not None and tmax > tmin:
        return tmin, tmax, tmax - tmin, "ops.npy: settings.tmin/tmax"

    # Cuando tmax=None (sesión completa), intentar estimar duración desde el
    # binario original registrado en ops. Para SpikeGLX int16: canales * 2 B.
    filenames = settings.get("filename") or settings.get("data_file_path")
    if isinstance(filenames, (str, Path)):
        candidates = [Path(filenames)]
    elif isinstance(filenames, (list, tuple)):
        candidates = [Path(p) for p in filenames]
    else:
        candidates = []

    n_chan = as_float(settings.get("n_chan_bin"))
    dtype_bytes = 2  # el pipeline de esta sesión interpreta el binario como int16
    for raw_path in candidates:
        try:
            if raw_path.exists() and n_chan and n_chan > 0:
                duration = raw_path.stat().st_size / (int(n_chan) * dtype_bytes * fs)
                if duration > tmin:
                    return tmin, duration, duration - tmin, "duración estimada desde el tamaño del .bin"
        except (OSError, TypeError, ValueError):
            pass

    duration = FALLBACK_SEGMENT_DURATION_S
    return tmin, tmin + duration, duration, (
        f"valor manual FALLBACK_SEGMENT_DURATION_S={FALLBACK_SEGMENT_DURATION_S:g} s"
    )


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator > 0 else float("nan")


def assess_cluster(ks_label: str, contam_pct: float,
                   presence: float, n_spikes: int) -> str:
    """Etiqueta conservadora para priorizar revisión, no veredicto neuronal.

    No usa la fracción bruta de ISI < 2 ms como regla de calidad porque depende
    de la tasa de disparo: una unidad rápida puede tener muchos ISI cortos aun
    sin contaminación. Para esta decisión se priorizan KSLabel y ContamPct.
    """
    if n_spikes < MIN_SPIKES_FOR_ASSESSMENT:
        return "too_few_spikes_to_assess"

    label = (ks_label or "").strip().lower()
    has_contam = math.isfinite(contam_pct)
    has_presence = math.isfinite(presence)

    if label == "mua":
        return "candidate_mua_or_contaminated"
    if has_contam and contam_pct >= BAD_MIN_CONTAMINATION_PCT:
        return "candidate_mua_or_contaminated"

    if (label == "good" and has_contam and has_presence
            and contam_pct <= GOOD_MAX_CONTAMINATION_PCT
            and presence >= GOOD_MIN_PRESENCE_RATIO):
        return "candidate_good_review_visually"

    return "needs_manual_review"


def main() -> None:
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de resultados:\n{RESULTS_DIR}\n\n"
            "Editá RESULTS_DIR al comienzo de este script para apuntar a tu carpeta."
        )

    print(f"Leyendo resultados: {RESULTS_DIR}")
    ops = load_ops(RESULTS_DIR)
    fs = resolve_sampling_rate(ops)
    tmin, tmax, duration_s, duration_source = get_segment_bounds(ops, fs)
    if duration_s <= 0:
        raise ValueError(f"Duración inválida: {duration_s}")

    spike_times = np.asarray(load_npy(RESULTS_DIR, "spike_times.npy")).reshape(-1).astype(np.float64)
    spike_clusters = np.asarray(load_npy(RESULTS_DIR, "spike_clusters.npy")).reshape(-1).astype(np.int64)
    if len(spike_times) != len(spike_clusters):
        raise ValueError(
            f"spike_times ({len(spike_times)}) y spike_clusters ({len(spike_clusters)}) "
            "no tienen la misma cantidad de elementos."
        )
    if len(spike_times) == 0:
        raise ValueError("No hay spikes en spike_times.npy.")

    amplitudes = load_npy(RESULTS_DIR, "amplitudes.npy", required=False)
    if amplitudes is not None:
        amplitudes = np.asarray(amplitudes).reshape(-1).astype(np.float64)
        if len(amplitudes) != len(spike_times):
            print("Aviso: amplitudes.npy no coincide en longitud; se omite esa métrica.")
            amplitudes = None

    positions = load_npy(RESULTS_DIR, "spike_positions.npy", required=False)
    if positions is not None:
        positions = np.asarray(positions)
        if positions.ndim != 2 or positions.shape[0] != len(spike_times) or positions.shape[1] < 2:
            print("Aviso: spike_positions.npy tiene una forma inesperada; se omite el mapa espacial.")
            positions = None

    ks_labels = read_cluster_tsv(RESULTS_DIR, "cluster_KSLabel.tsv", ("KSLabel", "group"))
    if not ks_labels:
        ks_labels = read_cluster_tsv(RESULTS_DIR, "cluster_group.tsv", ("group", "KSLabel"))
    contam_values = read_cluster_tsv(RESULTS_DIR, "cluster_ContamPct.tsv", ("ContamPct",))
    template_amplitudes = read_cluster_tsv(RESULTS_DIR, "cluster_Amplitude.tsv", ("Amplitude",))

    similar = load_npy(RESULTS_DIR, "similar_templates.npy", required=False)
    templates = load_npy(RESULTS_DIR, "templates.npy", required=False)

    cluster_ids, counts = np.unique(spike_clusters, return_counts=True)
    # Frecuencia con duración de segmento (no max(spike_times)-min(spike_times)).
    firing_rates = counts / duration_s

    rows: list[dict[str, Any]] = []
    per_cluster_isi: dict[int, np.ndarray] = {}
    per_cluster_depth: dict[int, float] = {}
    bin_count = max(1, int(math.ceil(duration_s / PRESENCE_BIN_SECONDS)))

    for idx, (cid_raw, count_raw) in enumerate(zip(cluster_ids, counts)):
        cid = int(cid_raw)
        mask = spike_clusters == cid
        times_s = spike_times[mask] / fs
        times_s = np.sort(times_s[np.isfinite(times_s)])
        isi_ms = np.diff(times_s) * 1000.0 if len(times_s) > 1 else np.array([], dtype=float)
        isi_ms = isi_ms[np.isfinite(isi_ms) & (isi_ms >= 0)]
        per_cluster_isi[cid] = isi_ms

        # Se limita al segmento de sorting para que presencia no incluya fuera de ventana.
        relative_s = times_s - tmin
        valid_rel = relative_s[(relative_s >= 0) & (relative_s < duration_s)]
        occupied_bins = np.unique(np.floor(valid_rel / PRESENCE_BIN_SECONDS).astype(int))
        presence = min(1.0, len(occupied_bins) / bin_count)

        isi_stats: dict[str, Any] = {}
        for threshold in ISI_THRESHOLDS_MS:
            violations = int(np.sum(isi_ms < threshold))
            denom = len(isi_ms)
            tag = str(threshold).replace(".", "p")
            isi_stats[f"isi_lt_{tag}ms_count"] = violations
            isi_stats[f"isi_lt_{tag}ms_pct"] = pct(violations, denom)

        unit_amps = amplitudes[mask] if amplitudes is not None else np.array([], dtype=float)
        unit_amps = unit_amps[np.isfinite(unit_amps)]
        amp_median = float(np.median(unit_amps)) if unit_amps.size else float("nan")
        amp_cv = (float(np.std(unit_amps) / np.mean(unit_amps))
                  if unit_amps.size and np.mean(unit_amps) != 0 else float("nan"))

        median_x = median_y = float("nan")
        if positions is not None:
            unit_pos = positions[mask]
            if len(unit_pos):
                finite_pos = unit_pos[np.all(np.isfinite(unit_pos[:, :2]), axis=1)]
                if len(finite_pos):
                    median_x = float(np.median(finite_pos[:, 0]))
                    median_y = float(np.median(finite_pos[:, 1]))
                    per_cluster_depth[cid] = median_y

        ks_label = str(ks_labels.get(cid, ""))
        contam_raw = as_float(contam_values.get(cid))
        # Kilosort suele exportar ContamPct en porcentaje. Si el valor viene
        # entre 0 y 1, también se deja como porcentaje tal como está en el TSV.
        contam_pct = contam_raw if contam_raw is not None else float("nan")
        isi2_pct = isi_stats["isi_lt_2p0ms_pct"]
        quality = assess_cluster(ks_label, contam_pct, presence, int(count_raw))

        template_amp = as_float(template_amplitudes.get(cid))
        rows.append({
            "cluster_id": cid,
            "spike_count": int(count_raw),
            "firing_rate_hz": float(count_raw) / duration_s,
            "KSLabel": ks_label or "not_available",
            "ContamPct_Kilosort": contam_pct,
            **isi_stats,
            "presence_ratio_1s": presence,
            "isi_median_ms": float(np.median(isi_ms)) if isi_ms.size else float("nan"),
            "amplitude_median_PC_L2_units": amp_median,
            "amplitude_CV": amp_cv,
            "template_amplitude_PC_L2_units": template_amp if template_amp is not None else float("nan"),
            "median_x_um": median_x,
            "median_y_um": median_y,
            "proposed_review_category": quality,
        })

    # Similaridad máxima con otro cluster; útil para detectar posibles duplicados,
    # pero no basta por sí sola para fusionar ni descartar clusters.
    if similar is not None:
        sim = np.asarray(similar, dtype=float)
        if sim.ndim == 2 and sim.shape[0] == sim.shape[1] and sim.shape[0] == len(cluster_ids):
            for i, row in enumerate(rows):
                values = sim[i].copy()
                values[i] = np.nan
                row["max_similarity_other_cluster"] = (
                    float(np.nanmax(values)) if np.any(np.isfinite(values)) else float("nan")
                )
        else:
            print("Aviso: similar_templates.npy no coincide con los clusters presentes; se omite similitud.")
            for row in rows:
                row["max_similarity_other_cluster"] = float("nan")
    else:
        for row in rows:
            row["max_similarity_other_cluster"] = float("nan")

    # Orden de mayor a menor cantidad de spikes.
    rows.sort(key=lambda r: r["spike_count"], reverse=True)
    out_dir = RESULTS_DIR / "cluster_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV UTF-8 compatible con Excel.
    csv_path = out_dir / "cluster_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # -----------------------------------------------------------------------
    # Figuras
    # -----------------------------------------------------------------------
    ids = [r["cluster_id"] for r in rows]
    counts_plot = [r["spike_count"] for r in rows]
    rates_plot = [r["firing_rate_hz"] for r in rows]
    contamination_plot = [r["ContamPct_Kilosort"] for r in rows]
    presence_plot = [r["presence_ratio_1s"] for r in rows]
    isi2_plot = [r["isi_lt_2p0ms_pct"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, counts_plot)
    ax.set_yticks(y, [str(i) for i in ids])
    ax.invert_yaxis()
    ax.set_title("Cantidad de spikes por cluster")
    ax.set_xlabel("Cantidad de spikes")
    ax.set_ylabel("Cluster ID")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "firing_rates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Contaminación Kilosort frente a violaciones simples ISI <2 ms.
    valid = np.isfinite(contamination_plot) & np.isfinite(isi2_plot)
    if np.any(valid):
        sc = axes[0].scatter(
            np.asarray(contamination_plot)[valid], np.asarray(isi2_plot)[valid],
            c=np.asarray(rates_plot)[valid], s=55, cmap="viridis", edgecolors="black", linewidths=0.3
        )
        for j, cid in enumerate(ids):
            if valid[j]:
                axes[0].annotate(str(cid), (contamination_plot[j], isi2_plot[j]), xytext=(4, 3),
                                 textcoords="offset points", fontsize=8)
        fig.colorbar(sc, ax=axes[0], label="Firing rate (Hz)")
    else:
        axes[0].text(0.5, 0.5, "No hay métricas suficientes", ha="center", va="center",
                    transform=axes[0].transAxes)
    axes[0].axvline(GOOD_MAX_CONTAMINATION_PCT, linestyle="--", linewidth=1, label="10% orientativo")
    axes[0].set_title("Contaminación Kilosort vs. ISI cortos")
    axes[0].set_xlabel("ContamPct de Kilosort (%)")
    axes[0].set_ylabel("ISI < 2 ms (% de intervalos)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].bar([str(i) for i in ids], presence_plot)
    axes[1].axhline(GOOD_MIN_PRESENCE_RATIO, linestyle="--", linewidth=1, label="0,50 orientativo")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Presencia durante el segmento")
    axes[1].set_xlabel("Cluster ID")
    axes[1].set_ylabel("Presence ratio (bins de 1 s)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Histogramas ISI por cluster. Ventana 0–100 ms para visualizar refractariedad.
    ncols = 3
    nrows = int(math.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.0 * nrows), squeeze=False)
    for i, cid in enumerate(ids):
        ax = axes[i // ncols][i % ncols]
        isi = per_cluster_isi[cid]
        if isi.size:
            ax.hist(isi[isi <= 100], bins=50)
        else:
            ax.text(0.5, 0.5, "Sin suficientes ISI", ha="center", va="center",
                    transform=ax.transAxes)
        ax.axvline(1.0, color="orange", linestyle="--", linewidth=1, label="1 ms")
        ax.axvline(2.0, color="red", linestyle="--", linewidth=1, label="2 ms")
        ax.set_title(f"Cluster {cid} (n={int(counts_plot[ids.index(cid)])})")
        ax.set_xlabel("ISI (ms)")
        ax.set_ylabel("Intervalos")
        ax.grid(alpha=0.2)
    for j in range(len(ids), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Distribución de intervalos entre spikes por cluster", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "isi_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Forma promedio (template) del canal con mayor amplitud peak-to-peak.
    if templates is not None:
        temp_arr = np.asarray(templates)
        if temp_arr.ndim == 3 and temp_arr.shape[0] == len(cluster_ids):
            fig, axes = plt.subplots(len(cluster_ids), 1, figsize=(9, max(4, 1.45 * len(cluster_ids))), squeeze=False)
            time_ms = np.arange(temp_arr.shape[1]) / fs * 1000.0
            for i, cid_raw in enumerate(cluster_ids):
                wave = temp_arr[i]
                ptp_by_channel = np.nanmax(wave, axis=0) - np.nanmin(wave, axis=0)
                peak_ch = int(np.nanargmax(ptp_by_channel))
                waveform = wave[:, peak_ch]
                ax = axes[i, 0]
                ax.plot(time_ms, waveform, linewidth=1.2)
                ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
                ax.set_ylabel(f"C{int(cid_raw)}\ncanal {peak_ch}")
                ax.grid(alpha=0.2)
            axes[-1, 0].set_xlabel("Tiempo dentro del template (ms)")
            fig.suptitle("Templates promedio por cluster (canal de mayor amplitud)", y=1.005)
            fig.tight_layout()
            fig.savefig(out_dir / "templates_by_cluster.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
        else:
            print("Aviso: templates.npy no tiene la forma esperada; no se genera templates_by_cluster.png.")

    if positions is not None:
        # Un punto por spike; el tamaño moderado evita cargar demasiado la figura.
        max_points = 100_000
        if len(spike_clusters) > max_points:
            rng = np.random.default_rng(0)
            sample_idx = np.sort(rng.choice(len(spike_clusters), max_points, replace=False))
        else:
            sample_idx = np.arange(len(spike_clusters))
        fig, ax = plt.subplots(figsize=(7, 9))
        sc = ax.scatter(positions[sample_idx, 0], positions[sample_idx, 1],
                        c=spike_clusters[sample_idx], s=3, cmap="turbo", alpha=0.65,
                        linewidths=0)
        fig.colorbar(sc, ax=ax, label="Cluster ID")
        ax.set_title("Posición espacial de los spikes")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y / profundidad (µm)")
        ax.invert_yaxis()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "spike_positions_by_cluster.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Informe de texto
    # -----------------------------------------------------------------------
    report_path = out_dir / "cluster_metrics.txt"
    with report_path.open("w", encoding="utf-8") as stream:
        stream.write("EVALUACIÓN INICIAL DE CLUSTERS — KILOSORT 4\n")
        stream.write("=" * 62 + "\n")
        stream.write(f"Carpeta: {RESULTS_DIR}\n")
        stream.write(f"Frecuencia de muestreo: {fs:g} Hz\n")
        stream.write(f"Segmento: {tmin:g}–{tmax:g} s ({duration_s:g} s)\n")
        stream.write(f"Origen de duración: {duration_source}\n")
        stream.write(f"Spikes totales: {len(spike_times):,}\n")
        stream.write(f"Clusters con spikes: {len(cluster_ids)}\n\n")
        stream.write("CLASIFICACIÓN PROPUESTA (solo para priorizar revisión)\n")
        for category in sorted({r["proposed_review_category"] for r in rows}):
            n_cat = sum(r["proposed_review_category"] == category for r in rows)
            stream.write(f"  {category}: {n_cat}\n")
        stream.write("\nCRITERIOS EDITABLES EN EL SCRIPT\n")
        stream.write(f"  mínimo de spikes para evaluar: {MIN_SPIKES_FOR_ASSESSMENT}\n")
        stream.write(f"  contaminación Kilosort para candidato bueno: <= {GOOD_MAX_CONTAMINATION_PCT:g}%\n")
        stream.write(f"  presence ratio para candidato bueno: >= {GOOD_MIN_PRESENCE_RATIO:g}\n")
        stream.write("\nIMPORTANTE\n")
        stream.write("- candidate_good_review_visually NO significa neurona confirmada.\n")
        stream.write("- too_few_spikes_to_assess NO significa ruido: hay pocos eventos para evaluar.\n")
        stream.write("- ISI < 1/1.5/2 ms es descriptivo; su porcentaje depende de la firing rate y no se usa solo para clasificar.\n")
        stream.write("- amplitudes.npy son normas L2 de PC features, no amplitud en microvoltios.\n")
        stream.write("- Revisar waveforms, autocorrelogramas, estabilidad y ubicación antes de interpretar unidades.\n")
        stream.write("\nTABLA RESUMIDA\n")
        for r in rows:
            stream.write(
                f"Cluster {r['cluster_id']:>4}: n={r['spike_count']:>7}, "
                f"FR={r['firing_rate_hz']:>8.3f} Hz, KS={r['KSLabel']}, "
                f"ContamPct={r['ContamPct_Kilosort']}, "
                f"ISI<2ms={r['isi_lt_2p0ms_pct']}, "
                f"presence={r['presence_ratio_1s']:.3f}, "
                f"propuesta={r['proposed_review_category']}\n"
            )

    print("\nEvaluación terminada.")
    print(f"Duración usada: {duration_s:g} s ({duration_source})")
    print(f"Clusters: {len(rows)} | Spikes: {len(spike_times):,}")
    print(f"CSV: {csv_path}")
    print(f"Informe: {report_path}")
    print(f"Gráficos: {out_dir}")
    print("\nNo se modificaron ni los archivos originales ni los resultados de Kilosort.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
