from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# CONFIGURACION
# Si ejecutas el script desde src/, encuentra el proyecto solo.
# También puedes pasar otra carpeta de resultados como argumento.
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "prueba_g0_t0.imec0.ap"

MIN_SPIKES = 100
REFRACTORY_MS = 1.5
MAX_ISI_VIOLATIONS_PCT = 1.0
MAX_CONTAMINATION_PCT = 20.0
SIMILARITY_REVIEW_THRESHOLD = 0.80


def load_tsv(path: Path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except Exception as exc:
        warnings.warn(f"No pude leer {path.name}: {exc}")
        return None


def find_column(df, candidates):
    if df is None:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def table_to_dict(df, value_candidates):
    if df is None or df.empty:
        return {}
    id_col = find_column(df, ["cluster_id", "clusterid", "id"])
    value_col = find_column(df, value_candidates)
    if id_col is None or value_col is None:
        return {}
    result = {}
    for _, row in df.iterrows():
        try:
            result[int(row[id_col])] = row[value_col]
        except (TypeError, ValueError):
            continue
    return result


def load_ops(results_dir: Path):
    path = results_dir / "ops.npy"
    if not path.exists():
        return {}, None, None
    try:
        ops = np.load(path, allow_pickle=True).item()
        settings = ops.get("settings", {})
        fs = ops.get("fs", settings.get("fs", 30000))
        fs = float(fs)
        tmin = float(settings.get("tmin", 0.0))
        tmax = settings.get("tmax")
        duration = float(tmax) - tmin if tmax is not None else None
        return ops, fs, duration
    except Exception as exc:
        warnings.warn(f"No pude leer ops.npy: {exc}")
        return {}, None, None


def load_array(results_dir: Path, name):
    path = results_dir / name
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        warnings.warn(f"No pude leer {name}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evalua candidatos a unidades individuales con resultados guardados de Kilosort 4."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Carpeta de resultados de Kilosort (por defecto: results/prueba_g0_t0.imec0.ap).",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de resultados: {results_dir}")

    spike_times = load_array(results_dir, "spike_times.npy")
    spike_clusters = load_array(results_dir, "spike_clusters.npy")
    amplitudes = load_array(results_dir, "amplitudes.npy")
    similar_templates = load_array(results_dir, "similar_templates.npy")
    kept_spikes = load_array(results_dir, "kept_spikes.npy")

    if spike_times is None or spike_clusters is None:
        raise FileNotFoundError("Faltan spike_times.npy o spike_clusters.npy")

    spike_times = np.asarray(spike_times).ravel()
    spike_clusters = np.asarray(spike_clusters).ravel()
    if len(spike_times) != len(spike_clusters):
        raise ValueError("spike_times.npy y spike_clusters.npy no tienen la misma longitud.")

    ops, fs, duration = load_ops(results_dir)
    if fs is None:
        fs = 30000.0
        warnings.warn("No encontré fs en ops.npy; se asume 30000 Hz.")

    times_s = spike_times.astype(np.float64) / fs
    if duration is None or not np.isfinite(duration) or duration <= 0:
        duration = float(np.max(times_s)) if len(times_s) else 0.0
        warnings.warn("No pude recuperar la duración desde ops.npy; uso el último spike como aproximación.")

    ks_df = load_tsv(results_dir / "cluster_KSLabel.tsv")
    contam_df = load_tsv(results_dir / "cluster_ContamPct.tsv")
    if ks_df is None:
        # cluster_group.tsv puede existir, pero no siempre es la etiqueta automática de Kilosort.
        ks_df = load_tsv(results_dir / "cluster_group.tsv")

    ks_labels = table_to_dict(ks_df, ["KSLabel", "group", "label"])
    contamination = table_to_dict(contam_df, ["ContamPct", "contam_pct", "contamination"])

    clusters = np.sort(np.unique(spike_clusters)).astype(int)
    rows = []

    # La matriz de Kilosort tiene correlaciones entre clusters, no entre índices de templates.
    # Según la salida de Kilosort, su dimensión corresponde al número de clusters.
    similarity_by_cluster = {}
    if similar_templates is not None:
        sim = np.asarray(similar_templates)
        if sim.ndim == 2 and sim.shape == (len(clusters), len(clusters)):
            for i, cluster in enumerate(clusters):
                if len(clusters) > 1:
                    others = [(float(sim[i, j]), int(clusters[j]))
                              for j in range(len(clusters)) if j != i]
                    best_value, best_cluster = max(others, key=lambda item: item[0])
                    similarity_by_cluster[int(cluster)] = (best_value, best_cluster)
        else:
            warnings.warn(
                "similar_templates.npy no coincide con la cantidad de clusters; "
                "no se calculará la similitud entre clusters."
            )

    for cluster in clusters:
        mask = spike_clusters == cluster
        cluster_times = np.sort(times_s[mask])
        n_spikes = int(mask.sum())
        firing_rate = n_spikes / duration if duration > 0 else np.nan

        isi_ms = np.diff(cluster_times) * 1000.0
        isi_violations_pct = (
            float(np.mean(isi_ms < REFRACTORY_MS) * 100.0)
            if len(isi_ms) else np.nan
        )
        n_isi_violations = int(np.sum(isi_ms < REFRACTORY_MS)) if len(isi_ms) else 0
        min_isi_ms = float(np.min(isi_ms)) if len(isi_ms) else np.nan

        label = str(ks_labels.get(int(cluster), "unknown")).strip().lower()
        try:
            contam = float(contamination.get(int(cluster), np.nan))
        except (TypeError, ValueError):
            contam = np.nan

        if np.isfinite(contam) and 0 <= contam <= 1:
            # Algunas exportaciones representan la proporción como 0-1 y otras como 0-100.
            contam *= 100.0

        active_bins = np.unique(np.floor(cluster_times)).size if n_spikes else 0
        presence_1s = active_bins / duration if duration > 0 else np.nan

        amp_cv_pct = np.nan
        if amplitudes is not None and len(amplitudes) == len(spike_clusters):
            a = np.asarray(amplitudes[mask], dtype=float)
            a = a[np.isfinite(a)]
            if len(a) and np.mean(np.abs(a)) > 0:
                amp_cv_pct = float(np.std(a) / np.mean(np.abs(a)) * 100.0)

        max_similarity, most_similar_cluster = similarity_by_cluster.get(
            int(cluster), (np.nan, np.nan)
        )

        # Clasificación conservadora: no etiqueta una unidad como single-unit
        # solo por tener muchos spikes o una frecuencia alta.
        if n_spikes < MIN_SPIKES:
            category = "insufficient_spikes"
            reason = f"Menos de {MIN_SPIKES} spikes; no alcanza para evaluar con confianza."
        elif label == "good" and np.isfinite(contam) and contam < MAX_CONTAMINATION_PCT \
                and np.isfinite(isi_violations_pct) and isi_violations_pct <= MAX_ISI_VIOLATIONS_PCT:
            category = "candidate_single_unit"
            reason = "Etiqueta KS good y métricas de refractariedad/contaminación favorables; requiere inspección visual."
        elif label == "mua" or (np.isfinite(contam) and contam >= MAX_CONTAMINATION_PCT) \
                or (np.isfinite(isi_violations_pct) and isi_violations_pct > MAX_ISI_VIOLATIONS_PCT):
            category = "likely_mua_or_contaminated"
            reason = "Etiqueta MUA o evidencia de contaminación/violaciones del período refractario."
        else:
            category = "needs_review"
            reason = "No hay evidencia suficiente o coherente para clasificarlo automáticamente."

        if np.isfinite(max_similarity) and max_similarity >= SIMILARITY_REVIEW_THRESHOLD:
            reason += f" Alta similitud con cluster {int(most_similar_cluster)}; revisar posible duplicación/separación."

        rows.append({
            "cluster_id": int(cluster),
            "spike_count": n_spikes,
            "firing_rate_hz": firing_rate,
            "KSLabel": label,
            "ContamPct_Kilosort": contam,
            "ISI_violations_below_1p5ms_pct": isi_violations_pct,
            "ISI_violations_count": n_isi_violations,
            "minimum_ISI_ms": min_isi_ms,
            "presence_ratio_1s_bins_descriptive_only": presence_1s,
            "amplitude_CV_pct_descriptive_only": amp_cv_pct,
            "most_similar_cluster": most_similar_cluster,
            "similarity_score": max_similarity,
            "preliminary_category": category,
            "reason": reason,
        })

    result = pd.DataFrame(rows)

    # Marcar pares de clusters muy similares; no se fusionan ni eliminan automáticamente.
    if "similarity_score" in result.columns:
        result["possible_duplicate_pair"] = result["similarity_score"] >= SIMILARITY_REVIEW_THRESHOLD

    out_dir = results_dir / "unit_candidates"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "unit_candidates.csv"
    txt_path = out_dir / "unit_candidates_report.txt"
    result.to_csv(csv_path, index=False)

    # Comprobación de la máscara de spikes eliminados como duplicados.
    kept_note = "kept_spikes.npy no disponible."
    if kept_spikes is not None:
        kept = np.asarray(kept_spikes).astype(bool).ravel()
        if int(kept.sum()) == len(spike_times):
            kept_note = (
                f"kept_spikes consistente: {len(kept)} detecciones originales, "
                f"{int(kept.sum())} conservadas y {int((~kept).sum())} eliminadas."
            )
        else:
            kept_note = (
                f"kept_spikes: {len(kept)} valores; {int(kept.sum())} True; "
                f"{len(spike_times)} spikes exportados. No se aplica la máscara de nuevo."
            )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("KILOSORT 4 - EVALUACION DE CANDIDATOS A UNIDADES\n")
        f.write(f"Resultados: {results_dir}\n")
        f.write(f"Frecuencia de muestreo: {fs:g} Hz\n")
        f.write(f"Duracion evaluada: {duration:.3f} s\n")
        f.write(f"Total de spikes exportados: {len(spike_times)}\n")
        f.write(kept_note + "\n\n")
        f.write(result.to_string(index=False))
        f.write("\n\nInterpretacion: candidate_single_unit es un candidato, no una neurona confirmada.\n")
        f.write("La clasificacion usa etiquetas/contaminacion de Kilosort e ISI; no sustituye\n")
        f.write("la inspeccion de waveforms, autocorrelogramas y calidad de la senal cruda.\n")

    print("\nKILOSORT 4 - CANDIDATOS A UNIDADES")
    print("---------------------------------")
    print(f"Resultados: {results_dir}")
    print(f"FS: {fs:g} Hz | Duracion: {duration:.2f} s | Spikes: {len(spike_times)}")
    print("\n" + result[[
        "cluster_id", "spike_count", "firing_rate_hz", "KSLabel",
        "ContamPct_Kilosort", "ISI_violations_below_1p5ms_pct",
        "most_similar_cluster", "similarity_score", "preliminary_category"
    ]].to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"Informe: {txt_path}")
    print("\nNOTA: el script identifica candidatos; no puede confirmar por sí solo la identidad neuronal.")


if __name__ == "__main__":
    main()
