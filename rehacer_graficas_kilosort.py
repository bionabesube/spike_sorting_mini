"""Regenera gráficos claros a partir de los resultados guardados de Kilosort 4.

No vuelve a ejecutar el spike sorting. Lee ops.npy y los archivos .npy exportados
por Kilosort, y guarda las figuras en results_dir/figures_light.

Nota: diagnostics y drift_scatter son resúmenes construidos con los spikes
finales exportados. No son una reproducción idéntica de las figuras internas
originales de Kilosort, que utilizan variables intermedias no exportadas.
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # guardar PNG sin abrir ventanas
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# CAMBIÁ ESTA RUTA SOLO SI TUS RESULTADOS ESTÁN EN OTRA CARPETA
# ------------------------------------------------------------
RESULTS_DIR = Path(
    r"C:\Users\User\Desktop\Neurobiologia del Sueño\spike sorting\aog_kilosort4\results\prueba_g0_t0.imec0.ap"
)

# Tema claro consistente
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#444444",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#D9DDE3",
    "font.size": 10,
    "axes.titlesize": 12,
})


def load_array(name, required=True):
    path = RESULTS_DIR / name
    if not path.exists():
        if required:
            raise FileNotFoundError(f"No encuentro {path}")
        return None
    return np.load(path, allow_pickle=True)


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_ops():
    path = RESULTS_DIR / "ops.npy"
    if not path.exists():
        raise FileNotFoundError(f"No encuentro {path}")
    # load_ops es el lector recomendado por Kilosort; fallback para versiones
    # donde ops.npy es directamente un diccionario serializado con NumPy.
    try:
        from kilosort.io import load_ops as kilosort_load_ops
        return kilosort_load_ops(path)
    except Exception:
        loaded = np.load(path, allow_pickle=True)
        if hasattr(loaded, "item"):
            return loaded.item()
        raise


def save(fig, out_dir, filename):
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OK: {path}")


def plot_drift_amount(ops, out_dir):
    if "dshift" not in ops:
        print("Omitido drift amount: ops.npy no contiene 'dshift'.")
        return
    dshift = as_numpy(ops["dshift"])
    if dshift.ndim == 1:
        dshift = dshift[:, None]
    settings = ops.get("settings", {})
    fs = float(settings.get("fs", ops.get("fs", 30000)))
    batch_size = float(settings.get("batch_size", 60000))
    tmin = float(settings.get("tmin", 0) or 0)
    times = np.arange(dshift.shape[0]) * (batch_size / fs) + tmin

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for section in range(dshift.shape[1]):
        ax.plot(times, dshift[:, section], marker="o", markersize=3,
                linewidth=1.6, label=f"Sección {section + 1}")
    ax.set_title("Deriva estimada por sección de la sonda")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Desplazamiento en profundidad (µm)")
    ax.grid(True, alpha=0.7)
    if dshift.shape[1] > 1:
        ax.legend(frameon=False)
    save(fig, out_dir, "drift_amount_light.png")


def get_spike_data():
    times = np.asarray(load_array("spike_times.npy")).reshape(-1)
    clusters = np.asarray(load_array("spike_clusters.npy")).reshape(-1)
    positions = np.asarray(load_array("spike_positions.npy"))
    amplitudes = np.asarray(load_array("amplitudes.npy")).reshape(-1)

    n = min(len(times), len(clusters), len(positions), len(amplitudes))
    if n == 0:
        raise ValueError("Los archivos de resultados no contienen spikes.")
    if not (len(times) == len(clusters) == len(positions) == len(amplitudes)):
        print("Aviso: los arrays no tienen igual longitud; se usará el tramo común.")
    return times[:n], clusters[:n], positions[:n], amplitudes[:n]


def plot_drift_scatter(times, positions, amplitudes, fs, out_dir):
    # Representación de la posición de los spikes finales a lo largo del tiempo.
    # La figura interna de Kilosort usa spikes candidatos intermedios (st0),
    # que normalmente no se guardan como archivo independiente.
    time_s = times / fs
    depth = positions[:, 1]
    amp = np.maximum(amplitudes, 1e-8)

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(time_s, depth, c=amp, s=5, cmap="viridis",
                    alpha=0.75, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Amplitud del spike (unidades exportadas)")
    ax.set_title("Posición en profundidad de los spikes a lo largo del tiempo")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Profundidad (µm)")
    ax.grid(True, alpha=0.25)
    save(fig, out_dir, "drift_scatter_light.png")


def plot_spike_positions(clusters, positions, out_dir):
    # Se conserva la orientación del gráfico original de Kilosort:
    # profundidad en X y posición lateral en Y.
    fig, ax = plt.subplots(figsize=(8, 9))
    sc = ax.scatter(positions[:, 1], positions[:, 0], c=clusters,
                    cmap="turbo", s=4, alpha=0.7, linewidths=0)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("ID de cluster")
    ax.set_title("Posición de los spikes sobre la sonda")
    ax.set_xlabel("Profundidad (µm)")
    ax.set_ylabel("Posición lateral (µm)")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("auto")
    save(fig, out_dir, "spike_positions_light.png")


def plot_diagnostics(times, clusters, amplitudes, fs, out_dir):
    # Resumen calculado a partir de los spikes finales exportados.
    unit_ids, inverse, counts = np.unique(
        clusters, return_inverse=True, return_counts=True
    )
    duration = max((times.max() - times.min()) / fs, 1 / fs)
    rates = counts / duration
    median_amp = np.array([
        np.median(amplitudes[inverse == i]) for i in range(len(unit_ids))
    ])

    # ISI intra-cluster (ms), evitando mezclar spikes de distintas unidades.
    all_isis = []
    for i in range(len(unit_ids)):
        unit_times = np.sort(times[inverse == i]) / fs
        if len(unit_times) > 1:
            isi_ms = np.diff(unit_times) * 1000
            all_isis.append(isi_ms[(isi_ms > 0) & (isi_ms < 1000)])
    isi_ms = np.concatenate(all_isis) if all_isis else np.array([])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # A: número de spikes por unidad (hasta 30 unidades más activas)
    order = np.argsort(counts)[::-1][:30]
    axes[0, 0].bar(np.arange(len(order)), counts[order], color="#3977A8")
    axes[0, 0].set_title("Spikes por unidad (máximo 30)")
    axes[0, 0].set_xlabel("Unidades ordenadas por cantidad de spikes")
    axes[0, 0].set_ylabel("Cantidad de spikes")
    axes[0, 0].grid(axis="y", alpha=0.25)

    # B: amplitud mediana frente a tasa de disparo
    axes[0, 1].scatter(rates, median_amp, c=counts, cmap="viridis", s=24)
    axes[0, 1].set_title("Amplitud mediana vs. tasa de disparo")
    axes[0, 1].set_xlabel("Tasa aproximada (spikes/s)")
    axes[0, 1].set_ylabel("Amplitud mediana")
    axes[0, 1].grid(True, alpha=0.25)

    # C: distribución de intervalos entre spikes de una misma unidad
    if isi_ms.size:
        bins = np.logspace(np.log10(max(isi_ms.min(), 0.05)),
                           np.log10(max(isi_ms.max(), 1)), 45)
        axes[1, 0].hist(isi_ms, bins=bins, color="#5B9B78", alpha=0.9)
        axes[1, 0].axvline(2, color="#C44747", linestyle="--",
                           linewidth=1.5, label="2 ms")
        axes[1, 0].set_xscale("log")
        axes[1, 0].legend(frameon=False)
    else:
        axes[1, 0].text(0.5, 0.5, "No hay suficientes spikes para calcular ISI",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("Intervalos entre spikes de cada unidad")
    axes[1, 0].set_xlabel("ISI (ms, escala logarítmica)")
    axes[1, 0].set_ylabel("Frecuencia")
    axes[1, 0].grid(True, alpha=0.2)

    # D: amplitud por unidad
    axes[1, 1].hist(amplitudes[np.isfinite(amplitudes)], bins=40,
                    color="#D18B45", alpha=0.9)
    axes[1, 1].set_title("Distribución de amplitudes de spikes")
    axes[1, 1].set_xlabel("Amplitud")
    axes[1, 1].set_ylabel("Cantidad de spikes")
    axes[1, 1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Resumen de resultados Kilosort 4 — {len(unit_ids)} clusters, {len(times):,} spikes",
        fontsize=14,
    )
    save(fig, out_dir, "diagnostics_light.png")


def main():
    if not RESULTS_DIR.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta de resultados:\n{RESULTS_DIR}\n\n"
            "Editá RESULTS_DIR al comienzo del script."
        )

    out_dir = RESULTS_DIR / "figures_light"
    out_dir.mkdir(exist_ok=True)

    ops = load_ops()
    settings = ops.get("settings", {})
    fs = float(settings.get("fs", ops.get("fs", 30000)))

    plot_drift_amount(ops, out_dir)
    times, clusters, positions, amplitudes = get_spike_data()
    plot_drift_scatter(times, positions, amplitudes, fs, out_dir)
    plot_spike_positions(clusters, positions, out_dir)
    plot_diagnostics(times, clusters, amplitudes, fs, out_dir)

    print("\nTerminado. Las figuras nuevas están en:")
    print(out_dir)
    print("\nNo se modificaron los PNG originales ni los datos del sorting.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
