from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "config" / "probe_contacts.csv"

probe = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(7, 10))

for shank, group in probe.groupby("shank"):
    ax.scatter(
        group["x_um"],
        group["y_um"],
        s=12,
        label=f"Shank {shank}"
    )

ax.set_xlabel("X (µm)")
ax.set_ylabel("Y (µm)")
ax.set_title("Neuropixels 2.0 - Probe geometry")
ax.invert_yaxis()
ax.set_aspect("equal")
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(ROOT / "results" / "probe_geometry.png", dpi=200)
plt.show()