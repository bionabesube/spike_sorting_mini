"""Ejecuta inspección y validación. El sorting se lanza por separado y de forma explícita."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
scripts = [
    root / "src" / "01_inspect_spikeglx.py",
    root / "src" / "03_validate.py",
]
for script in scripts:
    print(f"\n>>> {script.name}")
    subprocess.run([sys.executable, str(script)], cwd=root, check=True)
print("\nInspección y validación terminadas. Para ordenar, ejecutá: python src/04_run_kilosort.py")
