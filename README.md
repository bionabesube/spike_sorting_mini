# Mini spike sorting: de SpikeGLX a Kilosort 4
Explorar las etapas básicas del *spike sorting*: inspeccionar una grabación, definir la geometría de la sonda, ejecutar Kilosort y revisar los resultados. La idea es aprender qué hace cada paso de un pipeline, no ofrecer una herramienta lista para analizar cualquier grabación.

## ¿Qué hace?
A partir de una grabación de voltaje neuronal en formato binario de SpikeGLX (`.ap.bin`) y sus metadatos (`.ap.meta`), el proyecto:
1. Inspecciona la grabación y comprueba información básica, como el número de canales y la frecuencia de muestreo.
2. Prepara el mapa de contactos de la sonda a partir de las coordenadas y los canales indicados.
3. Ejecuta Kilosort 4 sobre un segmento de la grabación o sobre la sesión completa.
4. Guarda los resultados para resumirlos y revisarlos posteriormente.

Kilosort detecta eventos neuronales (*spikes*) y los agrupa en *clusters*. Un cluster puede corresponder a una neurona, pero el resultado requiere revisión y control de calidad.

## Requisitos
- Windows y PowerShell.
- Python 3.11 recomendado.
- Una grabación de SpikeGLX (`.ap.bin`) y su archivo `.ap.meta`.
- El mapa de canales y la geometría correctos para la sonda y la sesión.

El proyecto está configurado para ejecutar Kilosort en CPU. El procesamiento de grabaciones largas puede tomar bastante tiempo.

## Instalación
Desde la carpeta del proyecto:
```powershell
conda create -n kilosort4 python=3.11 -y
conda activate kilosort4
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuración
Edita `config/settings.json` y ajusta `data_file` para que apunte al archivo `.ap.bin` de tu sesión. Por ejemplo:
```json
{
  "data_file": "D:\\datos\\mi_sesion\\grabacion.imec0.ap.bin",
  "probe_file": "config\\probe_np2.json",
  "results_root": "results",
  "test_duration_seconds": 20,
  "start_time_seconds": 0,
  "torch_threads": 4,
  "do_CAR": true,
  "invert_sign": false,
  "save_extra_vars": false
}
```

El archivo `.meta` correspondiente debe estar junto al `.bin`. No publiques rutas locales que revelen información privada.

Antes de generar el mapa de la sonda, completa `config/probe_contacts.csv` con los contactos neuronales realmente grabados:

- `binary_channel`: índice del canal en el binario, empezando en 0.
- `x_um` y `y_um`: coordenadas físicas del contacto en micrómetros.
- `shank`: identificador del shank.

Usa el mapa correspondiente a la sonda y a la configuración de adquisición. No inventes coordenadas ni asumas que todos los canales guardados son canales neuronales.

## Flujo de trabajo
Ejecuta los pasos desde la carpeta del proyecto:

```powershell
python src/01_inspect_spikeglx.py
python src/02_make_probe.py
python src/03_validate.py
python src/04_run_kilosort.py
```
Para la primera prueba, configura un segmento corto con `test_duration_seconds`. Cuando hayas comprobado que funciona, puedes usar `null` para procesar toda la grabación. El tiempo necesario dependerá de la duración, el número de canales y el equipo.
También puedes ejecutar `python main.py` para realizar la inspección y la validación iniciales. **Ese comando no ejecuta Kilosort**; el sorting se lanza por separado con `python src/04_run_kilosort.py`.

## Resultados
Los resultados se guardan bajo `results/`, en una carpeta asociada a la grabación. Según la versión de Kilosort, pueden incluir tiempos de spikes, asignaciones de clusters, plantillas, métricas y gráficas.
Los scripts adicionales en `src/` sirven para generar resúmenes y apoyar la evaluación de los clusters. Las métricas son herramientas de orientación, no una confirmación automática de que una unidad sea válida.


## Referencias

### Spike sorting
- [Kilosort 4 - Official Documentation](https://kilosort.readthedocs.io/en/latest/)
- Pachitariu, M., Sridhar, S., Pennington, J., & Stringer, C. (2024). [Spike sorting with Kilosort4](https://doi.org/10.1038/s41592-024-02232-7). Nature Methods, 21, 914–921.

### Data acquisition and probe geometry
- [SpikeGLX - User Manual](https://billkarsh.github.io/SpikeGLX/Sgl_help/UserManual.html)
- [Neuropixels 2.0 - IMEC](https://www.neuropixels.org/probe2-0)
- Steinmetz, N. A., et al. (2021). [Neuropixels 2.0: A miniaturized high-density probe for stable, long-term brain recordings](https://doi.org/10.1126/science.abf4588). Science, 372(6539).

### Spike sorting quality control
- [Phy - Documentation](https://phy.readthedocs.io/en/latest/)
- [SpikeInterface - Quality Metrics](https://spikeinterface.readthedocs.io/en/stable/modules/metrics/quality_metrics.html)
- Fabre, J. M. J., et al. (2023). [Bombcell: Automated curation and cell classification of spike-sorted electrophysiology data](https://doi.org/10.5281/zenodo.8172821).

