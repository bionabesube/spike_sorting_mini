# AOG15 — SpikeGLX → Kilosort 4 (CPU)

Pipeline mínimo para ejecutar Kilosort 4 desde VS Code en Windows, leyendo `imec0.ap.bin` desde la unidad de red `Z:` y guardando resultados en el disco local.

**No usa SpikeInterface. No convierte ni copia el binario.** Kilosort lee el `.bin` por bloques. La lectura desde red puede ser lenta; no desconectes la unidad Z: durante el proceso.

## 1. Entorno (PowerShell terminal de VS Code)

```powershell
conda create -n kilosort4 python=3.11 -y
conda activate kilosort4
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verificá CPU:

```powershell
python -c "import torch; print('torch', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())"
```

Debe indicar `CUDA disponible: False` en tu ThinkPad. Se ejecutará en CPU.

## 2. Configurar la grabación

Editá `config/settings.json`. La ruta de ejemplo apunta a la grabación de prueba que aparece en el listado que compartiste:

`Z:\AOG15\prueba_g0\prueba_g0_imec0\prueba_g0_t0.imec0.ap.bin`

El `.meta` se busca automáticamente cambiando `.bin` por `.meta`.

## 3. Inspeccionar primero (no ejecuta Kilosort)

```powershell
python src/01_inspect_spikeglx.py
```

El script muestra `nSavedChans`, `imSampRate`, tamaño, duración estimada y comprueba que el tamaño del `.bin` sea múltiplo de `nSavedChans * 2` bytes (int16). Si el binario está incompleto o los metadatos no coinciden, no sigas al sorting.

## 4. Preparar el mapa de la sonda

Kilosort necesita un mapa real de los contactos usados en esa grabación. No se debe asumir que todos los canales guardados son canales neuronales ni inventar el mapa de Neuropixels 2.0.

Completá `config/probe_contacts.csv` con una fila por contacto AP neuronal que realmente se grabó:

- `binary_channel`: índice de canal en el `.ap.bin`, empezando en **0**. No incluyas el canal SY si existe.
- `x_um`, `y_um`: coordenadas físicas de cada contacto en micrómetros, según el modelo/cableado real.
- `shank`: identificador numérico del shank.

Usá el mapa de canales correspondiente a la sonda y a la configuración de adquisición de esa sesión (por ejemplo, documentación del laboratorio / SpikeGLX / mapa de sonda). No copies un mapa NP1 ni asumas orden de canales.

Luego:

```powershell
python src/02_make_probe.py
```

Se genera `config/probe_np2.json`. El script valida índices únicos, dimensiones y canales dentro del binario. **No genera coordenadas ficticias.**

## 5. Validar y ejecutar un segmento corto

En `config/settings.json` dejá `test_duration_seconds` en 20 para la primera prueba. Después:

```powershell
python src/03_validate.py
python src/04_run_kilosort.py
```

El script llama a `kilosort.run_kilosort` con `device=torch.device('cpu')`, `data_dtype='int16'`, `n_chan_bin` y `fs` leídos del `.meta`, y `tmin/tmax` para limitar el segmento. Resultados en `results/<nombre_grabacion>/`.

Para procesar toda la sesión, cambiá `test_duration_seconds` a `null` (sin comillas) y volvé a ejecutar. **No hagas esto en el T590 sin antes comprobar que el segmento de prueba terminó y que el tiempo/espacio son razonables.** El sorting por CPU de una sesión larga puede tardar muchísimo.

## 6. Qué produce Kilosort

Kilosort guarda sus resultados y archivos compatibles con Phy en la carpeta de resultados (por ejemplo `spike_times.npy`, `spike_clusters.npy`, `templates.npy`, `cluster_*.tsv`, `params.py`, etc.; los nombres exactos dependen de la versión). Kilosort realiza internamente filtrado/preprocesamiento, detección, corrección de drift y clustering. No filtres previamente el `.bin` para este flujo.

## Notas sobre el servidor

- El `.bin` se lee desde `Z:`; los resultados van a `results/` en el proyecto local.
- No abras ni modifiques el binario mientras corre Kilosort.
- Si la conexión de red es inestable o la lectura muy lenta, la alternativa es copiar **solo la grabación de prueba** a un SSD con espacio suficiente; no hace falta duplicar todas las sesiones.
- Este proyecto es un punto de partida de spike sorting, no una reproducción exacta del pipeline IBL ni un reemplazo del control de calidad manual en Phy.
