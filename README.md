# LOS80

LOS80 is a production-style Python pipeline for batch video restoration. It scans a Google Drive input folder, generates Spanish and English subtitles, upscales videos, encodes them to H.265, validates the outputs, uploads them, and archives the source files.

## Features

- Recursive scanning of input folders
- SQLite-backed job tracking and resume support
- Automatic retries and stage-level status tracking
- Dashboard-ready summaries and HTML/CSV reports
- Modular components for subtitles, translation, upscaling, encoding, validation, and Google Drive integration
- CLI entry point and a Colab notebook for one-click execution

## Project layout

- config/config.yaml: canonical YAML configuration
- los80/configuration.py: YAML configuration loading
- los80/scanner.py: recursive video discovery
- los80/database.py: SQLite job persistence and stage tracking
- los80/subtitles.py: Spanish subtitle generation
- los80/translator.py: subtitle translation flow
- los80/upscaler.py: AI upscaling abstraction
- los80/encoder.py: H.265 encode step
- los80/validator.py: output validation
- los80/drive.py: upload integration
- los80/reports.py: HTML/CSV/JSON reports
- los80/dashboard.py: queue/progress summary
- los80/run.py: CLI entry point

## Quick start

1. Install LOS80 and its Faster-Whisper, PyTorch, Transformers, and
   SentencePiece dependencies:

```bash
python3 -m pip install --user --break-system-packages -e .
```

Subtitle translation runs fully offline after the first model download. LOS80
automatically downloads `facebook/nllb-200-distilled-600M` and caches it in
`~/.cache/los80/models` locally or `/content/.cache/los80/models` in Colab.
Set `translation_device: auto` to use CUDA when available and CPU otherwise;
`translation_batch_size` controls translation throughput and memory usage.
Run `los80 doctor` to download and verify both the translation model and the
Real-ESRGAN runtime before processing.

2. Edit `config/config.yaml` and choose a Whisper model such as `tiny`, `base`, `small`, `medium`, or `large-v3`:

```yaml
whisper_model: base
whisper_device: cpu
whisper_compute_type: int8
```

3. Real-ESRGAN is downloaded automatically on first use and cached under
   `~/.cache/los80/realesrgan` (or `/content/.cache/los80/realesrgan` in Colab).
   Run `los80 doctor` to install it early and verify the runtime.

   Video upscaling is frame-based: LOS80 extracts timestamped PNG frames,
   processes resumable batches across available GPUs, and remuxes the result
   with the source audio, subtitles, chapters, metadata, aspect ratio, and
   color tags. Temporary frames are removed after a successful remux and kept
   after an interruption so the next run can resume.

4. Configure the upscaling block in `config/config.yaml`:

```yaml
upscaler_model: RealESRGAN_x4plus
realesrgan_backend_path: null  # optional executable or extracted runtime directory
target_width: 3840
target_height: 2160
tile_size: 0
tile_padding: 10
face_enhance: false
denoise: false
sharpen: false
skip_if_target_reached: true
```

For GPU acceleration, install a CUDA-compatible build of the backend and set `whisper_device` or the upscaler runtime to `cuda` where supported.

5. Place videos under the configured input folder.
6. Run:

```bash
los80
```

The CLI uses `config/config.yaml` by default. An explicit path remains
available for compatibility with existing automation:

```bash
los80 --config config/config.yaml
```

## Google Drive integration

LOS80 now includes a production-style Google Drive synchronization layer that uses OAuth-based authentication and supports both local execution and Google Colab.

### 1. Create a Google Cloud project

1. Open the Google Cloud Console.
2. Create a new project or select an existing one.
3. Enable the Google Drive API.
4. Create OAuth credentials for a desktop app (local) or a web application (Colab).
5. Download the client credentials JSON and save it locally.

### 2. Configure local execution

Install the Google client library:

```bash
python3 -m pip install --user --break-system-packages google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

Add the following values to `config/config.yaml`:

```yaml
drive_enabled: true
drive_parent_id: YOUR_SHARED_FOLDER_ID
# optional local credentials file
# drive_credentials_path: ./credentials.json
drive_token_path: ./drive_token.json
drive_client_id: YOUR_CLIENT_ID
drive_client_secret: YOUR_CLIENT_SECRET
drive_use_colab: false
```

Run the pipeline once to authenticate interactively. The client will create the required Drive folders automatically:

- INPUT
- OUTPUT
- ARCHIVE
- LOGS
- REPORTS
- TEMP

### 3. Configure Google Colab

In Colab, install the client libraries and set the OAuth flow to use the Colab helper:

```python
!pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

Then enable the Drive integration in `config/config.yaml`:

```yaml
drive_enabled: true
drive_use_colab: true
```

The Colab flow will authenticate via the Google Colab auth helper and use the same Drive folder structure.

### 4. Transfer behavior

The Drive integration now supports:

- resumable uploads
- download caching to avoid re-downloading files already present locally
- duplicate detection and verification using size and checksum where possible
- automatic retries according to the configured retry policy
- clear errors for quota or access issues
- database updates after each transfer step
- archive moves that only happen after upload verification succeeds

### 5. Safety guarantees

The pipeline never overwrites user data during transfer flows. Existing local files are reused as cache, uploads are verified before success is recorded, and archived sources are only moved after the required Drive uploads are confirmed.

## Colab

Open Run_All.ipynb in Google Colab and run all cells.
# los80

### One-frame CUDA smoke test (Colab Tesla T4)

Select a GPU runtime and use the updated checkout. The default upscaler now checks
`torch.cuda.is_available()`: CUDA selects `TorchRealESRGANBackend`, using the official
`RealESRGANer` and `RealESRGAN_x4plus` weights with FP16. Non-CUDA systems retain
NCNN/Vulkan; this fallback requires a working Vulkan runtime, and is not a PyTorch
CPU inference path. CUDA setup/inference failures are reported without switching to
NCNN. An explicit injected backend class remains supported.

Run these notebook cells from the project directory (adjust only the Pilot path
if your Drive layout differs):

```python
%cd /content/los80
%pip install -e '.[cuda]'
```

```bash
!mkdir -p /content/los80_smoke
!ffmpeg -hide_banner -loglevel error -y -ss 00:01:00 -i "/content/drive/MyDrive/LOS80/INPUT/Pilot.mp4" -map 0:v:0 -frames:v 1 /content/los80_smoke/input.png
!los80 smoke --require-cuda
```

The smoke command defaults to `/content/los80_smoke/input.png` and
`/content/los80_smoke/output.png`. It loads the configured model and scale and uses
256-pixel tiles (override with `--tile-size 128` if necessary). It does not open the
job database, scan Drive, translate subtitles, or process a video. Model weights
are downloaded on the first invocation to `/content/.cache/los80/realesrgan/torch`.
The reported elapsed time includes model setup/download and frame processing.

Expected output includes `backend selected: TorchRealESRGANBackend`,
`CUDA availability: True`, `GPU name: Tesla T4` (sometimes reported as `Tesla T4` or
`NVIDIA T4`), `model used: RealESRGAN_x4plus`, `precision: FP16`, input/output
resolution, scale factor, elapsed seconds, and the output path. At scale 4 an
input of 720x480 produces 2880x1920. Display the actual result before running a
full video:

```python
from IPython.display import display, Image
display(Image(filename='/content/los80_smoke/output.png'))
```

CUDA video processing uses the existing extraction, timestamp-based frame names,
FFmpeg reassembly, track/metadata mapping, and database stages. Real FFmpeg
regression fixtures cover 25 and 30000/1001 FPS, frame count, audio, subtitles,
chapters, title metadata, duration, and sample aspect ratio. They exposed and
fixed existing extraction time-base, concat frame-rate, and setsar syntax bugs. Completed CUDA
frames are decoded and checked for the expected size when resuming; failed writes
never publish a partial final frame. Remux output is also published atomically
so a failed FFmpeg run cannot be mistaken for a completed intermediate. Inference errors (including tiled OOMs) stop
the stage while keeping completed frames for retry. Same-size changes to model
settings still require clearing the intermediate frames before a new run.

Local tests mock CUDA and the model; passing tests do not establish actual T4
inference success. The one-frame Colab run is required before the full Pilot.
