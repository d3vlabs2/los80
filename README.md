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

1. Install Faster-Whisper:

```bash
python3 -m pip install --user --break-system-packages faster-whisper
```

2. Edit `config/config.yaml` and choose a Whisper model such as `tiny`, `base`, `small`, `medium`, or `large-v3`:

```yaml
whisper_model: base
whisper_device: cpu
whisper_compute_type: int8
```

3. Install Real-ESRGAN and its dependencies if you want the AI upscaling backend to run locally:

```bash
python3 -m pip install --user --break-system-packages realesrgan
```

4. Configure the upscaling block in `config/config.yaml`:

```yaml
upscaler_model: RealESRGAN_x4plus
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
