# LOS80 Project Status

## Completed features
- Recursive input scanning and job discovery
- SQLite-backed stage tracking with resumable execution
- Structured pipeline stage execution with timing and failure tracking
- Real dependency-aware subtitle generation via faster-whisper
- Structured error handling for translation, upscaling, encoding, validation, and Drive transfer steps
- Drive upload/download helpers with duplicate detection and verification
- Automated tests for core pipeline, database, encoder, upscaler, validator, and Drive behaviors
- Integration test scaffold for running a real small video through the pipeline when FFmpeg/FFprobe are installed

## Optional dependencies
- faster-whisper: required for subtitle generation
- Hugging Face Transformers/PyTorch: offline NLLB subtitle translation
- Real-ESRGAN: required for AI upscaling
- FFmpeg and FFprobe: required for encoding and validation
- Google Drive API client libraries: required for live Drive integration

## Remaining limitations
- Subtitle translation requires an optional translation package and network access
- AI upscaling requires a compatible Real-ESRGAN binary installed locally
- Full Drive uploads require valid OAuth credentials or access tokens
- The integration test is skipped unless FFmpeg and FFprobe are present

## Future enhancements
- Add richer progress reporting to the dashboard
- Support per-stage checkpoint artifacts for larger resumable jobs
- Add configurable translation backends and offline fallback strategies
- Expand integration coverage for real Drive and real transcription workflows
