from pathlib import Path

import pytest

from los80.configuration import AppConfig
from los80.database import JobDatabase
from los80.pipeline import Pipeline
from los80.realesrgan_runtime import RuntimeInfo
from los80.upscaler import AIUpscaler, RealESRGANBackend, UpscalingError


class FakeBackend:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def upscale(self, input_path: Path, output_path: Path, config: dict[str, object]) -> Path:
        output_path.write_bytes(b"upscaled")
        return output_path


def test_upscaler_skips_existing_intermediate(tmp_path: Path) -> None:
    input_path = tmp_path / "clip.mp4"
    input_path.write_bytes(b"video")
    output_path = tmp_path / "clip.upscaled.mp4"
    output_path.write_bytes(b"already")

    upscaler = AIUpscaler(backend_cls=FakeBackend)
    result = upscaler.upscale(input_path, output_path, {"skip_if_target_reached": False})

    assert result == output_path
    assert output_path.read_bytes() == b"already"


def test_pipeline_resumes_upscaling_from_existing_intermediate(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    video_path = input_dir / "clip.mp4"
    video_path.write_bytes(b"video")

    working_dir = tmp_path / "working"
    working_dir.mkdir(parents=True)
    upscaled_path = working_dir / "clip.mp4" / "clip.upscaled.mp4"
    upscaled_path.parent.mkdir(parents=True)
    upscaled_path.write_bytes(b"upscaled")

    config = AppConfig(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "output"),
        archive_dir=str(tmp_path / "archive"),
        working_dir=str(working_dir),
        reports_dir=str(tmp_path / "reports"),
        database_path=str(tmp_path / "jobs.sqlite"),
        max_retries=1,
        include_subtitles=False,
        include_translation=False,
    )
    database = JobDatabase(config.database_path)
    database.add_job("clip.mp4", str(video_path))
    database.mark_stage("clip.mp4", "scan", "completed", "discovered")
    database.mark_stage("clip.mp4", "upscaling", "completed", "existing")

    pipeline = Pipeline(config, database)

    class FakeEncoderBackend:
        def detect_hardware_encoder(self, preference: str = "auto") -> str:
            return "libx265"

    monkeypatch.setattr(
        pipeline.upscaler,
        "upscale",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed upscaling stage should be skipped")
        ),
    )

    def encode(input_path: Path, output_path: Path, options: dict) -> Path:
        output_path.write_bytes(b"encoded")
        return output_path

    monkeypatch.setattr(pipeline.encoder, "encode", encode)
    pipeline.encoder.backend_cls = FakeEncoderBackend
    monkeypatch.setattr(pipeline.validator, "validate", lambda *args, **kwargs: None)
    pipeline.run()

    assert database.get_stages("clip.mp4")["upscaling"] == "completed"


def _runtime(tmp_path: Path) -> RuntimeInfo:
    executable = tmp_path / "realesrgan-ncnn-vulkan"
    executable.write_text("binary", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    for suffix in (".bin", ".param"):
        (models / f"realesrgan-x4plus{suffix}").write_text("model", encoding="utf-8")
    return RuntimeInfo(executable, models, "test", tmp_path)


def test_frame_pipeline_batches_remuxes_tracks_and_cleans_up(tmp_path: Path, monkeypatch, caplog) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr("los80.upscaler.RealESRGANRuntime.ensure", lambda self: runtime)
    monkeypatch.setattr("los80.upscaler.shutil.which", lambda name: f"/usr/bin/{name}")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "ffprobe":
            return type("Result", (), {"returncode": 0, "stderr": "", "stdout":
                '{"streams":[{"time_base":"1/1000","avg_frame_rate":"25/1",'
                '"start_time":"1.5","sample_aspect_ratio":"4:3","pix_fmt":"yuv420p",'
                '"color_primaries":"bt709","color_transfer":"bt709","color_space":"bt709"}]}'})()
        if command[0] == "ffmpeg" and "-frame_pts" in command:
            destination = Path(command[-1]).parent
            for pts in (1500, 1540, 1580):
                (destination / f"frame-{pts:020d}.png").write_bytes(b"frame")
        elif command[0] == str(runtime.executable):
            source_dir = Path(command[command.index("-i") + 1])
            output_dir = Path(command[command.index("-o") + 1])
            for frame in source_dir.glob("*.png"):
                (output_dir / frame.name).write_bytes(b"upscaled")
        elif command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"video")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("los80.upscaler.subprocess.run", run)
    output = tmp_path / "clip.upscaled.mp4"
    config = {
        "batch_size": 2, "scale": 4, "tile_size": 128, "output_format": "png",
        "verbose": True, "tile_padding": 10, "denoise": True, "sharpen": True,
    }
    with caplog.at_level("WARNING"):
        RealESRGANBackend().upscale(tmp_path / "clip.mp4", output, config)

    ncnn = [command for command in commands if command[0] == str(runtime.executable)]
    remux = [command for command in commands if command[0] == "ffmpeg" and "-frame_pts" not in command][0]
    assert len(ncnn) == 2
    for command in ncnn:
        assert command[command.index("-s") + 1] == "4"
        assert command[command.index("-t") + 1] == "128"
        assert command[command.index("-f") + 1] == "png"
        assert "-v" in command
        assert "-p" not in command and "-dn" not in command
        assert "-g" not in command
    assert "tile_padding=10" in caplog.text
    assert "denoise=True" in caplog.text
    assert "sharpen=True" in caplog.text
    assert ["-map", "1:a?"] == remux[remux.index("-map", remux.index("-map") + 1):][:2]
    assert "1:s?" in remux
    assert "-map_metadata" in remux and "-map_chapters" in remux
    assert "setsar=4:3" in remux and "bt709" in remux
    assert not (tmp_path / ".clip.upscaled.mp4.realesrgan-frames").exists()


def test_ncnn_v025_command_uses_only_supported_mapped_options(tmp_path: Path) -> None:
    command = RealESRGANBackend._build_ncnn_command(
        tmp_path / "realesrgan-ncnn-vulkan", tmp_path / "input", tmp_path / "output",
        tmp_path / "models", "realesrgan-x4plus", 2,
        {"scale": 4, "tile_size": 256, "output_format": "jpg", "verbose": True,
         "tile_padding": 10, "denoise": True, "sharpen": True},
    )

    assert command == [
        str(tmp_path / "realesrgan-ncnn-vulkan"),
        "-i", str(tmp_path / "input"), "-o", str(tmp_path / "output"),
        "-m", str(tmp_path / "models"), "-n", "realesrgan-x4plus",
        "-s", "4", "-t", "256", "-g", "2", "-f", "jpg", "-v",
    ]


def test_frame_pipeline_resumes_missing_frames_and_uses_multiple_gpus(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr("los80.upscaler.RealESRGANRuntime.ensure", lambda self: runtime)
    monkeypatch.setattr("los80.upscaler.shutil.which", lambda name: f"/usr/bin/{name}")
    output = tmp_path / "clip.upscaled.mp4"
    work = tmp_path / ".clip.upscaled.mp4.realesrgan-frames"
    source = work / "source"
    upscaled = work / "upscaled"
    source.mkdir(parents=True)
    upscaled.mkdir()
    (work / ".extraction-complete").write_text("complete\n", encoding="utf-8")
    for pts in (0, 40, 80):
        (source / f"frame-{pts:020d}.png").write_bytes(b"frame")
    (upscaled / "frame-00000000000000000000.png").write_bytes(b"done")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "ffprobe":
            return type("Result", (), {"returncode": 0, "stderr": "", "stdout":
                '{"streams":[{"time_base":"1/1000","avg_frame_rate":"25/1"}]}'})()
        if command[:2] == ["nvidia-smi", "-L"]:
            return type("Result", (), {"returncode": 0, "stderr": "", "stdout": "GPU 0: A\nGPU 1: B\n"})()
        if command[0] == str(runtime.executable):
            input_dir = Path(command[command.index("-i") + 1])
            output_dir = Path(command[command.index("-o") + 1])
            for frame in input_dir.glob("*.png"):
                (output_dir / frame.name).write_bytes(b"upscaled")
        elif command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"video")
        return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()

    monkeypatch.setattr("los80.upscaler.subprocess.run", run)
    RealESRGANBackend().upscale(tmp_path / "clip.mp4", output, {"batch_size": 1, "device": "cuda"})

    assert not any(command[0] == "ffmpeg" and "-frame_pts" in command for command in commands)
    gpu_ids = {command[command.index("-g") + 1] for command in commands if command[0] == str(runtime.executable)}
    assert gpu_ids == {"0", "1"}


def test_frame_pipeline_keeps_partial_frames_after_failure(tmp_path: Path, monkeypatch) -> None:
    runtime = _runtime(tmp_path)
    monkeypatch.setattr("los80.upscaler.RealESRGANRuntime.ensure", lambda self: runtime)
    monkeypatch.setattr("los80.upscaler.shutil.which", lambda name: f"/usr/bin/{name}")

    def run(command, **kwargs):
        if command[0] == "ffprobe":
            return type("Result", (), {"returncode": 0, "stderr": "", "stdout":
                '{"streams":[{"time_base":"1/25","avg_frame_rate":"25/1"}]}'})()
        if command[0] == "ffmpeg":
            destination = Path(command[-1]).parent
            (destination / "frame-00000000000000000000.png").write_bytes(b"frame")
            return type("Result", (), {"returncode": 0, "stderr": "", "stdout": ""})()
        return type("Result", (), {"returncode": 1, "stderr": "GPU failure", "stdout": ""})()

    monkeypatch.setattr("los80.upscaler.subprocess.run", run)
    output = tmp_path / "clip.upscaled.mp4"
    with pytest.raises(UpscalingError, match="GPU failure"):
        RealESRGANBackend().upscale(tmp_path / "clip.mp4", output, {})
    assert (tmp_path / ".clip.upscaled.mp4.realesrgan-frames" / ".extraction-complete").exists()
