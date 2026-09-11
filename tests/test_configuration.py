from pathlib import Path

from los80.configuration import AppConfig, load_config


def test_load_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "input_dir: /tmp/input\n"
        "output_dir: /tmp/output\n"
        "archive_dir: /tmp/archive\n"
        "max_retries: 3\n"
        "stages:\n"
        "  - scan\n"
        "  - subtitles\n",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.input_dir == "/tmp/input"
    assert config.output_dir == "/tmp/output"
    assert config.archive_dir == "/tmp/archive"
    assert config.max_retries == 3
    assert "scan" in config.stages
    assert "subtitles" in config.stages


def test_default_config_values() -> None:
    config = AppConfig()

    assert config.input_dir == "./input"
    assert config.output_dir == "./output"
    assert config.archive_dir == "./archive"
    assert config.max_retries == 2
    assert config.include_subtitles is True


def test_load_nested_realesrgan_override(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("ai_upscaling:\n  backend_path: /opt/realesrgan\n", encoding="utf-8")

    assert load_config(config_file).realesrgan_backend_path == "/opt/realesrgan"


def test_load_nested_translation_settings(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "translation:\n"
        "  model: local/nllb\n"
        "  device: cpu\n"
        "  batch_size: 4\n",
        encoding="utf-8",
    )

    config = load_config(config_file)
    assert config.translation_model == "local/nllb"
    assert config.translation_device == "cpu"
    assert config.translation_batch_size == 4
