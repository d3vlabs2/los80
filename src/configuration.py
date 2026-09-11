"""Configuration loading and validation for LOS80.

The :class:`Config` class reads the project's YAML configuration and reports
configuration problems with messages intended to be useful to end users.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


class ConfigError(ValueError):
    """Raised when an LOS80 configuration file is missing or invalid."""


class Config(Mapping[str, Any]):
    """Load, validate, and expose an LOS80 YAML configuration.

    Args:
        path: YAML file to load. When omitted, the repository's
            ``config/config.yaml`` file is used.

    Raises:
        ConfigError: If the file cannot be read, contains invalid YAML, or
            does not satisfy the required LOS80 configuration schema.
    """

    _REQUIRED_SECTIONS = (
        "general",
        "google_drive",
        "database",
        "whisper",
        "translation",
        "ai_upscaling",
        "ffmpeg",
        "output",
        "retry",
        "logging",
        "dashboard",
        "reports",
    )
    _DRIVE_FOLDERS = ("input", "output", "archive", "logs", "reports", "temp")

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
        self._data = self._load()
        self._validate()

    def __getitem__(self, key: str) -> Any:
        """Return a top-level configuration section or value."""
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over top-level configuration keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of top-level configuration keys."""
        return len(self._data)

    def get_value(self, dotted_path: str) -> Any:
        """Return a value addressed with a dotted path such as ``ffmpeg.crf``.

        Raises:
            ConfigError: If any part of the requested path does not exist.
        """
        value: Any = self._data
        for part in dotted_path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ConfigError(f"Unknown configuration option: '{dotted_path}'.")
            value = value[part]
        return value

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the top-level configuration mapping."""
        return dict(self._data)

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            raise ConfigError(
                f"Configuration file not found: '{self.path}'. "
                "Create it or pass a valid path to Config()."
            )

        try:
            with self.path.open("r", encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file)
        except OSError as exc:
            raise ConfigError(
                f"Could not read configuration file '{self.path}': {exc}"
            ) from exc
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Invalid YAML in configuration file '{self.path}': {exc}"
            ) from exc

        if not isinstance(loaded, dict):
            raise ConfigError("The configuration root must be a YAML mapping.")
        return loaded

    def _validate(self) -> None:
        errors: list[str] = []

        for section_name in self._REQUIRED_SECTIONS:
            section = self._data.get(section_name)
            if section is None:
                errors.append(f"missing required section '{section_name}'")
            elif not isinstance(section, dict):
                errors.append(f"'{section_name}' must be a mapping")

        if errors:
            self._raise_validation_errors(errors)

        self._require_strings(
            errors,
            "general",
            ("project_name", "execution_environment"),
        )
        self._choice(errors, "general.execution_environment", {"local", "colab"})
        self._require_booleans(
            errors, "general", ("recursive_scan", "duplicate_detection", "resume")
        )

        self._require_strings(errors, "google_drive", self._DRIVE_FOLDERS)
        self._require_strings(errors, "database", ("path",))
        self._positive_number(errors, "database.timeout_seconds", integer=True)

        self._require_strings(
            errors, "whisper", ("model", "device", "compute_type", "language")
        )
        self._choice(errors, "whisper.device", {"auto", "cpu", "cuda"})
        self._choice(
            errors, "whisper.compute_type", {"auto", "int8", "float16", "float32"}
        )
        self._positive_number(errors, "whisper.beam_size", integer=True)

        self._require_booleans(errors, "translation", ("enabled",))
        self._require_strings(
            errors, "translation", ("source_language", "target_language")
        )

        self._require_booleans(errors, "ai_upscaling", ("enabled",))
        for option in ("target_width", "target_height", "batch_size"):
            self._positive_number(errors, f"ai_upscaling.{option}", integer=True)

        self._require_strings(
            errors,
            "ffmpeg",
            ("binary", "video_codec", "preset", "audio_codec", "audio_bitrate"),
        )
        self._integer_range(errors, "ffmpeg.crf", 0, 51)

        self._require_strings(errors, "output", ("container", "filename_suffix"))
        self._require_booleans(
            errors,
            "output",
            ("preserve_directory_structure", "validate", "overwrite"),
        )

        self._positive_number(errors, "retry.max_attempts", integer=True)
        self._positive_number(errors, "retry.initial_delay_seconds", allow_zero=True)
        self._positive_number(errors, "retry.backoff_multiplier")

        self._require_strings(errors, "logging", ("level", "filename"))
        self._choice(
            errors,
            "logging.level",
            {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        )
        self._require_booleans(errors, "logging", ("console", "file"))

        self._require_booleans(errors, "dashboard", ("enabled",))
        self._require_strings(errors, "dashboard", ("host",))
        self._integer_range(errors, "dashboard.port", 1, 65535)
        self._positive_number(errors, "dashboard.refresh_interval_seconds")

        self._require_booleans(errors, "reports", ("html", "json", "csv"))
        self._require_strings(errors, "reports", ("filename_prefix",))

        if errors:
            self._raise_validation_errors(errors)

    def _require_strings(
        self, errors: list[str], section_name: str, option_names: tuple[str, ...]
    ) -> None:
        section = self._data[section_name]
        for option_name in option_names:
            value = section.get(option_name)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"'{section_name}.{option_name}' must be a non-empty string"
                )

    def _require_booleans(
        self, errors: list[str], section_name: str, option_names: tuple[str, ...]
    ) -> None:
        section = self._data[section_name]
        for option_name in option_names:
            if not isinstance(section.get(option_name), bool):
                errors.append(f"'{section_name}.{option_name}' must be true or false")

    def _choice(self, errors: list[str], path: str, choices: set[str]) -> None:
        value = self._value_or_none(path)
        if isinstance(value, str) and value not in choices:
            allowed = ", ".join(sorted(choices))
            errors.append(f"'{path}' must be one of: {allowed}")

    def _positive_number(
        self,
        errors: list[str],
        path: str,
        *,
        integer: bool = False,
        allow_zero: bool = False,
    ) -> None:
        value = self._value_or_none(path)
        expected_type = int if integer else (int, float)
        valid_type = isinstance(value, expected_type) and not isinstance(value, bool)
        if valid_type:
            minimum_valid = value >= 0 if allow_zero else value > 0
        else:
            minimum_valid = False
        if not valid_type or not minimum_valid:
            description = "a non-negative" if allow_zero else "a positive"
            kind = "integer" if integer else "number"
            errors.append(f"'{path}' must be {description} {kind}")

    def _integer_range(
        self, errors: list[str], path: str, minimum: int, maximum: int
    ) -> None:
        value = self._value_or_none(path)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not minimum <= value <= maximum
        ):
            errors.append(f"'{path}' must be an integer from {minimum} to {maximum}")

    def _value_or_none(self, dotted_path: str) -> Any:
        value: Any = self._data
        for part in dotted_path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return None
            value = value[part]
        return value

    def _raise_validation_errors(self, errors: list[str]) -> None:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ConfigError(
            f"Invalid LOS80 configuration in '{self.path}':\n{details}"
        )
