from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"


class TranslationError(RuntimeError):
    pass


def translation_cache_dir() -> Path:
    if Path("/content").is_dir() or "COLAB_RELEASE_TAG" in os.environ:
        return Path("/content/.cache/los80/models")
    return Path.home() / ".cache" / "los80" / "models"


class TranslationService:
    def __init__(self, logger: Optional[logging.Logger] = None,
                 model_name: str = DEFAULT_TRANSLATION_MODEL, device: str = "auto",
                 batch_size: int = 8, cache_dir: str | Path | None = None,
                 runtime_loader: Callable[..., tuple[Any, Any, Any, str]] | None = None) -> None:
        self.logger = logger or logging.getLogger("los80.translator")
        self.model_name = model_name
        self.device_preference = device
        self.batch_size = max(int(batch_size), 1)
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else translation_cache_dir()
        self.runtime_loader = runtime_loader or self._load_huggingface_runtime
        self._runtime: tuple[Any, Any, Any, str] | None = None

    def prepare(self) -> dict[str, object]:
        self._get_runtime()
        assert self._runtime is not None
        return {"model": self.model_name, "cache_dir": self.cache_dir, "device": self._runtime[3]}

    def translate_subtitles(self, source_path: str | Path, output_path: str | Path) -> Path:
        input_path = Path(source_path)
        output_file = Path(output_path)
        if output_file.exists():
            self.logger.info("Using existing English subtitles %s", output_file)
            return output_file
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        # Decode bytes directly so Python's universal-newline handling does not
        # alter CRLF subtitle formatting.
        content = input_path.read_bytes().decode("utf-8")
        if not content.strip():
            raise TranslationError(f"Subtitle source is empty: {input_path}")

        self.logger.info("Starting offline subtitle translation from %s", input_path.name)
        lines = content.splitlines(keepends=True)
        text_indexes = [index for index, line in enumerate(lines) if _is_subtitle_text(line)]
        parts = [_line_parts(lines[index]) for index in text_indexes]
        texts = [part[1] for part in parts]
        try:
            translated: list[str] = []
            for start in range(0, len(texts), self.batch_size):
                translated.extend(self._translate_batch(texts[start:start + self.batch_size]))
        except TranslationError:
            raise
        except Exception as exc:
            raise TranslationError(f"NLLB translation failed for {input_path}: {exc}") from exc
        if len(translated) != len(text_indexes):
            raise TranslationError("NLLB returned an unexpected number of translated subtitle lines")

        for index, text, (prefix, _body, suffix) in zip(text_indexes, translated, parts):
            original_body = lines[index].rstrip("\r\n")
            newline = lines[index][len(original_body):]
            lines[index] = prefix + text.strip() + suffix + newline
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes("".join(lines).encode("utf-8"))
        self.logger.info("Completed subtitle translation for %s", input_path.name)
        return output_file

    def _get_runtime(self) -> tuple[Any, Any, Any, str]:
        if self._runtime is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._runtime = self.runtime_loader(self.model_name, self.device_preference, self.cache_dir)
            except Exception as exc:
                raise TranslationError(
                    f"Could not load or download translation model {self.model_name} in {self.cache_dir}: {exc}"
                ) from exc
        return self._runtime

    def _translate_batch(self, texts: list[str]) -> list[str]:
        tokenizer, model, torch, device = self._get_runtime()
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=256,
            )
        return list(tokenizer.batch_decode(output, skip_special_tokens=True))

    @staticmethod
    def _load_huggingface_runtime(model_name: str, device_preference: str, cache_dir: Path):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise TranslationError("Offline translation requires torch, transformers, and sentencepiece") from exc
        requested = device_preference.lower()
        if requested not in {"auto", "cpu", "cuda"}:
            raise TranslationError(f"Unsupported translation device: {device_preference}")
        device = "cuda" if requested in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, src_lang="spa_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        model.to(device)
        model.eval()
        return tokenizer, model, torch, device


def _line_parts(line: str) -> tuple[str, str, str]:
    body = line.rstrip("\r\n")
    match = re.match(r"^(\s*(?:(?:<[^>]+>|\{[^}]+\})\s*)*)(.*?)(\s*(?:(?:</[^>]+>|\{[^}]+\})\s*)*)$", body)
    assert match is not None
    return match.group(1), match.group(2), match.group(3)


def _is_subtitle_text(line: str) -> bool:
    stripped = line.rstrip("\r\n").strip()
    return bool(stripped and not stripped.isdigit() and "-->" not in stripped)
