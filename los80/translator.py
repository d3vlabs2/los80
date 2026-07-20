from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional


class TranslationError(RuntimeError):
    pass


class TranslationService:
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("los80.translator")

    def translate_subtitles(self, source_path: str | Path, output_path: str | Path) -> Path:
        input_path = Path(source_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        self.logger.info("Starting subtitle translation from %s", input_path.name)
        content = input_path.read_text(encoding="utf-8")
        if not content.strip():
            raise TranslationError(f"Subtitle source is empty: {input_path}")

        try:
            from deep_translator import GoogleTranslator  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            self.logger.error("Translation dependency is unavailable: %s", exc)
            raise TranslationError("Optional translation dependency is not available. Install deep-translator or disable translation.") from exc

        translator = GoogleTranslator(source="es", target="en")
        translated_lines: list[str] = []
        for line in content.splitlines():
            if not line.strip() or re.match(r"^\d+$", line) or "-->" in line:
                translated_lines.append(line)
                continue
            translated_lines.append(translator.translate(line))

        translated_text = "\n".join(translated_lines).strip() + "\n"
        output_file.write_text(translated_text, encoding="utf-8")
        self.logger.info("Completed subtitle translation for %s", input_path.name)
        return output_file
