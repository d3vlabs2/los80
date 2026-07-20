import sys
import types
from pathlib import Path

from los80.translator import TranslationService


class FakeGoogleTranslator:
    def __init__(self, source: str, target: str) -> None:
        assert source == "es"
        assert target == "en"

    def translate(self, text: str) -> str:
        return {"Hola": "Hello", "Adiós": "Goodbye"}[text]


def test_translation_service_uses_optional_translator(
    tmp_path: Path, monkeypatch
) -> None:
    fake_module = types.SimpleNamespace(GoogleTranslator=FakeGoogleTranslator)
    monkeypatch.setitem(sys.modules, "deep_translator", fake_module)

    source_path = tmp_path / "clip.es.srt"
    source_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHola\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nAdiós\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "clip.en.srt"

    result = TranslationService().translate_subtitles(source_path, output_path)

    assert result == output_path
    assert "Hello" in output_path.read_text(encoding="utf-8")
    assert "Goodbye" in output_path.read_text(encoding="utf-8")
