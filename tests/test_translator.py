from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

import pytest

from los80.translator import TranslationError, TranslationService


class FakeTensor:
    def __init__(self, value, moves=None):
        self.value = value
        self.moves = moves if moves is not None else []

    def to(self, device):
        self.moves.append(device)
        return self


class FakeTokenizer:
    translations = {"Hola": "Hello", "Adiós": "Goodbye", "Gracias": "Thank you"}

    def __init__(self):
        self.batches = []

    def __call__(self, texts, **kwargs):
        self.batches.append(list(texts))
        return {"input_ids": FakeTensor(list(texts))}

    def convert_tokens_to_ids(self, token):
        assert token == "eng_Latn"
        return 42

    def batch_decode(self, output, **kwargs):
        return output


class FakeModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate(self, input_ids, **kwargs):
        return [self.tokenizer.translations[text] for text in input_ids.value]


class FakeTorch:
    inference_mode = staticmethod(contextlib.nullcontext)


def fake_service(tmp_path: Path, batch_size: int = 8):
    tokenizer = FakeTokenizer()
    loader = lambda *args: (tokenizer, FakeModel(tokenizer), FakeTorch, "cpu")
    return TranslationService(batch_size=batch_size, cache_dir=tmp_path / "models", runtime_loader=loader), tokenizer


def write_srt(path: Path):
    path.write_text(
        "1\r\n00:00:00,000 --> 00:00:01,000\r\n<i>Hola</i>\r\n\r\n"
        "2\r\n00:00:01,000 --> 00:00:02,000\r\nAdiós\r\n",
        encoding="utf-8", newline="",
    )


def test_successful_translation_preserves_srt_formatting(tmp_path: Path) -> None:
    service, _ = fake_service(tmp_path)
    source = tmp_path / "clip.es.srt"
    output = tmp_path / "clip.en.srt"
    write_srt(source)

    assert service.translate_subtitles(source, output) == output
    assert output.read_bytes() == (
        b"1\r\n00:00:00,000 --> 00:00:01,000\r\n<i>Hello</i>\r\n\r\n"
        b"2\r\n00:00:01,000 --> 00:00:02,000\r\nGoodbye\r\n"
    )


def test_translation_resume_uses_existing_output(tmp_path: Path) -> None:
    service, tokenizer = fake_service(tmp_path)
    output = tmp_path / "clip.en.srt"
    output.write_text("existing\n", encoding="utf-8")

    assert service.translate_subtitles(tmp_path / "missing.es.srt", output) == output
    assert tokenizer.batches == []


def test_translation_batches_text_lines(tmp_path: Path) -> None:
    service, tokenizer = fake_service(tmp_path, batch_size=2)
    source = tmp_path / "clip.es.srt"
    source.write_text("Hola\nAdiós\nGracias\n", encoding="utf-8")

    service.translate_subtitles(source, tmp_path / "clip.en.srt")

    assert tokenizer.batches == [["Hola", "Adiós"], ["Gracias"]]


def _install_fake_ml_modules(monkeypatch, cuda_available: bool):
    moves = []

    class TorchModule:
        cuda = types.SimpleNamespace(is_available=lambda: cuda_available)

    class LoadedModel:
        def to(self, device):
            moves.append(device)
            return self

        def eval(self):
            return self

    tokenizer_factory = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    model_factory = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: LoadedModel())
    monkeypatch.setitem(sys.modules, "torch", TorchModule)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=tokenizer_factory, AutoModelForSeq2SeqLM=model_factory,
    ))
    return moves


def test_cpu_fallback_when_cuda_unavailable(tmp_path: Path, monkeypatch) -> None:
    moves = _install_fake_ml_modules(monkeypatch, False)
    runtime = TranslationService._load_huggingface_runtime("model", "cuda", tmp_path)
    assert runtime[3] == "cpu"
    assert moves == ["cpu"]


def test_gpu_execution_when_cuda_available(tmp_path: Path, monkeypatch) -> None:
    moves = _install_fake_ml_modules(monkeypatch, True)
    runtime = TranslationService._load_huggingface_runtime("model", "auto", tmp_path)
    assert runtime[3] == "cuda"
    assert moves == ["cuda"]


def test_model_download_uses_configured_cache(tmp_path: Path) -> None:
    calls = []

    def loader(model, device, cache):
        calls.append((model, device, cache))
        tokenizer = FakeTokenizer()
        return tokenizer, FakeModel(tokenizer), FakeTorch, "cpu"

    info = TranslationService(cache_dir=tmp_path, runtime_loader=loader).prepare()
    assert calls == [("facebook/nllb-200-distilled-600M", "auto", tmp_path)]
    assert info["cache_dir"] == tmp_path


def test_translation_failure_does_not_create_output(tmp_path: Path) -> None:
    def failing_loader(*args):
        raise OSError("download failed")

    source = tmp_path / "clip.es.srt"
    output = tmp_path / "clip.en.srt"
    source.write_text("Hola\n", encoding="utf-8")
    service = TranslationService(cache_dir=tmp_path / "models", runtime_loader=failing_loader)

    with pytest.raises(TranslationError, match="download failed"):
        service.translate_subtitles(source, output)
    assert not output.exists()
