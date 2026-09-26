"""One neural pass, pixel-accurate blends, atomic output and CLI isolation."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from PIL import Image

from los80 import smoke
from los80.torch_realesrgan import restoration_variant
from los80.upscaler import TorchRealESRGANBackend, UpscalingError


@pytest.fixture
def pixels():
    return pytest.importorskip('numpy'), pytest.importorskip('cv2')


@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('scale', [2, 3, 4])
@pytest.mark.parametrize('strength', [0, 0.25, 0.5, 0.75, 1])
def test_blend_values_dimensions_and_bit_depth(pixels, dtype, scale, strength):
    np, cv2 = pixels
    original = np.full((5, 7, 3), 20, dtype=dtype)
    neural = np.full((20, 28, 3), 100, dtype=dtype)
    result = restoration_variant(original, neural, scale, strength)
    assert result.shape == (5*scale, 7*scale, 3)
    assert result.dtype == original.dtype
    np.testing.assert_array_equal(result, np.full(result.shape, 20 + strength*80, dtype=dtype))


def test_neural_result_is_downsampled_with_lanczos(pixels):
    np, cv2 = pixels
    original = np.zeros((5, 7, 3), dtype=np.uint8)
    neural = np.random.default_rng(9).integers(0, 256, (20, 28, 3), dtype=np.uint8)
    for scale in (2, 3):
        expected = cv2.resize(neural, (7*scale, 5*scale), interpolation=cv2.INTER_LANCZOS4)
        np.testing.assert_array_equal(restoration_variant(original, neural, scale), expected)
    assert restoration_variant(original, neural, 4) is neural


@pytest.mark.parametrize('channels', [1, 4])
def test_blend_preserves_grayscale_and_alpha(pixels, channels):
    np, _ = pixels
    shape = (3, 4) if channels == 1 else (3, 4, 4)
    original = np.full(shape, 20, dtype=np.uint16)
    neural = np.repeat(np.repeat(original + 40, 4, axis=0), 4, axis=1)
    result = restoration_variant(original, neural, 2, .25)
    assert result.shape == ((6, 8) if channels == 1 else (6, 8, 4))
    assert (result == 30).all()


@pytest.fixture
def comparison_backend(pixels, monkeypatch):
    np, _ = pixels
    backend = TorchRealESRGANBackend()
    backend.half = True
    calls = []
    preparations = []
    def enhance(frame, outscale):
        calls.append((frame.shape, outscale))
        return np.full((frame.shape[0]*4, frame.shape[1]*4, 3), 100, dtype=np.uint8), 'RGB'
    backend.engine = SimpleNamespace(enhance=enhance)
    def prepare(config, require_cuda):
        preparations.append((config, require_cuda))
        return backend, None, 'RealESRGAN_x4plus'
    monkeypatch.setattr(smoke, '_prepare_backend', prepare)
    return calls, preparations


def test_comparison_one_inference_all_variants_and_labels(tmp_path, pixels, comparison_backend, capsys):
    np, cv2 = pixels
    calls, preparations = comparison_backend
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 6), (20, 20, 20)).save(source)
    output_dir = tmp_path / 'comparison'
    result = smoke.smoke_compare(source, output_dir, {}, require_cuda=True, contact_sheet=True)
    assert calls == [((6, 8, 3), 4.0)]
    assert len(preparations) == 1 and preparations[0][1] is True
    assert preparations[0][0]['scale'] == 4
    assert len(result['variants']) == 12
    for item in result['variants']:
        scale, strength = item['scale'], item['strength']
        assert item['path'].name == f'RealESRGAN_x4plus_scale{scale}_strength{strength:.2f}.png'
        image = cv2.imread(str(item['path']))
        assert image.shape == (6*scale, 8*scale, 3)
        assert (image == 20 + strength*80).all()
    with Image.open(result['contact_sheet']) as sheet:
        assert sheet.size == (2048, 1792)  # original plus 12 labeled variants
        assert sheet.getpixel((0, 64)) == (20, 20, 20)
    assert len(list(output_dir.iterdir())) == 13
    assert 'one native 4x pass' in capsys.readouterr().out


def test_contact_crop_keeps_full_input_for_inference(tmp_path, pixels, comparison_backend):
    calls, _ = comparison_backend
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 6)).save(source)
    result = smoke.smoke_compare(source, tmp_path / 'comparison', {}, scales=[2], strengths=[.5],
                                 contact_sheet=True, crop=(1, 2, 3, 3))
    assert calls == [((6, 8, 3), 4.0)]
    with Image.open(result['contact_sheet']) as sheet:
        assert sheet.size == (2048, 576)
    assert result['contact_sheet'].name == 'contact-sheet-crop.png'
    with Image.open(result['variants'][0]['path']) as image:
        assert image.size == (16, 12)


def test_single_smoke_reduced_strength_uses_same_native_path(tmp_path, pixels, comparison_backend):
    _, cv2 = pixels
    calls, _ = comparison_backend
    source = tmp_path / 'input.png'; output = tmp_path / 'output.png'
    Image.new('RGB', (8, 6), (20, 20, 20)).save(source)
    result = smoke.smoke_frame(source, output, {'scale': 3, 'restoration_strength': .25})
    assert calls == [((6, 8, 3), 4.0)]
    assert (cv2.imread(str(output)) == 40).all()
    assert result['output resolution'] == '24x18'
    assert result['restoration strength (output blend)'] == .25
    assert not list(tmp_path.glob('.smoke-*'))


@pytest.mark.parametrize('options', [
    {'scales': []}, {'scales': [5]}, {'strengths': []}, {'strengths': [-.1]},
    {'strengths': [1.1]}, {'strengths': [float('nan')]},
    {'contact_sheet': True, 'crop': (0, 0, 100, 100)}, {'crop': (0, 0, 1, 1)},
])
def test_bad_comparison_settings_fail_before_inference(tmp_path, pixels, monkeypatch, options):
    source = tmp_path / 'input.png'; Image.new('RGB', (8, 6)).save(source)
    monkeypatch.setattr(smoke, '_prepare_backend', lambda *a: pytest.fail('model prepared'))
    with pytest.raises(ValueError):
        smoke.smoke_compare(source, tmp_path / 'comparison', {}, **options)


def test_atomic_variant_write_keeps_existing_file(tmp_path, pixels, monkeypatch):
    np, cv2 = pixels
    output = tmp_path / 'existing.png'; output.write_bytes(b'existing output')
    def fail(path, image):
        Path(path).write_bytes(b'partial')
        return False
    monkeypatch.setattr(cv2, 'imwrite', fail)
    with pytest.raises(UpscalingError, match='Cannot write'):
        smoke._atomic_png(output, np.zeros((2, 2, 3), dtype=np.uint8))
    assert output.read_bytes() == b'existing output'
    assert list(tmp_path.iterdir()) == [output]


def test_inference_failure_publishes_no_variants(tmp_path, pixels, comparison_backend, monkeypatch):
    source = tmp_path / 'input.png'; Image.new('RGB', (8, 6)).save(source)
    def fail(*args):
        raise UpscalingError('CUDA failure')
    monkeypatch.setattr(smoke, '_upscale_frame', fail)
    output = tmp_path / 'comparison'
    with pytest.raises(UpscalingError, match='CUDA failure'):
        smoke.smoke_compare(source, output, {})
    assert not list(output.iterdir())


def test_comparison_cli_defaults_and_pipeline_isolation(monkeypatch):
    from los80 import run
    calls = []
    monkeypatch.setattr(smoke, 'smoke_compare', lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(run, 'JobDatabase', lambda *a: pytest.fail('database opened'))
    monkeypatch.setattr(run, 'Pipeline', lambda *a: pytest.fail('pipeline started'))
    monkeypatch.setattr(sys, 'argv', ['los80', 'smoke-compare', '--require-cuda', '--contact-sheet'])
    run.main()
    args, kwargs = calls[0]
    assert args[:2] == (Path('/content/los80_smoke/input.png'), Path('/content/los80_smoke/comparison'))
    assert kwargs['scales'] == [2, 3, 4]
    assert kwargs['strengths'] == [.25, .5, .75, 1]
    assert kwargs['require_cuda'] is True and kwargs['contact_sheet'] is True


def test_comparison_with_real_official_engine(tmp_path, pixels, monkeypatch):
    import os
    model_path = os.environ.get('LOS80_TEST_MODEL')
    if not model_path:
        pytest.skip('set LOS80_TEST_MODEL to a cached official x4plus checkpoint')
    pytest.importorskip('torch')
    from los80.torch_realesrgan import _load_model, SpandrelEngine
    np, cv2 = pixels
    backend = TorchRealESRGANBackend()
    backend.half = False
    backend.engine = SpandrelEngine(_load_model(Path(model_path)).eval(), tile=6, tile_pad=10)
    calls = []
    enhance = backend.engine.enhance
    def counted(image, outscale):
        calls.append(outscale)
        return enhance(image, outscale)
    monkeypatch.setattr(backend.engine, 'enhance', counted)
    monkeypatch.setattr(smoke, '_prepare_backend', lambda *a: (backend, None, 'RealESRGAN_x4plus'))
    source = tmp_path / 'input.png'
    image = np.random.default_rng(23).integers(0, 256, (8, 10, 3), dtype=np.uint8)
    cv2.imwrite(str(source), image)
    result = smoke.smoke_compare(source, tmp_path / 'comparison', {}, contact_sheet=True)
    assert calls == [4.0]
    native = cv2.imread(str(tmp_path / 'comparison' / 'RealESRGAN_x4plus_scale4_strength1.00.png'))
    for item in result['variants']:
        expected = restoration_variant(image, native, item['scale'], item['strength'])
        np.testing.assert_array_equal(cv2.imread(str(item['path'])), expected)
    assert result['contact_sheet'].is_file()
