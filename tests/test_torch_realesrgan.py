"""Dependency/cache checks plus real PyTorch tensor tests when cuda extras exist."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from los80 import torch_realesrgan as runtime
from los80.upscaler import UpscalingError


def test_cuda_dependencies_exclude_legacy_packages():
    import tomllib
    metadata = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
    extra = metadata['project']['optional-dependencies']['cuda']
    assert any(item.startswith('spandrel>=0.4.2') for item in extra)
    assert not any('basicsr' in item or 'realesrgan' in item for item in extra)


def test_model_cache_download_reuse_and_explicit_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime, 'default_cache_dir', lambda: tmp_path)
    def download(url, path, **kwargs):
        calls.append(url)
        Path(path).write_bytes(b'checkpoint')
    def load(path, **kwargs):
        assert kwargs == {'map_location': 'cpu', 'weights_only': True}
        return {}
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(hub=SimpleNamespace(download_url_to_file=download), load=load))
    target = runtime._model_path({})
    assert target == tmp_path / 'torch' / runtime.MODEL_FILENAME
    assert runtime._model_path({}) == target
    assert runtime._model_path({'model_path': '/custom/model.pth'}) == Path('/custom/model.pth')
    assert calls == [runtime.MODEL_URL]
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize('bad_checkpoint', [False, True])
def test_failed_download_never_publishes_cache(tmp_path, monkeypatch, bad_checkpoint):
    monkeypatch.setattr(runtime, 'default_cache_dir', lambda: tmp_path)
    def download(url, path, **kwargs):
        Path(path).write_bytes(b'partial')
        if not bad_checkpoint:
            raise OSError('network interrupted')
    def load(*args, **kwargs):
        raise ValueError('bad checkpoint')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(hub=SimpleNamespace(download_url_to_file=download), load=load))
    with pytest.raises((OSError, ValueError)):
        runtime._model_path({})
    assert not list((tmp_path / 'torch').iterdir())


@pytest.mark.parametrize('config', [{'model': 'unknown'}, {'scale': 0}, {'scale': float('nan')}, {'tile_size': -1}, {'tile_padding': -1}])
def test_invalid_config_fails_before_download(monkeypatch, config):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(runtime, '_model_path', lambda _: pytest.fail('download attempted'))
    with pytest.raises(UpscalingError):
        runtime.prepare_engine(config)


@pytest.fixture
def tensor_model():
    torch = pytest.importorskip('torch')
    pytest.importorskip('cv2')
    class Nearest:
        scale = 4
        device = torch.device('cpu')
        dtype = torch.float32
        def __call__(self, tensor):
            assert not torch.is_grad_enabled()
            return torch.nn.functional.interpolate(tensor, scale_factor=4, mode='nearest')
    return Nearest()


@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
@pytest.mark.parametrize('channels', [1, 3, 4])
def test_tiled_pixels_match_whole_frame_and_preserve_format(tensor_model, dtype, channels):
    import numpy as np
    shape = (7, 9) if channels == 1 else (7, 9, channels)
    image = np.random.default_rng(12).integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
    whole, mode = runtime.SpandrelEngine(tensor_model).enhance(image)
    tiled, tiled_mode = runtime.SpandrelEngine(tensor_model, tile=3, tile_pad=2).enhance(image)
    assert mode == tiled_mode == {1: 'L', 3: 'RGB', 4: 'RGBA'}[channels]
    assert whole.dtype == tiled.dtype == image.dtype
    assert whole.shape[:2] == (28, 36)
    np.testing.assert_array_equal(tiled, whole)
    np.testing.assert_array_equal(whole, np.repeat(np.repeat(image, 4, axis=0), 4, axis=1))


def test_model_receives_rgb_and_custom_scale(tensor_model):
    import numpy as np
    captured = []
    class Capture(type(tensor_model)):
        def __call__(self, tensor):
            captured.append(tensor[0, :, 0, 0].tolist())
            return super().__call__(tensor)
    image = np.full((5, 7, 3), [0, 127, 255], dtype=np.uint8)
    output, _ = runtime.SpandrelEngine(Capture()).enhance(image, outscale=2)
    assert captured[0] == pytest.approx([1, 127/255, 0])
    assert output.shape == (10, 14, 3)
    np.testing.assert_array_equal(output[0, 0], image[0, 0])


@pytest.mark.parametrize('nonfinite', [False, True])
def test_failed_tile_aborts_frame(tensor_model, nonfinite):
    import numpy as np
    class Broken(type(tensor_model)):
        calls = 0
        def __call__(self, tensor):
            self.calls += 1
            if self.calls == 2:
                if nonfinite:
                    return super().__call__(tensor) * float('nan')
                raise RuntimeError('out of memory')
            return super().__call__(tensor)
    model = Broken()
    with pytest.raises((RuntimeError, ValueError), match='out of memory|non-finite'):
        runtime.SpandrelEngine(model, tile=3).enhance(np.zeros((7, 9, 3), dtype=np.uint8))
    assert model.calls == 2


def test_real_spandrel_loads_official_weight_layout(tmp_path):
    """A synthetic x4plus state verifies real architecture detection, not mocks."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('spandrel')
    # Original RealESRGAN_x4plus key names/shapes (23 RRDBs with 3 dense blocks).
    # Zero weights keep the fixture compressible and yield a predictable output.
    state = {}
    def conv(name, out_channels, in_channels):
        state[name + '.weight'] = torch.zeros(out_channels, in_channels, 3, 3)
        state[name + '.bias'] = torch.zeros(out_channels)
    conv('conv_first', 64, 3)
    for block in range(23):
        for dense in range(1, 4):
            for layer in range(1, 6):
                conv(f'body.{block}.rdb{dense}.conv{layer}', 64 if layer == 5 else 32, 64 + (layer-1)*32)
    for name in ('conv_body', 'conv_up1', 'conv_up2', 'conv_hr'):
        conv(name, 64, 64)
    conv('conv_last', 3, 64)
    path = tmp_path / 'model.pth'
    torch.save({'params_ema': state, 'params': {'invalid': torch.zeros(1)}}, path)
    model = runtime._load_model(path)
    assert model.scale == 4 and model.architecture.id == 'ESRGAN'
    with torch.inference_mode():
        output = model.eval()(torch.ones(1, 3, 2, 3))
    assert output.shape == (1, 3, 8, 12)
    assert torch.count_nonzero(output) == 0


def test_official_checkpoint_inference_when_provided():
    """Opt-in network-free regression against downloaded official weights."""
    import os
    path = os.environ.get('LOS80_TEST_MODEL')
    if not path:
        pytest.skip('set LOS80_TEST_MODEL to a cached official x4plus checkpoint')
    torch = pytest.importorskip('torch')
    np = pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    model = runtime._load_model(Path(path)).eval()
    image = np.random.default_rng(21).integers(0, 256, (8, 10, 3), dtype=np.uint8)
    whole, _ = runtime.SpandrelEngine(model).enhance(image)
    # Pad covers this tiny input so every tile has the same inference context.
    tiled, _ = runtime.SpandrelEngine(model, tile=6, tile_pad=10).enhance(image)
    assert whole.shape == (32, 40, 3) and whole.dtype == np.uint8
    assert whole.std() > 1  # Real output, not uninitialized/black frames.
    np.testing.assert_array_equal(tiled, whole)
