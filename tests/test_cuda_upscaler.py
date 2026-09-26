from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from PIL import Image

from los80 import upscaler as u
from los80.smoke import smoke_frame
from los80.torch_realesrgan import prepare_engine


@pytest.mark.parametrize('available', [True, False])
def test_selection_uses_torch_not_nvidia_smi(monkeypatch, available):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available)))
    monkeypatch.setattr(u.shutil, 'which', lambda _: '/usr/bin/nvidia-smi')
    assert u.select_backend() is (u.TorchRealESRGANBackend if available else u.RealESRGANBackend)
    assert u.AIUpscaler().detect_device() == ('cuda' if available else 'cpu')


def test_no_torch_falls_back_to_ncnn(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert u.select_backend() is u.RealESRGANBackend


def test_cuda_failure_never_uses_ncnn(tmp_path, monkeypatch):
    monkeypatch.setattr(u, 'cuda_available', lambda: True)
    monkeypatch.setattr(u.RealESRGANRuntime, 'ensure', lambda _: pytest.fail('NCNN selected'))
    def fail(*args):
        raise u.UpscalingError('CUDA setup failed')
    monkeypatch.setattr(u.TorchRealESRGANBackend, 'prepare', fail)
    with pytest.raises(u.UpscalingError, match='CUDA setup failed'):
        u.AIUpscaler().upscale(tmp_path / 'in.mp4', tmp_path / 'out.mp4')


@pytest.mark.parametrize('half', [True, False])
def test_engine_configuration_and_tile_failures(monkeypatch, half):
    captured = {}
    def forward(*args):
        raise RuntimeError('out of memory')
    def engine(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model=SimpleNamespace(forward=forward))
    modules = {
        'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), device=lambda x: x),
        'torchvision.transforms.functional_tensor': SimpleNamespace(),
        'basicsr.archs.rrdbnet_arch': SimpleNamespace(RRDBNet=lambda **kw: kw),
        'basicsr.utils.download_util': SimpleNamespace(load_file_from_url=lambda **kw: '/tmp/model.pth'),
        'realesrgan': SimpleNamespace(RealESRGANer=engine),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    result, actual_half = prepare_engine({'half': half, 'tile_size': 256, 'tile_padding': 10})
    assert actual_half is half and captured['half'] is half
    assert captured['device'] == 'cuda'
    assert captured['tile'] == 256 and captured['tile_pad'] == 10
    assert captured['model']['num_block'] == 23
    with pytest.raises(u.UpscalingError, match='out of memory'):
        result.model.forward(None)


@pytest.fixture
def frame_backend(monkeypatch):
    # Exercise actual frame IO/atomic rename using Pillow-backed cv2 stand-ins.
    class Frame:
        def __init__(self, size):
            self.shape = (size[1], size[0], 3)
    def read(path, flags):
        try:
            with Image.open(path) as image:
                image.load()
                return Frame(image.size)
        except OSError:
            return None
    def write(path, frame):
        Image.new('RGB', (frame.shape[1], frame.shape[0])).save(path)
        return True
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(IMREAD_UNCHANGED=-1, imread=read, imwrite=write))
    backend = u.TorchRealESRGANBackend()
    backend.engine = SimpleNamespace(enhance=lambda frame, outscale: (Frame((int(frame.shape[1]*outscale), int(frame.shape[0]*outscale))), 'RGB'))
    return backend


def test_frame_name_resolution_and_atomic_failure(tmp_path, frame_backend):
    source = tmp_path / 'input.png'
    Image.new('RGB', (8, 6)).save(source)
    output = tmp_path / 'frame-00000000000000001540.png'
    frame_backend.upscale_frame(source, output, {})
    assert Image.open(output).size == (32, 24)
    original = output.read_bytes()
    def fail(*args, **kwargs):
        raise RuntimeError('OOM')
    frame_backend.engine.enhance = fail
    with pytest.raises(u.UpscalingError, match='input.png.*OOM'):
        frame_backend.upscale_frame(source, output, {})
    assert output.read_bytes() == original
    assert not list(tmp_path.glob('*.partial.png'))


def test_resume_skips_valid_retries_corrupt_and_missing(tmp_path, frame_backend, monkeypatch):
    source = tmp_path / 'source'; source.mkdir()
    output = tmp_path / 'upscaled'; output.mkdir()
    names = [f'frame-{pts:020d}.png' for pts in (0, 40, 80)]
    for name in names:
        Image.new('RGB', (8, 6)).save(source / name)
    Image.new('RGB', (32, 24)).save(output / names[0])
    (output / names[1]).write_bytes(b'broken')
    called = []
    original = frame_backend.upscale_frame
    def upscale(src, dst, config):
        called.append(src.name)
        return original(src, dst, config)
    monkeypatch.setattr(frame_backend, 'upscale_frame', upscale)
    frame_backend._process_frames([], tmp_path, output, None, 'RealESRGAN_x4plus', {})
    assert called == names[1:]
    assert sorted(p.name for p in output.iterdir()) == names


def test_failed_write_and_bad_input(tmp_path, frame_backend, monkeypatch):
    source = tmp_path / 'input.png'; output = tmp_path / 'output.png'
    source.write_bytes(b'bad')
    with pytest.raises(u.UpscalingError, match='Cannot decode'):
        frame_backend.upscale_frame(source, output, {})
    Image.new('RGB', (8, 6)).save(source)
    monkeypatch.setattr(sys.modules['cv2'], 'imwrite', lambda *args: False)
    with pytest.raises(u.UpscalingError, match='Cannot write'):
        frame_backend.upscale_frame(source, output, {})
    assert not output.exists()


def test_smoke_reports_cuda_and_uses_frame_api(tmp_path, frame_backend, monkeypatch, capsys):
    import los80.smoke as smoke
    source = tmp_path / 'input.png'; output = tmp_path / 'output.png'
    Image.new('RGB', (8, 6)).save(source)
    monkeypatch.setattr(smoke, 'cuda_available', lambda: True)
    monkeypatch.setattr(smoke, 'select_backend', lambda: lambda: frame_backend)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda _: 'Tesla T4')))
    frame_backend.half = True
    monkeypatch.setattr(frame_backend, 'prepare', lambda _: (None, 'RealESRGAN_x4plus'))
    smoke_frame(source, output, {'scale': 4}, require_cuda=True)
    report = capsys.readouterr().out
    for text in ['TorchRealESRGANBackend', 'CUDA availability: True', 'Tesla T4', 'FP16', '8x6', '32x24', 'scale factor: 4', 'processing time', str(output)]:
        assert text in report


def test_smoke_requires_cuda(tmp_path, monkeypatch):
    import los80.smoke as smoke
    source = tmp_path / 'input.png'; Image.new('RGB', (8, 6)).save(source)
    monkeypatch.setattr(smoke, 'cuda_available', lambda: False)
    with pytest.raises(u.UpscalingError, match='CUDA unavailable'):
        smoke_frame(source, tmp_path / 'output.png', {}, require_cuda=True)


def test_smoke_cli_does_not_open_database_or_run_pipeline(monkeypatch):
    from los80 import run
    import los80.smoke as smoke
    called = []
    monkeypatch.setattr(smoke, 'smoke_frame', lambda *a, **kw: called.append((a, kw)))
    monkeypatch.setattr(run, 'JobDatabase', lambda *a: pytest.fail('database opened'))
    monkeypatch.setattr(run, 'Pipeline', lambda *a: pytest.fail('pipeline started'))
    monkeypatch.setattr(sys, 'argv', ['los80', 'smoke', '--require-cuda'])
    run.main()
    assert called[0][0][:2] == (Path('/content/los80_smoke/input.png'), Path('/content/los80_smoke/output.png'))
    assert called[0][1]['require_cuda'] is True


def test_cuda_video_reuses_remux_and_retains_frames_on_failure(tmp_path, frame_backend, monkeypatch):
    output = tmp_path / 'clip.mkv'
    work = tmp_path / '.clip.mkv.realesrgan-frames'
    commands = []
    metadata = {'time_base': '1/1000', 'avg_frame_rate': '25/1', 'start_time': '1.5',
                'sample_aspect_ratio': '4:3', 'pix_fmt': 'yuv420p', 'color_primaries': 'bt709'}
    monkeypatch.setattr(frame_backend, 'prepare', lambda config: (None, 'RealESRGAN_x4plus'))
    monkeypatch.setattr(frame_backend, '_probe', lambda *a: metadata)
    monkeypatch.setattr(u.shutil, 'which', lambda name: name)
    fail_remux = True
    concat = []
    def run(command, operation):
        commands.append(command)
        if '-frame_pts' in command:
            for pts in (1500, 1540, 1580):
                Image.new('RGB', (8, 6)).save(work / 'source' / f'frame-{pts:020d}.png')
        else:
            concat.append((work / 'frames.ffconcat').read_text())
            if fail_remux:
                Path(command[-1]).write_bytes(b'partial video')
                raise u.UpscalingError('remux failed')
            Path(command[-1]).write_bytes(b'video')
    monkeypatch.setattr(frame_backend, '_run', run)
    with pytest.raises(u.UpscalingError, match='remux failed'):
        frame_backend.upscale(tmp_path / 'source.mkv', output, {})
    assert (work / '.extraction-complete').exists()
    assert not output.exists()
    assert not list(tmp_path.glob('*.partial.mkv'))
    assert len(list((work / 'upscaled').glob('frame-*.png'))) == 3
    monkeypatch.setattr(frame_backend, 'upscale_frame', lambda *a: pytest.fail('valid frame reprocessed'))
    fail_remux = False
    frame_backend.upscale(tmp_path / 'source.mkv', output, {})
    assert sum('-frame_pts' in cmd for cmd in commands) == 1
    remux = commands[-1]
    for option, value in [('-map_metadata', '1'), ('-map_chapters', '1'), ('-c:a', 'copy'),
                          ('-c:s', 'copy'), ('-fps_mode', 'vfr'), ('-vf', 'setsar=4/3'),
                          ('-pix_fmt', 'yuv420p'), ('-color_primaries', 'bt709'), ('-itsoffset', '1.5')]:
        assert remux[remux.index(option) + 1] == value
    assert all(mapping in remux for mapping in ['0:v:0', '1:a?', '1:s?'])
    assert concat[0].count('duration 0.040000000') == 3
    assert not work.exists()


def test_ncnn_smoke_fallback(tmp_path, monkeypatch, capsys):
    import los80.smoke as smoke
    source = tmp_path / 'input.png'; Image.new('RGB', (8, 6)).save(source)
    output = tmp_path / 'output.png'
    monkeypatch.setattr(smoke, 'cuda_available', lambda: False)
    monkeypatch.setattr(smoke, 'select_backend', lambda: u.RealESRGANBackend)
    runtime = SimpleNamespace(executable=Path('/ncnn'), model_dir=tmp_path)
    monkeypatch.setattr(u.RealESRGANBackend, 'prepare', lambda *a: (runtime, 'realesrgan-x4plus'))
    def run(command, operation):
        Image.new('RGB', (32, 24)).save(command[command.index('-o') + 1])
    monkeypatch.setattr(u.RealESRGANBackend, '_run', staticmethod(run))
    smoke_frame(source, output, {})
    assert Image.open(output).size == (32, 24)
    assert 'CUDA availability: False' in capsys.readouterr().out


def test_cuda_startup_skips_ncnn_runtime(monkeypatch):
    from los80 import run
    monkeypatch.setattr(run, 'cuda_available', lambda: True)
    monkeypatch.setattr(run, 'RealESRGANRuntime', lambda *a: pytest.fail('NCNN startup'))
    monkeypatch.setattr(run, 'JobDatabase', lambda *a: None)
    monkeypatch.setattr(run, 'Pipeline', lambda *a: SimpleNamespace(run=lambda: {}))
    monkeypatch.setattr(sys, 'argv', ['los80'])
    run.main()


@pytest.mark.parametrize("fps", ["25", "30000/1001"])
def test_real_ffmpeg_reassembly_with_cuda_frame_adapter(tmp_path, frame_backend, monkeypatch, fps):
    """Run real media IO around a fake neural model; no weights/GPU required."""
    import json
    import shutil
    import subprocess
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg/FFprobe not installed')
    subtitle = tmp_path / 'sub.srt'
    subtitle.write_text('1\n00:00:00,000 --> 00:00:00,200\nHello\n')
    metadata = tmp_path / 'metadata.txt'
    metadata.write_text(';FFMETADATA1\ntitle=Smoke fixture\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=240\ntitle=Opening\n')
    source = tmp_path / 'source.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=size=16x16:rate={fps}:duration=0.24',
                    '-f', 'lavfi', '-i', 'sine=duration=0.24', '-i', str(subtitle), '-i', str(metadata),
                    '-map', '0:v', '-map', '1:a', '-map', '2:s', '-map_metadata', '3', '-map_chapters', '3',
                    '-vf', 'setsar=4/3', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le',
                    '-c:s', 'srt', str(source)], check=True, capture_output=True)
    monkeypatch.setattr(frame_backend, 'prepare', lambda config: (None, 'RealESRGAN_x4plus'))
    output = tmp_path / 'output.mkv'
    frame_backend.upscale(source, output, {})
    def probe(path):
        return json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                                         '-show_chapters', '-show_format', '-of', 'json', str(path)],
                                        check=True, capture_output=True, text=True).stdout)
    before, after = probe(source), probe(output)
    assert [s['codec_type'] for s in after['streams']] == ['video', 'audio', 'subtitle']
    video = after['streams'][0]
    assert (video['width'], video['height']) == (64, 64)
    assert video['sample_aspect_ratio'] == before['streams'][0]['sample_aspect_ratio']
    assert video['avg_frame_rate'] == before['streams'][0]['avg_frame_rate']
    assert video['nb_read_frames'] == before['streams'][0]['nb_read_frames']
    assert after['chapters'] == before['chapters']
    assert after['format']['tags']['title'] == 'Smoke fixture'
    assert after['streams'][1]['codec_name'] == 'pcm_s16le'
    assert after['streams'][2]['codec_name'] == 'subrip'
    assert abs(float(after['format']['duration']) - float(before['format']['duration'])) < 0.05
