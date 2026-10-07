import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from los80 import upscaler as u
from los80.configuration import AppConfig, load_config


@pytest.mark.parametrize('backend', ['ffmpeg', 'auto', None])
def test_ffmpeg_selection_without_neural_initialization(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(u, 'cuda_available', lambda: pytest.fail('CUDA detection'))
    monkeypatch.setattr(u, 'RealESRGANRuntime', lambda *a: pytest.fail('NCNN initialization'))
    monkeypatch.setattr(u.TorchRealESRGANBackend, 'prepare', lambda *a: pytest.fail('CUDA initialization'))
    commands = []
    output = tmp_path / 'output.mkv'
    def execute(command, **kwargs):
        commands.append(command)
        assert not output.exists()
        Path(command[-1]).write_bytes(b'video')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(u.subprocess, 'run', execute)
    if backend is not None:
        assert u.select_backend(backend) is u.FFmpegBackend
    config = {'backend': backend} if backend is not None else None
    assert u.AIUpscaler().upscale(tmp_path / 'source.mkv', output, config) == output
    assert output.read_bytes() == b'video'
    assert not list(tmp_path.glob('*.partial.mkv'))
    command = commands[0]
    for option, value in [('-vf', 'scale=iw*2:ih*2:flags=lanczos'), ('-c:a', 'copy'),
                          ('-c:s', 'copy'), ('-map_metadata', '0'), ('-map_chapters', '0'),
                          ('-fps_mode', 'passthrough')]:
        assert command[command.index(option) + 1] == value
    assert '-r' not in command and '-frame_pts' not in command
    assert command[-1] == str(tmp_path / 'output.partial.mkv')
    assert '-nostdin' in command
    assert not list(tmp_path.glob('.*realesrgan*'))


@pytest.mark.parametrize('cuda', [True, False])
def test_explicit_realesrgan_selection(monkeypatch, cuda):
    monkeypatch.setattr(u, 'cuda_available', lambda: cuda)
    assert u.select_backend('realesrgan') is (u.TorchRealESRGANBackend if cuda else u.RealESRGANBackend)


@pytest.mark.parametrize('failure', ['exit', 'missing', 'empty', 'launch'])
def test_ffmpeg_failure_never_publishes_partial(tmp_path, monkeypatch, failure):
    output = tmp_path / 'output.mp4'
    def execute(command, **kwargs):
        if failure == 'launch':
            raise FileNotFoundError('ffmpeg missing')
        if failure != 'missing':
            Path(command[-1]).write_bytes(b'partial' if failure == 'exit' else b'')
        return SimpleNamespace(returncode=1 if failure == 'exit' else 0, stderr='failure', stdout='')
    monkeypatch.setattr(u.subprocess, 'run', execute)
    with pytest.raises(u.UpscalingError):
        u.AIUpscaler().upscale(tmp_path / 'source.mp4', output, {'backend': 'ffmpeg'})
    assert not output.exists()
    assert not list(tmp_path.glob('*.partial.mp4'))


def test_failed_validation_preserves_previous_output(tmp_path, monkeypatch):
    output = tmp_path / 'output.mp4'
    output.write_bytes(b'previous')
    def execute(command, **kwargs):
        Path(command[-1]).write_bytes(b'new video')
        return SimpleNamespace(returncode=0, stderr='', stdout='')
    def validate(path):
        assert path.name == 'output.partial.mp4'
        raise ValueError('invalid video')
    monkeypatch.setattr(u.subprocess, 'run', execute)
    with pytest.raises(ValueError, match='invalid video'):
        u.FFmpegBackend().upscale(tmp_path / 'source.mp4', output, {'output_validator': validate})
    assert output.read_bytes() == b'previous'
    assert not list(tmp_path.glob('*.partial.mp4'))


@pytest.mark.parametrize('backend', ['ffmpeg', 'auto'])
def test_batch_startup_does_not_prepare_realesrgan(monkeypatch, backend):
    from los80 import run
    monkeypatch.setattr(run, 'load_config', lambda _: AppConfig(upscaler_backend=backend))
    monkeypatch.setattr(run, 'cuda_available', lambda: pytest.fail('CUDA detection'))
    monkeypatch.setattr(run, 'RealESRGANRuntime', lambda *a: pytest.fail('Real-ESRGAN runtime'))
    monkeypatch.setattr(run, 'JobDatabase', lambda *a: None)
    monkeypatch.setattr(run, 'Pipeline', lambda *a: SimpleNamespace(run=lambda: {}))
    monkeypatch.setattr('sys.argv', ['los80'])
    run.main()


def test_realesrgan_batch_startup_retains_runtime_check(monkeypatch):
    from los80 import run
    prepared = []
    monkeypatch.setattr(run, 'load_config', lambda _: AppConfig(upscaler_backend='realesrgan'))
    monkeypatch.setattr(run, 'cuda_available', lambda: False)
    monkeypatch.setattr(run, 'RealESRGANRuntime', lambda *a: SimpleNamespace(ensure=lambda: prepared.append(True)))
    monkeypatch.setattr(run, 'JobDatabase', lambda *a: None)
    monkeypatch.setattr(run, 'Pipeline', lambda *a: SimpleNamespace(run=lambda: {}))
    monkeypatch.setattr('sys.argv', ['los80'])
    run.main()
    assert prepared == [True]


@pytest.mark.parametrize('config_path', ['config.yaml', 'config/config.yaml'])
def test_production_defaults(config_path):
    config = load_config(config_path)
    assert config.upscaler_backend == 'ffmpeg'
    assert config.upscaler_scale == 2


@pytest.mark.parametrize('backend', ['ffmpeg', 'auto', 'realesrgan'])
def test_backend_configuration(tmp_path, backend):
    path = tmp_path / 'config.yaml'
    path.write_text(f'upscaler_backend: {backend}\nupscaler_scale: 3\n')
    config = load_config(path)
    assert config.upscaler_backend == backend and config.upscaler_scale == 3
    path.write_text(f'ai_upscaling:\n  backend: {backend}\n  scale: 2\n')
    assert load_config(path).upscaler_backend == backend
    path.write_text('upscaler_backend: typo\n')
    with pytest.raises(ValueError, match='Unsupported upscaler_backend'):
        load_config(path)


@pytest.mark.parametrize('fps', ['25', '30000/1001'])
def test_real_ffmpeg_preserves_media(tmp_path, fps):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg/FFprobe not installed')
    subtitle = tmp_path / 'sub.srt'
    subtitle.write_text('1\n00:00:00,000 --> 00:00:00,200\nHello\n')
    metadata = tmp_path / 'metadata.txt'
    metadata.write_text(';FFMETADATA1\ntitle=Fixture\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=240\ntitle=Opening\n')
    source, output = tmp_path / 'source.mkv', tmp_path / 'output.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=size=16x12:rate={fps}:duration=0.24',
                    '-f', 'lavfi', '-i', 'sine=duration=0.24', '-i', str(subtitle), '-i', str(metadata),
                    '-map', '0:v', '-map', '1:a', '-map', '2:s', '-map_metadata', '3', '-map_chapters', '3',
                    '-vf', 'setsar=4/3', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le',
                    '-c:s', 'srt', str(source)], check=True, capture_output=True)
    u.AIUpscaler().upscale(source, output, {'backend': 'ffmpeg'})
    def probe(path):
        return json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                                         '-show_chapters', '-show_format', '-of', 'json', str(path)],
                                        check=True, capture_output=True, text=True).stdout)
    before, after = probe(source), probe(output)
    assert [s['codec_type'] for s in after['streams']] == ['video', 'audio', 'subtitle']
    video = after['streams'][0]
    assert (video['width'], video['height']) == (32, 24)
    for key in ['sample_aspect_ratio', 'avg_frame_rate', 'nb_read_frames']:
        assert video[key] == before['streams'][0][key]
    assert after['chapters'] == before['chapters']
    assert after['format']['tags']['title'] == 'Fixture'
    assert after['streams'][1]['codec_name'] == 'pcm_s16le'
    assert after['streams'][2]['codec_name'] == 'subrip'
    assert abs(float(after['format']['duration']) - float(before['format']['duration'])) < 0.05
