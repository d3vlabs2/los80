from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import sys

import pytest

from los80 import motion_smoke as motion
from los80.upscaler import TorchRealESRGANBackend, UpscalingError


@pytest.fixture
def clip(tmp_path, request):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg required')
    duration = getattr(request, 'param', 1.001)
    source = tmp_path / 'original.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc2=size=32x24:rate=24000/1001:duration={duration}',
                    '-f', 'lavfi', '-i', f'sine=duration={duration}', '-vf', 'setsar=4/3',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(source)],
                   check=True, capture_output=True)
    return source


@pytest.fixture
def fake_cuda(monkeypatch):
    pytest.importorskip('numpy')
    cv2 = pytest.importorskip('cv2')
    monkeypatch.setattr(motion, 'cuda_available', lambda: True)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda _: 'Tesla T4')))
    calls = []
    fail_at = [None]
    def enhance(frame, outscale):
        assert outscale == 4
        calls.append(frame.copy())
        if len(calls) == fail_at[0]:
            raise UpscalingError('simulated interruption')
        return cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 'RGB'
    def prepare(self, config):
        self.half = True
        self.engine = SimpleNamespace(enhance=enhance)
        return None, 'RealESRGAN_x4plus'
    monkeypatch.setattr(TorchRealESRGANBackend, 'prepare', prepare)
    return calls, fail_at


def test_real_motion_fps_audio_blend_atomic_output_and_resume(clip, fake_cuda):
    import cv2
    import numpy as np
    from los80.torch_realesrgan import restoration_variant
    calls, _ = fake_cuda
    report = motion.smoke_motion(clip, require_cuda=True)
    output = clip.parent / 'restored_2x_strength0.25.mp4'
    assert output.is_file()
    assert report['source FPS'] == report['output FPS'] == '24000/1001'
    assert report['number of frames'] == report['frames inferred this run'] == 24
    assert report['output resolution'] == '64x48'
    assert report['precision'] == 'FP16'
    assert len(calls) == 24
    before, times = motion._probe_motion(clip, 'ffprobe')
    after, out_times = motion._probe_motion(output, 'ffprobe')
    assert times == out_times
    assert before['sample_aspect_ratio'] == after['sample_aspect_ratio'] == '4:3'
    assert motion._audio_packets(clip, 'ffprobe') == motion._audio_packets(output, 'ffprobe')
    work = clip.parent / f'.{output.name}.realesrgan-frames'
    source_frames = sorted((work / 'source').glob('frame-*.png'))
    assert len(source_frames) == 24
    for source in source_frames:
        original = cv2.imread(str(source))
        neural = cv2.resize(original, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        expected = restoration_variant(original, neural, 2, .25)
        np.testing.assert_array_equal(cv2.imread(str(work / 'upscaled' / source.name)), expected)
    assert not list(clip.parent.glob('*.partial.mp4'))
    second = motion.smoke_motion(clip)
    assert second['frames inferred this run'] == 0 and second['frames resumed'] == 24
    assert 'all frames resumed' in second['average inference time per new frame']
    assert len(calls) == 24


def test_interrupted_motion_resumes_valid_frames_and_retries_corrupt(clip, fake_cuda):
    calls, fail_at = fake_cuda
    fail_at[0] = 4
    with pytest.raises(UpscalingError, match='simulated interruption'):
        motion.smoke_motion(clip)
    output = clip.parent / 'restored_2x_strength0.25.mp4'
    work = clip.parent / f'.{output.name}.realesrgan-frames'
    assert not output.exists()
    frames = sorted((work / 'upscaled').glob('frame-*.png'))
    assert len(frames) == 3
    frames[1].write_bytes(b'corrupt frame')
    fail_at[0] = None
    report = motion.smoke_motion(clip)
    assert report['frames resumed'] == 2
    assert report['frames inferred this run'] == 22
    assert len(calls) == 26


@pytest.mark.parametrize('change', ['strength', 'source', 'tile'])
def test_resume_identity_rejects_changed_source_or_settings(clip, fake_cuda, change):
    calls, fail_at = fake_cuda
    fail_at[0] = 2
    with pytest.raises(UpscalingError):
        motion.smoke_motion(clip)
    output = clip.parent / 'restored_2x_strength0.25.mp4'
    config = {}
    if change == 'strength':
        config['restoration_strength'] = .5
    elif change == 'tile':
        config['tile_size'] = 128
    else:
        with clip.open('ab') as file:
            file.write(b'changed')
    with pytest.raises(UpscalingError, match='settings/source changed'):
        motion.smoke_motion(clip, output, config)
    assert len(calls) == 2


def test_failed_validation_keeps_previous_output_and_resumable_frames(clip, fake_cuda, monkeypatch):
    output = clip.parent / 'restored_2x_strength0.25.mp4'
    output.write_bytes(b'previous completed output')
    original = motion._validate_motion
    def fail(*args):
        raise UpscalingError('validation failure')
    monkeypatch.setattr(motion, '_validate_motion', fail)
    with pytest.raises(UpscalingError, match='validation failure'):
        motion.smoke_motion(clip)
    assert output.read_bytes() == b'previous completed output'
    assert not list(clip.parent.glob('*.partial.mp4'))
    monkeypatch.setattr(motion, '_validate_motion', original)
    report = motion.smoke_motion(clip)
    assert report['frames resumed'] == 24 and report['frames inferred this run'] == 0


def test_motion_no_cuda_fails_before_touching_files(tmp_path, monkeypatch):
    monkeypatch.setattr(motion, 'cuda_available', lambda: False)
    with pytest.raises(UpscalingError, match='CUDA unavailable'):
        motion.smoke_motion(tmp_path / 'missing.mp4', require_cuda=True)
    assert not list(tmp_path.iterdir())


def test_motion_cli_defaults_without_database_or_pipeline(monkeypatch):
    from los80 import run
    calls = []
    monkeypatch.setattr(motion, 'smoke_motion', lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(run, 'JobDatabase', lambda *a: pytest.fail('database opened'))
    monkeypatch.setattr(run, 'Pipeline', lambda *a: pytest.fail('pipeline started'))
    monkeypatch.setattr(sys, 'argv', ['los80', 'smoke-motion', '--require-cuda', '--scale', '2', '--strength', '0.25'])
    run.main()
    args, kwargs = calls[0]
    assert args[0] == Path('/content/los80_motion/original.mp4')
    assert args[1] is None
    assert args[2]['scale'] == 2 and args[2]['restoration_strength'] == .25
    assert kwargs['require_cuda'] is True


@pytest.mark.parametrize('clip', [10.01], indirect=True)
def test_ten_second_motion_preserves_all_240_frames(clip, fake_cuda):
    report = motion.smoke_motion(clip)
    assert report['number of frames'] == report['frames inferred this run'] == 240
    assert report['source FPS'] == report['output FPS'] == '24000/1001'
    assert len(fake_cuda[0]) == 240


@pytest.mark.parametrize('failure', ['fps', 'frames', 'timestamps', 'aspect', 'resolution', 'audio'])
def test_motion_validation_rejects_media_changes(monkeypatch, failure):
    from fractions import Fraction
    source = {'width': 32, 'height': 24, 'codec_name': 'h264', 'avg_frame_rate': '24000/1001',
              'sample_aspect_ratio': '4:3', 'time_base': '1/24000'}
    output = {**source, 'width': 64, 'height': 48}
    times = [Fraction(0), Fraction(1001, 24000)]
    out_times = list(times)
    audio = [{'data_hash': 'same'}]
    out_audio = audio
    if failure == 'fps':
        output['avg_frame_rate'] = '24/1'
    elif failure == 'frames':
        out_times = out_times[:1]
    elif failure == 'timestamps':
        out_times[1] += Fraction(1, 10)
    elif failure == 'aspect':
        output['sample_aspect_ratio'] = '1:1'
    elif failure == 'resolution':
        output['width'] = 65
    else:
        out_audio = [{'data_hash': 'changed'}]
    monkeypatch.setattr(motion, '_probe_motion', lambda *a: (output, out_times))
    monkeypatch.setattr(motion, '_audio_packets', lambda *a: out_audio)
    with pytest.raises(UpscalingError):
        motion._validate_motion(Path('temporary.mp4'), source, times, audio, 2, 'ffprobe')


@pytest.mark.parametrize('clip', [10.01], indirect=True)
@pytest.mark.parametrize('stale_source', [False, True])
def test_legacy_240_frame_resume_ignores_481_stale_pngs(clip, fake_cuda, monkeypatch, capsys, stale_source):
    """721 PNGs in the cache must still mean exactly 240 manifest video frames."""
    import json
    from PIL import Image
    calls, _ = fake_cuda
    first = motion.smoke_motion(clip)
    output = Path(first['output path'])
    work = clip.parent / f'.{output.name}.realesrgan-frames'
    restored = work / 'upscaled'
    source = work / 'source'
    manifest = json.loads((work / 'extraction-manifest.json').read_text())
    names = [frame['name'] for frame in manifest['frames']]
    assert len(names) == 240
    assert [frame['pts'] for frame in manifest['frames']] == [i * 1001 for i in range(240)]
    saved = {name: ((restored / name).read_bytes(), (restored / name).stat().st_mtime_ns) for name in names}
    # Reproduce legacy directories with no per-frame manifest, an old completion
    # marker, stale timestamp files, duplicate suffixes, and a crash leftover.
    (work / 'extraction-manifest.json').unlink()
    for index in range(240):
        stale_name = f'frame-{900000 + index:020d}.png'
        Image.new('RGB', (64, 48), 'red').save(restored / stale_name)
        if stale_source:
            Image.new('RGB', (32, 24), 'red').save(source / stale_name)
        Image.new('RGB', (64, 48), 'blue').save(restored / f'frame-{index:020d}.duplicate.png')
    Image.new('RGB', (64, 48), 'green').save(restored / 'frame-00000000000000000000.partial.png')
    assert len(list(restored.glob('*.png'))) == 721
    # Don't trust even an existing concat file: it may be interrupted or edited.
    (work / 'frames.ffconcat').write_text('\n'.join(f"file '{restored / name}'" for name in names) + '\n')
    original_run = TorchRealESRGANBackend._run
    def run(command, operation):
        assert '-frame_pts' not in command, 'valid source frames should not be extracted again'
        return original_run(command, operation)
    monkeypatch.setattr(TorchRealESRGANBackend, '_run', staticmethod(run))
    for _ in range(2):
        report = motion.smoke_motion(clip)
        assert report['frames resumed'] == 240
        assert report['frames inferred this run'] == 0
        assert report['number of frames'] == 240
        assert report['source FPS'] == report['output FPS'] == '24000/1001'
        assert len(calls) == 240  # no additional neural calls on either rerun
        lines = (work / 'frames.ffconcat').read_text().splitlines()
        references = [line for line in lines if line.startswith('file ')]
        assert references == [f"file '{(restored / name).resolve()}'" for name in names]
        assert lines.count('option pattern_type none') == 240
        assert len(list(restored.glob('*.png'))) == 721  # stale files retained, never consumed
        assert len(motion._probe_motion(output, 'ffprobe')[1]) == 240
    for name, (content, modified) in saved.items():
        assert (restored / name).read_bytes() == content
        assert (restored / name).stat().st_mtime_ns == modified
    assert f'ignoring 481 stale upscaled PNGs and {240 if stale_source else 0} stale source PNGs' in capsys.readouterr().out


def test_missing_source_png_reextracts_without_recomputing_restored(clip, fake_cuda):
    report = motion.smoke_motion(clip)
    output = Path(report['output path'])
    work = clip.parent / f'.{output.name}.realesrgan-frames'
    next((work / 'source').glob('frame-*.png')).unlink()
    repeated = motion.smoke_motion(clip)
    assert repeated['frames resumed'] == 24
    assert repeated['frames inferred this run'] == 0
    assert len(fake_cuda[0]) == 24
    assert not list(work.glob('.extract-*'))


def _make_legacy_timestamp_cache(clip, width=19):
    """Prepare an old cache with 19-digit names in a 100-tick convention."""
    import json
    report = motion.smoke_motion(clip)
    output = Path(report['output path'])
    work = clip.parent / f'.{output.name}.realesrgan-frames'
    manifest = json.loads((work / 'extraction-manifest.json').read_text())
    canonical = [frame['name'] for frame in manifest['frames']]
    saved = {}
    for kind in ('source', 'upscaled'):
        images = [(work / kind / name).read_bytes() for name in canonical]
        for name in canonical:
            (work / kind / name).unlink()
        for index, content in enumerate(images):
            legacy = f'frame-{index*100:0{width}d}.png'
            (work / kind / legacy).write_bytes(content)
            saved[(kind, canonical[index])] = content
    (work / 'extraction-manifest.json').unlink()
    return work, canonical, saved


@pytest.mark.parametrize('clip', [10.01], indirect=True)
@pytest.mark.parametrize('width', [19, 20])
def test_migrate_240_legacy_100_tick_frames_without_inference(clip, fake_cuda, monkeypatch, width):
    import json
    work, names, saved = _make_legacy_timestamp_cache(clip, width)
    assert len(list((work / 'source').glob('*.png'))) == 240
    assert len(list((work / 'upscaled').glob('*.png'))) == 240
    fake_cuda[0].clear()
    monkeypatch.setattr(TorchRealESRGANBackend, 'upscale_frame', lambda *a: pytest.fail('unexpected inference'))
    for _ in range(2):
        report = motion.smoke_motion(clip)
        assert report['frames resumed'] == report['number of frames'] == 240
        assert report['frames inferred this run'] == 0
        assert report['source FPS'] == report['output FPS'] == '24000/1001'
        assert not fake_cuda[0]
        for (kind, name), content in saved.items():
            assert (work / kind / name).read_bytes() == content
        manifest = json.loads((work / 'extraction-manifest.json').read_text())
        assert [frame['pts'] for frame in manifest['frames']] == [i*1001 for i in range(240)]
    assert (work / '.legacy-migration' / 'complete').exists()
    assert len(list((work / 'upscaled').glob('*.png'))) == (480 if width == 19 else 479)


def test_legacy_migration_rejects_equal_counts_with_wrong_pixels(clip, fake_cuda, monkeypatch):
    from PIL import Image
    work, names, saved = _make_legacy_timestamp_cache(clip)
    wrong = work / 'source' / f'frame-{500:019d}.png'
    Image.new('RGB', (32, 24), 'magenta').save(wrong)
    monkeypatch.setattr(TorchRealESRGANBackend, 'upscale_frame', lambda *a: pytest.fail('unexpected inference'))
    with pytest.raises(UpscalingError, match='Cannot verify legacy source-frame sequence'):
        motion.smoke_motion(clip)
    assert not (work / '.legacy-migration').exists()
    for index, name in enumerate(names):
        assert (work / 'upscaled' / f'frame-{index*100:019d}.png').read_bytes() == saved[('upscaled', name)]


def test_interrupted_legacy_migration_replays_snapshot(clip, fake_cuda, monkeypatch):
    from los80 import cache_migration
    work, names, saved = _make_legacy_timestamp_cache(clip)
    original = cache_migration._publish
    publications = []
    def interrupt(source, destination):
        publications.append(destination)
        if len(publications) == 7:
            raise KeyboardInterrupt('migration interrupted')
        original(source, destination)
    monkeypatch.setattr(cache_migration, '_publish', interrupt)
    monkeypatch.setattr(TorchRealESRGANBackend, 'upscale_frame', lambda *a: pytest.fail('unexpected inference'))
    with pytest.raises(KeyboardInterrupt):
        motion.smoke_motion(clip)
    assert (work / '.legacy-migration' / 'ready.json').exists()
    assert not (work / '.legacy-migration' / 'complete').exists()
    monkeypatch.setattr(cache_migration, '_publish', original)
    run = TorchRealESRGANBackend._run
    def no_extract(command, operation):
        assert operation != 'extract video frames'
        return run(command, operation)
    monkeypatch.setattr(TorchRealESRGANBackend, '_run', staticmethod(no_extract))
    report = motion.smoke_motion(clip)
    assert report['frames resumed'] == 24 and report['frames inferred this run'] == 0
    for (kind, name), content in saved.items():
        assert (work / kind / name).read_bytes() == content


def test_legacy_migration_ignores_unrelated_numeric_pngs(clip, fake_cuda, monkeypatch):
    from PIL import Image
    work, _, _ = _make_legacy_timestamp_cache(clip)
    for kind, size in [('source', (32, 24)), ('upscaled', (64, 48))]:
        Image.new('RGB', size, 'red').save(work / kind / f'frame-{999999:019d}.png')
        Image.new('RGB', size, 'red').save(work / kind / 'frame-0001.partial.png')
    monkeypatch.setattr(TorchRealESRGANBackend, 'upscale_frame', lambda *a: pytest.fail('unexpected inference'))
    report = motion.smoke_motion(clip)
    assert report['frames resumed'] == 24 and report['frames inferred this run'] == 0


def test_legacy_restored_frames_reassociated_when_canonical_sources_already_exist(clip, fake_cuda, monkeypatch):
    work, names, saved = _make_legacy_timestamp_cache(clip)
    for name in names:
        (work / 'source' / name).write_bytes(saved[('source', name)])
    monkeypatch.setattr(TorchRealESRGANBackend, 'upscale_frame', lambda *a: pytest.fail('unexpected inference'))
    report = motion.smoke_motion(clip)
    assert report['frames resumed'] == 24 and report['frames inferred this run'] == 0
