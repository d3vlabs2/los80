"""Verified reassociation of legacy frame caches; never reruns neural inference."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from los80.upscaler import UpscalingError


def _pixels(path):
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        return image.size, image.mode, hashlib.sha256(image.tobytes()).hexdigest()


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _publish(source, destination):
    temporary = destination.with_name(destination.name + '.migration.partial')
    temporary.unlink(missing_ok=True)
    _link_or_copy(source, temporary)
    temporary.replace(destination)


def finish_migration(work_dir, manifest):
    """Replay a fully staged transaction after interruption, without re-decoding."""
    transaction = work_dir / '.legacy-migration'
    if not transaction.exists():
        return False
    journal = json.loads((transaction / 'ready.json').read_text())
    if (transaction / 'complete').exists() and all(
            (work_dir / kind / entry['canonical']).is_file()
            for entry in journal['mapping'] for kind in ('source', 'upscaled')):
        return False
    if journal['manifest'] != manifest:
        raise UpscalingError('Legacy migration manifest changed; cached frames were preserved')
    # Validate the entire snapshot before publishing anything. All old restored
    # frames remain reachable here even when old/new filenames overlap.
    for entry in journal['mapping']:
        for kind in ('source', 'upscaled'):
            if _digest(transaction / kind / entry['canonical']) != entry[kind + '_sha256']:
                raise UpscalingError('Legacy migration snapshot damaged; cached frames were preserved')
    for entry in journal['mapping']:
        for kind in ('source', 'upscaled'):
            _publish(transaction / kind / entry['canonical'], work_dir / kind / entry['canonical'])
    (transaction / 'complete').write_text('complete\n')
    print(f"legacy migration: {len(journal['mapping'])} verified frames reassociated; no neural inference", flush=True)
    return True


def has_legacy_pairs(work_dir, manifest):
    canonical = {frame['name'] for frame in manifest['frames']}
    pairs = [path for path in (work_dir / 'source').iterdir()
             if re.fullmatch(r'frame-(-?\d+)\.png', path.name)
             and (work_dir / 'upscaled' / path.name).is_file()]
    return len(pairs) >= len(canonical) and any(path.name not in canonical for path in pairs)


def migrate_legacy_frames(work_dir, manifest, decoded, scale):
    """Require a unique monotonic affine timestamp mapping AND exact source pixels.

    Fresh sequential decoding establishes frame N independently of image2's
    timestamp rescaling. Filename count/order alone is never sufficient proof.
    """
    canonical = {entry['name'] for entry in manifest['frames']}
    groups = {}
    for path in (work_dir / 'source').iterdir():
        match = re.fullmatch(r'frame-(-?\d+)\.png', path.name)
        if not match or not (work_dir / 'upscaled' / path.name).is_file():
            continue
        try:
            signature = _pixels(path)
        except (OSError, ValueError):
            continue
        digits = match[1]
        groups.setdefault(len(digits), {})[int(digits)] = (path, signature)
    if not any(path.name not in canonical for group in groups.values() for path, _ in group.values()):
        return False
    signatures = [_pixels(path) for path in decoded]
    pts = [entry['pts'] for entry in manifest['frames']]
    mappings = []
    for group in groups.values():
        starts = [number for number, (_, sig) in group.items() if sig == signatures[0]]
        ends = [number for number, (_, sig) in group.items() if sig == signatures[-1]]
        for first in starts:
            for last in ends:
                if len(pts) > 1 and last <= first:
                    continue
                ratio = Fraction(last-first, pts[-1]-pts[0]) if len(pts) > 1 else Fraction(1)
                names = []
                for timestamp, signature in zip(pts, signatures):
                    number = first + (timestamp-pts[0]) * ratio
                    candidate = group.get(number) if number.denominator == 1 else None
                    if candidate is None or candidate[1] != signature:
                        break
                    names.append(candidate[0].name)
                if len(names) == len(pts) and names not in mappings:
                    mappings.append(names)
                if len(mappings) > 1:
                    raise UpscalingError('Ambiguous legacy frame sequence; refusing to guess or rerun inference')
    if not mappings:
        raise UpscalingError('Cannot verify legacy source-frame sequence against decoded video; refusing to rerun inference')
    names = mappings[0]
    # Check every expensive restored image before making any cache changes.
    for name, (size, _, _) in zip(names, signatures):
        try:
            restored_size, _, _ = _pixels(work_dir / 'upscaled' / name)
        except (OSError, ValueError) as exc:
            raise UpscalingError(f'Invalid legacy restored frame {name}; cached frames preserved') from exc
        if restored_size != tuple(int(value * scale) for value in size):
            raise UpscalingError(f'Legacy restored frame has wrong dimensions: {name}')
    with tempfile.TemporaryDirectory(prefix='.migration-stage-', dir=work_dir) as directory:
        staged = Path(directory)
        for kind in ('source', 'upscaled'):
            (staged / kind).mkdir()
        mapping = []
        for name, frame in zip(names, manifest['frames']):
            entry = {'legacy': name, 'canonical': frame['name'], 'pts': frame['pts']}
            for kind in ('source', 'upscaled'):
                target = staged / kind / frame['name']
                _link_or_copy(work_dir / kind / name, target)
                entry[kind + '_sha256'] = _digest(target)
            mapping.append(entry)
        (staged / 'ready.json').write_text(json.dumps({'manifest': manifest, 'mapping': mapping}, indent=2))
        staged.replace(work_dir / '.legacy-migration')
    return finish_migration(work_dir, manifest)
