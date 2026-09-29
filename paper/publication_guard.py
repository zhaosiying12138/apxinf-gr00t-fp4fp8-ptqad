"""Small dependency-free publication provenance helpers."""
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def resolve_inside(root, relative):
    root = Path(root).resolve()
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            f'Expected a relative evidence path: {relative!r}')
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f'Evidence path escapes its root: {relative}')
    require(path.is_file(), f'Missing evidence file: {path}')
    return path


def file_record(path, root):
    path = Path(path).resolve(); root = Path(root).resolve()
    require(path.is_relative_to(root), f'File is outside evidence root: {path}')
    payload = path.read_bytes()
    return {'path': path.relative_to(root).as_posix(), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}


def check_record(row, root):
    path = resolve_inside(root, row['path'])
    actual = file_record(path, root)
    require(actual['sha256'] == row['sha256'], f'Changed evidence file: {path}')
    if 'bytes' in row:
        require(actual['bytes'] == row['bytes'], f'Changed evidence size: {path}')
    return path


def html_build_inputs(paper):
    paper = Path(paper).resolve()
    names = ['build_html.py', 'render_math.cjs', 'publication_guard.py', 'meta.json', 'figures.json']
    paths = [paper / name for name in names]
    paths.extend(sorted((paper / 'sections').glob('*.md')))
    figures = json.loads((paper / 'figures.json').read_text())
    for name in figures:
        path = paper / 'figs' / (name + '.svg')
        if not path.exists(): path = path.with_suffix('.png')
        paths.append(path)
    paths.extend(path for path in (paper / 'assets/katex').rglob('*') if path.is_file())
    require(any(path.suffix == '.woff2' for path in paths), 'Missing offline math fonts')
    return [file_record(path, paper) for path in sorted(set(paths))]


def capture_crop_contract(sidecar, crop, image, expected_dimensions):
    require(sidecar.get('capture_status') == 'accepted_not_black', 'Capture was not accepted by the black-screen gate')
    require(type(sidecar.get('exit_status')) is int and sidecar['exit_status'] == 0, 'Capture command did not succeed')
    quality = sidecar['image_quality']
    require(quality.get('accepted_not_black') is True, 'Capture image-quality check failed')
    source_dimensions = [quality['width'], quality['height']]
    require(all(type(value) is int and value > 0 for value in source_dimensions), 'Invalid capture dimensions')
    require(crop.get('operation') == 'taskbar_crop_only', 'Unexpected image processing')
    require(crop['source_sha256'] == sidecar['sha256_before_taskbar_crop'], 'Crop is not linked to the accepted capture')
    require(crop['source_dimensions'] == source_dimensions, 'Crop source dimensions differ from accepted capture')
    require(crop['output_dimensions'] == expected_dimensions, 'Crop does not match the approved sample dimensions')
    width, height = expected_dimensions
    require(source_dimensions[0] == width and source_dimensions[1] >= height and
            crop['crop_box'] == [0, 0, width, height], 'Only the bottom taskbar may be cropped')
    require(hashlib.sha256(Path(image).read_bytes()).hexdigest() == crop['output_sha256'], 'Image differs from the recorded crop')
