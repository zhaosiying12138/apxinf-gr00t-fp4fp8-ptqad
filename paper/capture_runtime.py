#!/usr/bin/env python3
"""Capture current runtime and explicitly selected checkpoint provenance."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import re

ROOT = Path(__file__).resolve().parents[1]

# Public, maintained implementation only. Do not glob internal experiments into
# the publication manifest: every entry must be shipped by the public export.
DEFAULT_SOURCE_FILES = (
    'quant/fp4_quant.py', 'quant/torch_fp4.py',
    'quant/nvfp4_convert.py', 'quant/nvfp4_convert_packed.py',
    'quant/ptq/awq.py', 'quant/ptq/bake.py', 'quant/ptq/collector.py',
    'quant/ptq/folds.py', 'quant/ptq/quantizers.py', 'quant/ptq/verify_calibration.py',
    'rl/activation_checkpoint.py', 'rl/capture_onpolicy.py', 'rl/gr00t_runtime.py',
    'rl/lora_inventory.py', 'rl/lora_merge_bake.py', 'rl/lora_qad.py',
    'rl/lora_scope.py', 'rl/opd_probe_cache.py', 'rl/probe_distill.py',
    'rl/recovery_batch.py', 'rl/runtime_metrics.py', 'rl/scoped_quant.py',
    'rl/run_onpolicy_round.sh',
    'eval/compare_development.py', 'eval/compare_recovery.py',
    'eval/compare_ptq_frontier.py',
    'eval/rollout_seeded.py', 'eval/run_gr00t_server_fp4vla.py',
    'eval/run_recovery_eval.py', 'eval/serve_recovery.py',
    'exp/bench_engine.py', 'exp/prepare_native_pi05.py',
    'exp/prepare_native_pi05.sh', 'exp/run_native_graph_gates.sh', 'exp/recovery_protocol.json',
    'exp/reproduce_ptqad.sh',
    'exp/recovery_protocol_v5_exploratory_fp4.json', 'exp/run_category_development.py',
    'exp/run_category_recovery.py',
    'exp/verify_teacher_cache_cpu.py',
    "exp/run_development.py", "exp/run_development.sh", "exp/recipe_inventory.py", 'exp/run_ptq_frontier.sh', 'exp/ptq_frontier_protocol.json',
    'baselines/bench_gr00t_pt.py', 'baselines/bench_pi05_lerobot.py',
    'patches/apxinf-fp4vla-engine.patch', 'patches/gr00t-recovery-runtime.patch',
    'setup/02_build_engine.sh', 'setup/03_download_weights.sh',
    'setup/05_restore_all.sh', 'setup/06_install_recovery.sh',
    'setup/verify_weights.py', 'setup/locks/model-sources.json',
    'setup/locks/manifest.json', 'paper/capture_runtime.py',
)


def resolve_source_files(root, extra=()):
    """Resolve a public source allowlist; reject missing and external paths."""
    root = Path(root).resolve(strict=True)
    selected = {}
    for relative in (*DEFAULT_SOURCE_FILES, *extra):
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError(f'Source paths must be repository-relative: {relative}')
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Source must be a regular file inside the repository: {relative}')
        selected[name.as_posix()] = path
    return dict(sorted(selected.items()))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def command(args):
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{args[0]} failed ({result.returncode}): {result.stderr}')
    return {'command': args, 'stdout': result.stdout}


ARMS = frozenset(('bf16', 'ptq', 'qad', 'continued_qad', 'qad_opd'))


def validate_final_manifest(path):
    """Validate the explicit completed-run boundary used for publication provenance."""
    path = Path(path).resolve(strict=True)
    if path.name != 'final_manifest.json' or path.is_symlink():
        raise ValueError('--final-manifest must name a regular final_manifest.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not re.fullmatch(r'high_fp4_[a-z0-9_]+_final_manifest', str(data.get('format', ''))):
        raise ValueError('unsupported or incomplete final manifest format')
    if data.get('selection_uses_heldout') is not False:
        raise ValueError('final selection must be independent of heldout results')
    if set(data.get('required_arms', ())) != ARMS:
        raise ValueError('final manifest does not declare the complete five-arm comparison')
    comparison = Path(data.get('heldout_comparison', {}).get('path', ''))
    if not comparison.is_file():
        raise ValueError('final manifest heldout comparison is missing')
    declared = data['heldout_comparison']
    if declared.get('sha256') != digest(comparison) or declared.get('bytes') != comparison.stat().st_size:
        raise ValueError('final manifest heldout comparison identity differs')
    paired = json.loads(comparison.read_text(encoding='utf-8'))
    if paired.get('environment_pairing_verified') is not True or paired.get('protocol_consistency_verified') is not True:
        raise ValueError('final heldout comparison has not passed pairing/protocol checks')
    if set(paired.get('arms', ())) != ARMS:
        raise ValueError('final heldout comparison does not contain exactly five arms')
    state = path.parent / 'run_manifest.json'
    if state.is_file() and json.loads(state.read_text(encoding='utf-8')).get('status') != 'complete':
        raise ValueError('run_manifest is not complete')
    return path, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gr00t', type=Path, default=os.environ.get('GR00T_REPO', ROOT / 'third_party/Isaac-GR00T'))
    parser.add_argument('--server-python')
    parser.add_argument('--rollout-python')
    parser.add_argument('--checkpoint', action='append', required=True, metavar='ARM=PATH')
    boundary = parser.add_mutually_exclusive_group(required=True)
    boundary.add_argument('--final-manifest', type=Path,
                          help='Completed run final_manifest.json; required for publication provenance')
    boundary.add_argument('--run-dir', type=Path,
                          help='Completed run directory containing final_manifest.json')
    parser.add_argument('--source-file', action='append', default=[], metavar='RELATIVE_PATH',
                        help='Also hash this published repository file; repeat as needed')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a new provenance output directory')
    final_path = args.final_manifest or (args.run_dir / 'final_manifest.json')
    final_path, final = validate_final_manifest(final_path)
    source_files = resolve_source_files(ROOT, args.source_file)
    groot = args.gr00t.resolve()
    python = args.server_python or str(groot / '.venv/bin/python')
    simulation = args.rollout_python or str(groot / '.venv-libero/bin/python')
    code = 'import importlib.metadata as m; print("\\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions())))'
    records = {'platform': command(['uname', '-a']),
               'gpu': command(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv']),
               'project_head': command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
               'gr00t_head': command(['git', '-C', str(groot), 'rev-parse', 'HEAD']),
               'training_packages': command([python, '-c', code]),
               'rollout_packages': command([simulation, '-c', code]),
               'source_files': {}, 'checkpoints': {},
               'final_manifest': {'path': str(final_path), 'sha256': digest(final_path),
                                 'bytes': final_path.stat().st_size,
                                 'format': final['format'], 'required_arms': sorted(ARMS)}}
    records['source_selection'] = {
        'policy': 'explicit_public_allowlist_v1',
        'additional_files': args.source_file,
        'note': 'Every source_files path must be included in the public repository; external source is restored from pinned upstream plus patches.'}
    for relative, path in source_files.items():
        records['source_files'][relative] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    for argument in args.checkpoint:
        arm, separator, raw_path = argument.partition('=')
        if not separator or not arm or arm in records['checkpoints']:
            raise ValueError('Provide unique ARM=PATH checkpoint arguments')
        folder = Path(raw_path).resolve(strict=True)
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.safetensors', '.json', '.yaml'))
        if not any(p.suffix == '.safetensors' for p in files):
            raise ValueError(f'No checkpoint weights: {folder}')
        records['checkpoints'][arm] = {'path': str(folder), 'files': [
            {'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files]}
        print(f'[provenance] verified {arm}: {len(files)} files', flush=True)
    args.out.mkdir(parents=True)
    (args.out / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(f'[provenance] saved {args.out / "manifest.json"}')


if __name__ == '__main__':
    main()
