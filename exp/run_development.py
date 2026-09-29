#!/usr/bin/env python3
"""Evaluate the declared PTQ prefix serially; stop immediately after selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval'))
from compare_development import ORDER, audit_development
from run_recovery_eval import TASKS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    value = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            'File changed while hashing: ' + str(path))
    return {'path': str(path), 'bytes': before.st_size, 'sha256': value}


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def checkpoint_identity(folder):
    folder = Path(folder).resolve(strict=True)
    index = load(folder / 'model.safetensors.index.json')
    shards = sorted(set(index['weight_map'].values()))
    require(shards and all(Path(name).name == name and name.endswith('.safetensors') for name in shards),
            'Invalid checkpoint shard index: ' + str(folder))
    require({p.name for p in folder.glob('*.safetensors')} == set(shards), 'Shard set differs from index')
    metadata = ['config.json', 'statistics.json', 'processor_config.json', 'embodiment_id.json',
                'model.safetensors.index.json']
    return {'path': str(folder), 'weights': {n: identity(folder / n) for n in shards},
            'metadata': {n: identity(folder / n) for n in metadata}}


def validate_bake(folder, recipe, base):
    folder = Path(folder)
    require(folder.is_dir(), 'Missing recipe directory: ' + str(folder))
    require(not (folder / 'merge_manifest.json').exists(), 'Development reference must be pure PTQ')
    record = load(folder / 'bake_manifest.json')
    allocation = load(folder / 'ptq_recipe.json')
    require(record.get('status') == 'complete', 'Incomplete/failed bake cannot be reused')
    require(allocation['recipe'] == recipe, 'Recipe name differs from checkpoint metadata')
    require(Path(record['base']).resolve() == Path(base['path']) and
            Path(allocation['base']).resolve() == Path(base['path']), 'Bake uses another base')
    require(set(record['source_weight_files']) == set(base['weights']), 'Bake source shard set differs')
    for name, row in base['weights'].items():
        require(all(record['source_weight_files'][name][k] == row[k] for k in ('bytes', 'sha256')),
                'Bake source weight identity differs')
    calibration = (allocation.get('calibration_provenance') or {}).get('metadata', {})
    if recipe != 'fp8':
        require(allocation.get('calibration_mode') == 'required' and calibration.get('status') == 'complete' and
                calibration.get('windows_consumed') == 128 and calibration.get('windows_requested') == 128 and
                calibration.get('recipe_targets') == 'calib',
                'Reused promoted bake requires complete required full-scope 128-window calibration')
    for field, filename in [('base_config_sha256', 'config.json'), ('base_statistics_sha256', 'statistics.json')]:
        # New recipes record these in calibration provenance only when calibrated.
        value = (allocation.get('calibration_provenance') or {}).get('metadata', {})
        if recipe != 'fp8':
            require(field in value, 'Calibrated recipe lacks base metadata identity')
        if field in value:
            require(value[field] == base['metadata'][filename]['sha256'], 'Calibration base metadata differs')
    actual = checkpoint_identity(folder)
    for name in ('config.json', 'statistics.json', 'processor_config.json', 'embodiment_id.json'):
        require(actual['metadata'][name]['sha256'] == base['metadata'][name]['sha256'],
                'PTQ changed base metadata: ' + name)
    require(set(record['output_weight_files']) == set(actual['weights']), 'Bake output shard set differs')
    for name, row in actual['weights'].items():
        require(all(record['output_weight_files'][name][k] == row[k] for k in ('bytes', 'sha256')),
                'Completed bake bytes changed: ' + name)
    actual['bake_manifest'] = identity(folder / 'bake_manifest.json')
    actual['ptq_recipe'] = identity(folder / 'ptq_recipe.json')
    return actual


def run_logged(command, log, cwd, env):
    with Path(log).open('x') as stream:
        stream.write(json.dumps({'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()}) + '\n')
        stream.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        stream.write('COMPLETED UTC ' + datetime.now(timezone.utc).isoformat() + '\n')


def verify_result(folder, checkpoint):
    folder = Path(folder)
    manifest, rows, summary = (load(folder / name) for name in ['eval_manifest.json', 'task_results.json', 'summary.json'])
    require(Path(manifest['checkpoint']).resolve() == Path(checkpoint), 'Evaluated another checkpoint')
    require(manifest['protocol_sha256'] == sha(ROOT / 'exp/recovery_protocol.json'), 'Changed development protocol')
    require(set(rows) == set(TASKS) and summary['tasks_complete'] == 10, 'Incomplete development task set')
    successes = 0
    for task in TASKS:
        row = rows[task]
        require(row['returncode'] == 0 and len(row['results']) == 2 and all(type(v) is bool for v in row['results']),
                'Invalid/partial development result: ' + task)
        require(row['episodes'] == 2 and row['successes'] == sum(row['results']), 'Result counts differ')
        require(sha(folder / (task + '.log')) == row['log_sha256'], 'Raw task log identity differs')
        successes += row['successes']
    require(summary['total_successes'] == successes and summary['total_episodes'] == 20 and
            abs(summary['macro_success_rate'] - successes / 20) < 1e-12, 'Summary disagrees with raw outcomes')


def execute(args):
    run = Path(args.run_dir).resolve()
    development = Path(args.development_root).resolve()
    require(not development.exists(), 'Use a new development root; failed or completed output is never reused')
    require(not any(run.glob('train_*')), 'Development selection must finish before recovery training starts')
    env = os.environ.copy()
    env.update(PTQAD_RUN_DIR=str(run), PTQAD_BASE=str(Path(args.base).resolve()),
               GR00T_REPO=str(Path(args.gr00t).resolve()), PTQAD_PYTHON=args.server_python,
               LIBERO_PYTHON=args.rollout_python, PTQAD_RECOVERY_RECIPE='fp8')
    commands = []
    for arm in ORDER:
        checkpoint = str(Path(args.base).resolve()) if arm == 'bf16' else str(run / arm)
        command = [sys.executable, str(ROOT / 'eval/run_recovery_eval.py'), '--checkpoint', checkpoint,
                   '--out', str(development / arm), '--purpose', 'development', '--seed', '330000',
                   '--episodes', '2', '--gr00t', args.gr00t, '--server-python', args.server_python,
                   '--rollout-python', args.rollout_python, '--port', str(args.port)]
        commands.append({'arm': arm, 'checkpoint': checkpoint, 'evaluation': command,
                         'bake_if_missing': None if arm == 'bf16' else ['bash', str(ROOT / 'exp/reproduce_ptqad.sh'), arm]})
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'resource_identities_verified': False, 'development_root': str(development),
                          'commands_in_possible_order': commands,
                          'stop_rule': 'After each complete arm, audit the prefix; stop at first selected_recipe != null.',
                          'calibration_requirement': str(run / 'calibration/calib.pt')}, indent=2))
        return
    require(Path(args.server_python).is_file() and os.access(args.server_python, os.X_OK), 'Missing server Python')
    require(Path(args.rollout_python).is_file() and os.access(args.rollout_python, os.X_OK), 'Missing rollout Python')
    base = checkpoint_identity(args.base)
    development.mkdir(parents=True)
    inputs = ['exp/run_development.py', 'exp/run_development.sh', 'exp/reproduce_ptqad.sh',
              'eval/compare_development.py', 'eval/run_recovery_eval.py', 'exp/recovery_protocol.json']
    write_new(development / 'development_run.json', {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'run_dir': str(run), 'development_root': str(development), 'base_identity': base,
              'source_files': {n: identity(ROOT / n) for n in inputs}, 'possible_order': ORDER,
              'heldout_used_for_selection': False})
    completed = []
    for job in commands:
        arm, folder = job['arm'], Path(job['checkpoint'])
        if arm == 'bf16':
            current = base
        else:
            if not folder.exists():
                require(not (run / 'logs' / ('bake-' + arm + '.log')).exists(), 'A prior bake log exists without a completed artifact; use a new run directory')
                if arm != 'fp8':
                    meta = load(run / 'calibration/calib_meta.json')
                    require((run / 'calibration/calib.pt').is_file() and meta['status'] == 'complete' and
                            meta['windows_consumed'] == 128 and meta['recipe_targets'] == 'calib',
                            'Promoted recipes require completed full-scope 128-window calibration')
                run_logged(job['bake_if_missing'], development / (arm + '.bake.log'), ROOT, env)
            current = validate_bake(folder, arm, base)
        write_new(development / (arm + '.checkpoint.json'), current)
        print('[development] evaluating ' + arm + ': ' + str(folder), flush=True)
        run_logged(job['evaluation'], development / (arm + '.driver.log'), ROOT, env)
        verify_result(development / arm, folder)
        completed.append(arm)
        report = audit_development(development, completed)
        write_new(development / ('comparison_after_' + arm + '.json'), report)
        print(json.dumps({'completed': completed, 'selected_recipe': report['selected_recipe']}), flush=True)
        if report['selected_recipe'] is not None:
            write_new(development / 'selection.json', report)
            print('[development] STOP; selected=' + report['selected_recipe'] + '; selection=' + str(development / 'selection.json'))
            return
    raise RuntimeError('Declared ladder exhausted without a selected recipe')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', default=os.environ.get('PTQAD_RUN_DIR'))
    p.add_argument('--development-root')
    p.add_argument('--base', default=os.environ.get('PTQAD_BASE', str(ROOT / 'weights/GR00T-N1.7-LIBERO/libero_10')))
    p.add_argument('--gr00t', default=os.environ.get('GR00T_REPO', str(ROOT / 'third_party/Isaac-GR00T')))
    p.add_argument('--server-python', default=os.environ.get('PTQAD_PYTHON'))
    p.add_argument('--rollout-python', default=os.environ.get('LIBERO_PYTHON'))
    p.add_argument('--port', type=int, default=int(os.environ.get('PTQAD_PORT_BASE', '5610')))
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    require(args.run_dir, 'Set PTQAD_RUN_DIR or --run-dir')
    args.development_root = args.development_root or str(Path(os.environ.get('PTQAD_EVAL_RUNS_ROOT', str(Path(args.run_dir) / 'evaluations'))) / 'development')
    args.server_python = args.server_python or str(Path(args.gr00t) / '.venv/bin/python')
    args.rollout_python = args.rollout_python or str(Path(args.gr00t) / '.venv-libero/bin/python')
    require(1 <= args.port <= 65535, 'Port must be in 1..65535')
    execute(args)


if __name__ == '__main__':
    main()
