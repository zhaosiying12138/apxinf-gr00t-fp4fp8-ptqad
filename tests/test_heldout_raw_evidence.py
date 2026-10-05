"""CPU-only raw publication evidence tests; all rollouts are synthetic fixtures."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from paper import materialize_final_evidence as materialize
from paper import validate_publication as validation
from eval.run_recovery_eval import TASKS, parse_log

ROOT = Path(__file__).resolve().parents[1]


class HeldoutRawEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='synthetic-heldout-raw-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.origin = self.root / 'original'
        self.paper = self.root / 'publication' / 'paper'
        self.output = self.paper / 'evidence'
        self.protocol = self.root / 'protocol.json'
        self.protocol.write_bytes((ROOT / 'exp/recovery_protocol_v11_w4a4_category.json').read_bytes())
        heldout = json.loads(self.protocol.read_text())['partitions']['heldout']
        indices, seed = heldout['init_state_indices'], heldout['seed']
        self.task = TASKS[0]
        for arm in materialize.collect_pairing_evidence.ARMS:
            folder = self.origin / ('heldout_' + arm)
            folder.mkdir(parents=True)
            manifest = {'tasks': TASKS, 'seed': seed, 'episodes': len(indices),
                        'init_state_indices': indices}
            results = {}
            for ti, task in enumerate(TASKS):
                task_seed = seed + 1000 * ti
                resets = [{'episode_index': i, 'seed': task_seed + i,
                           'init_state_index': bank, 'settle_steps': 10,
                           'initial_state_sha256': f'{ti * 100 + i:064x}',
                           'restored_state_sha256': f'{ti * 100 + i + 3000:064x}',
                           'init_state_bank_sha256': f'{ti + 7000:064x}'}
                          for i, bank in enumerate(indices)]
                # Include the producer's terminal auto-reset: it must be
                # preserved and accounted for without becoming a scored reset.
                resets.append({**resets[-1], 'episode_index': len(indices)})
                text = ''.join('FP4VLA_EPISODE_RESET ' + json.dumps(row) + '\n' for row in resets)
                outcomes = [True] * (len(indices) - 1) + [False]
                text += f"results: ('libero_sim/{task}', {outcomes})\n"
                log = folder / (task + '.log')
                log.write_bytes(b'synthetic fixture only\r\n' + text.encode())
                (folder / (task + '.server.log')).write_bytes(b'synthetic server fixture\r\n\xff\n')
                results[task] = {**parse_log(log), 'returncode': 0, 'seed': task_seed}
            (folder / 'eval_manifest.json').write_text(json.dumps(manifest))
            (folder / 'task_results.json').write_text(json.dumps(results))
            destination = self.output / folder.name
            destination.mkdir(parents=True)
            for name in ('eval_manifest.json', 'task_results.json'):
                shutil.copyfile(folder / name, destination / name)
        parser_copy = self.paper.parent / 'eval/run_recovery_eval.py'
        parser_copy.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'eval/run_recovery_eval.py', parser_copy)

    def materialize(self):
        self.mapping = []
        return materialize._materialize_heldout_raw_logs(
            self.origin, self.output, self.protocol, self.mapping)

    def validate(self):
        inputs = set()
        with patch.object(validation, 'P', self.paper):
            validation.check_heldout_raw_logs(inputs, self.protocol)
        return inputs

    def mutate_result(self, folder, field, value):
        path = folder / 'task_results.json'
        data = json.loads(path.read_text())
        data[self.task][field] = value
        path.write_text(json.dumps(data))

    def test_all_one_hundred_logs_are_byte_identical_and_validate_without_private_run(self):
        receipt = self.materialize()
        self.assertEqual(receipt['path'], 'evidence/heldout_raw_logs.json')
        self.assertEqual(len(self.mapping), 100)
        self.assertEqual(sum(row['role'] == 'verified_heldout_server_log' for row in self.mapping), 50)
        for row in self.mapping:
            original = Path(row['source']['path'])
            self.assertEqual(original.read_bytes(), Path(row['published_path']).read_bytes())
        shutil.rmtree(self.origin)
        self.assertEqual(len(self.validate()), 102)  # 100 logs, parser source, log manifest

    def test_missing_source_rollout_or_server_is_rejected_before_log_copy(self):
        for suffix in ('.log', '.server.log'):
            path = self.origin / 'heldout_qad' / (self.task + suffix)
            payload = path.read_bytes()
            path.unlink()
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'Missing nonempty raw'):
                self.materialize()
            self.assertEqual(list(self.output.glob('heldout_*/*.log')), [])
            path.write_bytes(payload)

    def test_changed_source_rollout_fails_original_task_hash(self):
        path = self.origin / 'heldout_qad' / (self.task + '.log')
        path.write_bytes(path.read_bytes() + b'changed fixture\n')
        with self.assertRaisesRegex(ValueError, 'log hash differs'):
            self.materialize()

    def test_claimed_task_success_count_must_match_actual_parser(self):
        self.mutate_result(self.origin / 'heldout_qad', 'successes', 0)
        with self.assertRaisesRegex(ValueError, 'parse differs from task receipt'):
            self.materialize()

    def test_matching_hash_and_task_json_cannot_hide_wrong_reset_seed(self):
        folder = self.origin / 'heldout_qad'
        manifest = json.loads((folder / 'eval_manifest.json').read_text())
        path = folder / (self.task + '.log')
        path.write_text(path.read_text().replace('"seed": ' + str(manifest['seed']),
                                                '"seed": ' + str(manifest['seed'] + 99), 1))
        results = json.loads((folder / 'task_results.json').read_text())
        results[self.task].update(parse_log(path))
        (folder / 'task_results.json').write_text(json.dumps(results))
        with self.assertRaisesRegex(ValueError, 'reset differs from declared'):
            self.materialize()

    def test_missing_published_log_is_rejected(self):
        self.materialize()
        (self.output / 'heldout_ptq' / (self.task + '.server.log')).unlink()
        with self.assertRaisesRegex(RuntimeError, 'Missing evidence'):
            self.validate()

    def test_changed_published_server_log_is_rejected_by_hash(self):
        self.materialize()
        path = self.output / 'heldout_ptq' / (self.task + '.server.log')
        payload = path.read_bytes()
        path.write_bytes(b'X' + payload[1:])
        with self.assertRaisesRegex(RuntimeError, 'Changed evidence'):
            self.validate()

    def test_published_task_outcomes_must_match_raw_log(self):
        self.materialize()
        self.mutate_result(self.output / 'heldout_qad', 'results', [False] * 16)
        with self.assertRaisesRegex(ValueError, 'parse differs from task receipt'):
            self.validate()

    def test_published_manifest_cannot_omit_an_official_log(self):
        self.materialize()
        path = self.output / 'heldout_raw_logs.json'
        data = json.loads(path.read_text())
        data['files'].pop(next(iter(data['files'])))
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(RuntimeError, 'exactly fifty rollout/server'):
            self.validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
