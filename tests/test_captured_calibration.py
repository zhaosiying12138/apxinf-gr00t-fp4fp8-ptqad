"""CPU contracts for replaying audited teacher inputs into PTQ collectors."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from quant.ptq import captured_calibration as captured
from quant.ptq.collector import accumulate


@contextmanager
def capture_fixture(tmp_path):
    teacher = tmp_path / "teacher"
    checkpoint = tmp_path / "calibration_model"
    observations = tmp_path / "evaluation" / "observations"
    teacher.mkdir()
    checkpoint.mkdir()
    observations.mkdir(parents=True)
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"fixture": "training-only"}\n')
    for name in ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json"):
        content = json.dumps({"fixture_metadata": name}) + "\n"
        (teacher / name).write_text(content)
        (checkpoint / name).write_text(content)

    weights = {}
    for index in range(2):
        name = f"model-0000{index + 1}-of-00002.safetensors"
        path = teacher / name
        # Identity validation needs bytes, not a model or a tensor loader.
        path.write_bytes(f"synthetic teacher shard {index}".encode())
        weights[name] = {"bytes": path.stat().st_size, "sha256": captured.file_sha256(path)}

    tasks, originals, paths = {}, [], []
    # Deliberately non-alphabetical: preserve audited task order, not a new sort.
    for task in ("task_z", "task_a"):
        folder = observations / task
        folder.mkdir()
        sources = []
        for episode in (3, 8):
            index = len(paths)
            patch_count = index + 2
            inputs = {
                "pixel_values": (torch.arange(patch_count * 3).reshape(patch_count, 3) - 5 + index).to(torch.bfloat16),
                "image_grid_thw": torch.tensor([[1, 1, patch_count]], dtype=torch.int64),
                "input_ids": torch.tensor([[11, 12, 13 + index]], dtype=torch.int64),
                "attention_mask": torch.ones(1, 3, dtype=torch.int64),
                "embodiment_id": torch.tensor([2], dtype=torch.int64),
                "state": torch.full((1, 1, 132), index, dtype=torch.bfloat16),
                "action": torch.full((1, 40, 132), index + 0.5, dtype=torch.float32),
                "action_mask": torch.zeros(1, 40, 132),
            }
            inputs["action_mask"][:, :16, :7] = 1
            path = folder / f"sample_{len(sources):06d}.pt"
            torch.save({"inputs": inputs, "episode_index": episode}, path)
            originals.append(inputs)
            paths.append(path)
            sources.append({"path": str(path.relative_to(observations)), "sha256": captured.file_sha256(path)})
        tasks[task] = {"samples": 2, "contributing_successful_episodes": 2,
                       "samples_per_episode": {3: 1, 8: 1}, "source_files": sources}
    audit = {"format": "successful_teacher_replay_audit_v1", "status": "verified",
             "evaluation": str(observations.parent), "observations": str(observations),
             "teacher": str(teacher), "teacher_weights": weights,
             "task_count": len(tasks), "sample_count": len(paths), "tasks": tasks}
    calls = []

    def audit_replay(root, protocol_file, source_teacher):
        assert Path(root).resolve() in (observations.parent, observations)
        assert Path(protocol_file).resolve() == protocol
        assert Path(source_teacher).resolve() == teacher
        calls.append(Path(root).resolve())
        result = copy.deepcopy(audit)
        # Stand in for the upstream auditor's current source-byte identities.
        for task in result["tasks"].values():
            for row in task["source_files"]:
                row["sha256"] = captured.file_sha256(observations / row["path"])
        return result

    with patch.object(captured, "audit_replay", audit_replay):
        frozen = captured.freeze_capture(observations, protocol, teacher)
        manifest = tmp_path / "frozen_capture.json"
        manifest.write_text(json.dumps(frozen, indent=2) + "\n")
        yield SimpleNamespace(teacher=teacher, checkpoint=checkpoint, observations=observations,
                              protocol=protocol, audit=audit, frozen=frozen, manifest=manifest,
                              originals=originals, paths=paths, weights=weights, calls=calls)


def load_fixture(fixture, **overrides):
    settings = {"manifest": fixture.manifest, "checkpoint": fixture.checkpoint,
                "windows": len(fixture.paths), "batch": 1}
    settings.update(overrides)
    return captured.CapturedCalibration(**settings)


class CapturedCalibrationTests(unittest.TestCase):
    def setUp(self):
        raw = self.enterContext(tempfile.TemporaryDirectory())
        self.fixture = self.enterContext(capture_fixture(Path(raw)))

    def test_freeze_json_round_trip_accepts_integer_episode_keys(self):
        fixture = self.fixture
        self.assertEqual(fixture.audit["tasks"]["task_z"]["samples_per_episode"], {3: 1, 8: 1})
        round_tripped = json.loads(fixture.manifest.read_text())
        self.assertEqual(fixture.frozen, round_tripped)
        self.assertEqual(round_tripped["source_audit"]["tasks"]["task_z"]["samples_per_episode"], {"3": 1, "8": 1})
        replay = load_fixture(fixture)
        self.assertEqual(fixture.calls, [fixture.observations, fixture.observations.parent])
        self.assertEqual(replay.record["source_audit"], round_tripped["source_audit"])
        self.assertEqual(replay.record["sha256"], captured.file_sha256(fixture.manifest))
        self.assertEqual(replay.record["bytes"], fixture.manifest.stat().st_size)

    def test_iteration_preserves_patch_and_single_sample_axes_once(self):
        fixture = self.fixture
        replay = load_fixture(fixture)
        original_load = torch.load
        loaded = []

        def record_load(path, *args, **kwargs):
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            loaded.append(Path(path))
            return original_load(path, *args, **kwargs)

        before = {path: captured.file_sha256(path) for path in fixture.paths}
        with patch.object(torch, "load", record_load):
            iterator = iter(replay)
            outputs = [next(iterator) for _ in fixture.paths]
            with self.assertRaises(StopIteration):
                next(iterator)
        self.assertEqual(loaded, fixture.paths)
        self.assertEqual(len(set(loaded)), len(fixture.paths))
        for actual, expected in zip(outputs, fixture.originals):
            self.assertEqual(actual.keys(), expected.keys())
            for name in expected:
                self.assertEqual(actual[name].shape, expected[name].shape)
                self.assertEqual(actual[name].dtype, expected[name].dtype)
                self.assertEqual(actual[name].device.type, "cpu")
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
            self.assertEqual(actual["pixel_values"].ndim, 2)
            self.assertEqual(actual["state"].shape, (1, 1, 132))
            self.assertEqual(actual["action"].shape, (1, 40, 132))
        outputs[0]["pixel_values"].zero_()
        self.assertEqual(before, {path: captured.file_sha256(path) for path in fixture.paths})

    def test_captured_inputs_produce_exact_unnormalized_second_moment(self):
        fixture = self.fixture
        entry = {"H": None, "abs": None, "n": 0, "calls": 0}
        for inputs in load_fixture(fixture):
            accumulate(entry, inputs["pixel_values"])
        x = torch.cat([row["pixel_values"].float() for row in fixture.originals])
        torch.testing.assert_close(entry["H"], x.T @ x, rtol=0, atol=0)
        torch.testing.assert_close(entry["abs"], x.abs().sum(0), rtol=0, atol=0)
        self.assertEqual(entry["n"], x.shape[0])
        self.assertEqual(entry["calls"], len(fixture.paths))
        self.assertEqual(entry["H"].dtype, torch.float32)
        self.assertEqual(entry["H"].device.type, "cpu")

    def test_batch_must_be_integer_one(self):
        for batch in (0, 2, True, 1.0, "1"):
            with self.subTest(batch=batch), self.assertRaisesRegex(ValueError, "requires batch=1"):
                load_fixture(self.fixture, batch=batch)

    def test_window_budget_must_consume_the_full_capture_once(self):
        for windows in (0, 3, 5, True, 4.0):
            with self.subTest(windows=windows), self.assertRaisesRegex(ValueError, "all 4 windows exactly once"):
                load_fixture(self.fixture, windows=windows)

    def test_root_weight_identity_accepts_exact_teacher_shards(self):
        load_fixture(self.fixture).validate_root_weights(copy.deepcopy(self.fixture.weights))

    def test_wrong_root_weight_identity_is_rejected(self):
        for mutation in ("missing_shard", "extra_shard", "wrong_bytes", "wrong_sha", "missing_sha"):
            with self.subTest(mutation=mutation):
                weights = copy.deepcopy(self.fixture.weights)
                name = next(iter(weights))
                if mutation == "missing_shard":
                    weights.pop(name)
                elif mutation == "extra_shard":
                    weights["unexpected.safetensors"] = copy.deepcopy(weights[name])
                elif mutation == "wrong_bytes":
                    weights[name]["bytes"] += 1
                elif mutation == "wrong_sha":
                    weights[name]["sha256"] = "0" * 64
                else:
                    weights[name].pop("sha256")
                with self.assertRaisesRegex(ValueError, "Calibration root (shard set|weights) differ"):
                    load_fixture(self.fixture).validate_root_weights(weights)

    def test_calibration_model_metadata_must_match_teacher(self):
        for name in ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json"):
            with self.subTest(name=name):
                path = self.fixture.checkpoint / name
                before = path.read_bytes()
                path.write_text('{"different": true}\n')
                with self.assertRaisesRegex(ValueError, "model metadata differ: " + name):
                    load_fixture(self.fixture)
                path.write_bytes(before)

    def test_mutating_both_metadata_copies_cannot_change_frozen_teacher(self):
        for name in ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json"):
            with self.subTest(name=name):
                paths = [self.fixture.teacher / name, self.fixture.checkpoint / name]
                before = [path.read_bytes() for path in paths]
                for path in paths:
                    path.write_text('{"changed_in_both_directories": true}\n')
                with self.assertRaisesRegex(ValueError, "no longer matches its source audit"):
                    load_fixture(self.fixture)
                for path, original in zip(paths, before):
                    path.write_bytes(original)

    def test_changed_audit_is_rejected_after_freezing(self):
        self.fixture.audit["tasks"]["task_z"]["samples_per_episode"] = {3: 2, 8: 0}
        with self.assertRaisesRegex(ValueError, "no longer matches its source audit"):
            load_fixture(self.fixture)

    def test_changed_sample_is_rejected_by_fresh_audit(self):
        self.fixture.paths[0].write_bytes(b"changed capture")
        with self.assertRaisesRegex(ValueError, "no longer matches its source audit"):
            load_fixture(self.fixture)

    def test_changed_sample_is_rejected_between_validation_and_iteration(self):
        iterator = iter(load_fixture(self.fixture))
        next(iterator)
        self.fixture.paths[1].write_bytes(b"changed after the first window")
        with self.assertRaisesRegex(ValueError, "Captured input changed after validation"):
            next(iterator)

    def test_duplicate_or_missing_audited_windows_are_rejected(self):
        fixture = self.fixture
        original_rows = copy.deepcopy(fixture.audit["tasks"]["task_a"]["source_files"])
        for mutation in ("duplicate", "missing"):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(original_rows)
                if mutation == "duplicate":
                    rows[1] = copy.deepcopy(rows[0])
                else:
                    rows.pop()
                fixture.audit["tasks"]["task_a"]["source_files"] = rows
                fixture.manifest.write_text(json.dumps(captured.freeze_capture(
                    fixture.observations, fixture.protocol, fixture.teacher)))
                with self.assertRaisesRegex(ValueError, "duplicate or missing windows"):
                    load_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
