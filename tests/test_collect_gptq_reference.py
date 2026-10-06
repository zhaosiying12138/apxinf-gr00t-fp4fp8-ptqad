"""Synthetic ten-task raw logs; never read real results or construct a model.

Only collect's expensive live comparison is mocked. Archive parsing, source
snapshot loading, reset/activation validation and paired statistics are real.
The synthetic plan keeps the production contracts, with a smaller bootstrap
budget and its own explicitly bound comparison-source PLAN_SHA256 constant.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from eval import compare_gptq_reference as live
from eval.run_recovery_eval import TASKS, parse_log
from paper import collect_gptq_reference as archive

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def rid(path):
    return archive.record(path)


class Fixture:
    def __init__(self, root):
        self.root = root
        self.private = root / "private"
        self.main = root / "publication/evidence"
        self.output = self.main / "gptq_reference"
        self.protocol = self.private / "exp/recovery_protocol_v12_rtn_w4a4.json"
        self.protocol.parent.mkdir(parents=True)
        self.protocol.write_bytes((ROOT / "exp/recovery_protocol_v12_rtn_w4a4.json").read_bytes())
        self.data = archive.read(self.protocol)
        plan = archive.read(ROOT / "exp/gptq_reference_protocol_v12.json")
        for row in plan["preparation_source_files"].values():
            target = self.private / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / row["path"]).read_bytes())
        prior = self.private / plan["amendment"]["previous_protocol_path"]
        prior.parent.mkdir(parents=True)
        prior.write_bytes((ROOT / plan["amendment"]["previous_protocol_path"]).read_bytes())
        for name in ("eval/compare_gptq_reference.py", "eval/gptq_reference_evidence.py",
                     "exp/action_chunk_diagnostics.py", "exp/bake_gptq_reference_category.py",
                     "paper/activation_evidence.py", "paper/publication_guard.py",
                     "paper/collect_training_costs.py"):
            target = self.private / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.models = {}
        for name in (*archive.ARMS, "gptq", "ordinary"):
            folder = self.private / "models" / name
            metadata = {}
            for filename in ("config.json", "statistics.json", "processor_config.json", "embodiment_id.json",
                             "model.safetensors.index.json"):
                write(folder / filename, {"fixture": filename})
                metadata[filename] = rid(folder / filename)
            self.models[name] = {"path": str(folder), "metadata": metadata,
                "weights": {"model.safetensors": {"path": str(folder / "model.safetensors"),
                             "bytes": 100, "sha256": "a" * 64}}}
        ordinary = Path(self.models["ordinary"]["path"])
        for name, value in (("ptq_recipe.json", {"actual_method_tensor_counts": {"nvfp4_gptq": 469}}),
                            ("bake_manifest.json", {"elapsed_seconds": 3.25})):
            write(ordinary / name, value)
        # Use the five-field checkpoint_identity schema. Collection must copy
        # ordinary provenance even without validate_parent's two extra entries.
        category = Path(self.models["gptq"]["path"])
        write(category / "category_ptq_recipe.json", {
            "parent_recipe_sha256": rid(ordinary / "ptq_recipe.json")["sha256"],
            "parent_bake_manifest_sha256": rid(ordinary / "bake_manifest.json")["sha256"]})
        write(category / "category_bake_manifest.json", {
            "parent_recipe": rid(ordinary / "ptq_recipe.json"),
            "parent_bake_manifest": rid(ordinary / "bake_manifest.json")})
        self.collection = self.private / "run/collection"
        self.collection_audit = self.evaluation(self.collection, "qad", "collection")
        self.round = self.private / "run/heldout_round"
        audits, episodes = {}, {}
        for arm in archive.ARMS:
            folder = self.round / ("heldout_" + arm)
            audits[arm] = self.evaluation(folder, arm)
            episodes[arm] = live.read_episodes(folder)
        gptq_folder = self.private / "supplement/heldout_gptq"
        audits["gptq"] = self.evaluation(gptq_folder, "gptq")
        episodes["gptq"] = live.read_episodes(gptq_folder)
        write(self.round / "paired_comparison.json", live.compare_round(self.round))
        selection = self.private / "run/selection.json"
        write(selection, {"protocol_sha256": rid(self.protocol)["sha256"], "fixture": "development only"})
        self.final = self.private / "run/final_manifest.json"
        write(self.final, {"format": "w4a4_recovery_v12_final_manifest", "selection_uses_heldout": False,
                          "required_arms": list(archive.ARMS), "protocol_sha256": rid(self.protocol)["sha256"],
                          "selection_file": str(selection), "selection_sha256": rid(selection)["sha256"],
                          "selected_qad_model_identity": {"path": self.models["qad"]["path"]}})
        state = self.private / "run/run_manifest.json"
        write(state, {"status": "complete"})
        teacher_folder = self.private / "teacher"
        self.evaluation(teacher_folder, "bf16", "teacher_supervision")
        capture_tasks = {}
        for task in TASKS:
            cp = teacher_folder / "observations" / task / "capture_manifest.json"
            reset_log = cp.with_name("reset_events.jsonl")
            reset_log.parent.mkdir(parents=True)
            reset_log.write_text(json.dumps({"fixture": "teacher reset receipt", "task": task}) + "\n")
            write(cp, {"task_name": task, "finalized": True, "reset_event_file_sha256": rid(reset_log)["sha256"]})
            capture_tasks[task] = {"capture_manifest_sha256": rid(cp)["sha256"],
                                  "raw_log_sha256": rid(teacher_folder / (task + ".log"))["sha256"]}
        capture = self.private / plan["capture_manifest"]["path"]
        write(capture, {"source_audit": {"evaluation": str(teacher_folder),
            "observations": str(teacher_folder / "observations"), "tasks": capture_tasks}})
        plan["capture_manifest"] = {"path": plan["capture_manifest"]["path"], **archive.identity(capture)}
        plan["comparison"]["confidence_interval"]["replicates"] = 200
        self.plan = self.private / "exp/gptq_reference_protocol_v12.json"
        write(self.plan, plan)
        comparator = self.private / "eval/compare_gptq_reference.py"
        comparator.write_text(comparator.read_text().replace(live.PLAN_SHA256, rid(self.plan)["sha256"]), encoding="utf-8")
        calibration = {}
        for kind, elapsed in (("ordinary", 1.5), ("category", 2.75)):
            path = self.private / f"supplement/{kind}_h/calib_meta.json"
            write(path, {"status": "complete", "elapsed_seconds": elapsed, "cache_sha256": "b" * 64})
            calibration[kind] = {"metadata": rid(path), "metadata_content": archive.read(path),
                                 "cache": {"path": str(path.with_name("calib.pt")), "bytes": 100,
                                           "sha256": "b" * 64}}
        log = self.private / "supplement/category_bake.log"
        log.write_text("Synthetic category bake only\n")
        invocation = self.private / "supplement/category_bake_invocation.json"
        write(invocation, {"status": "complete", "returncode": 0, "command": ["fixture", "--gptq-damp", "0.01"],
            "started_utc": "2026-10-07T00:00:00+00:00", "finished_utc": "2026-10-07T00:00:04.500000+00:00"})
        frozen = {r["path"]: rid(self.private / r["path"]) for r in (
            plan["main_protocol"], plan["capture_manifest"], *plan["preparation_source_files"].values())}
        analysis = {name: rid(self.private / name) for name in ("eval/compare_gptq_reference.py",
            "eval/gptq_reference_evidence.py", "exp/action_chunk_diagnostics.py", "exp/bake_gptq_reference_category.py")}
        self.report = {"format": "w4a4_gptq_supplement_comparison_v1", "status": "verified",
            "protocol": rid(self.plan), "final_manifest": rid(self.final), "main_run_manifest": rid(state),
            "main_comparison": rid(self.round / "paired_comparison.json"), "frozen_sources": frozen,
            "analysis_sources": analysis, "raw_evaluation_audits": audits,
            "collection_audit": self.collection_audit,
            "selected_checkpoint_sources": {arm: {str(Path(value["path"]) / "config.json"):
                archive.identity(Path(value["path"]) / "config.json")} for arm, value in self.models.items() if arm in archive.ARMS},
            "gptq_evidence": {"checkpoint": self.models["gptq"]["path"],
                "root_identity": self.models["bf16"], "parent_identity": self.models["ordinary"],
                "weight_identity": self.models["gptq"], "calibration_provenance": calibration,
                "category_bake_invocation": {"receipt": rid(invocation), "log": rid(log),
                                             "command": archive.read(invocation)["command"]},
                "actual_method_counts": {"ordinary": {"nvfp4_gptq": 469}}, "memory": {"eligible_tensor_count": 479}},
            "paired_scope": "Official environment resets; policy noise is seeded per task, not paired per episode.",
            "episodes": episodes, **live.summarize(episodes, plan)}
        self.report_path = self.private / "supplement/comparison.json"
        write(self.report_path, self.report)
        self.main.mkdir(parents=True)
        shutil.copyfile(self.final, self.main / "final_manifest.json")
        shutil.copyfile(self.round / "paired_comparison.json", self.main / "paired_comparison.json")
        for arm in archive.ARMS:
            shutil.copytree(self.round / ("heldout_" + arm), self.main / ("heldout_" + arm))

    def evaluation(self, folder, arm, purpose="heldout"):
        folder.mkdir(parents=True)
        part = self.data["partitions"][purpose]
        quantized, adapter = arm != "bf16", arm in ("qad", "continued_qad", "qad_opd")
        manifest = {**self.data["evaluation_contract"], "tasks": TASKS, "seed": part["seed"],
            "episodes": part["episodes_per_task"], "init_state_indices": part["init_state_indices"],
            "purpose": purpose, "protocol_file": str(self.protocol), "protocol_sha256": rid(self.protocol)["sha256"],
            "checkpoint": self.models[arm]["path"], "collection_manifest": str(self.collection / "eval_manifest.json"),
            "environment_summary": {"variables": {"FP4VLA_QUANT": "0", "FP4VLA_W4A4": str(int(quantized)),
                "FP4VLA_W4A4_ADAPTER": str(int(adapter)), "FP4VLA_SATURATE_F16_ACTIVATIONS": str(int(quantized))}}}
        results, pairs, raw, servers = {}, [], {}, []
        successes = 0
        for ti, task in enumerate(TASKS):
            seed = part["seed"] + 1000 * ti
            resets = [{"episode_index": i, "seed": seed + i, "init_state_index": bank, "settle_steps": 10,
                "initial_state_sha256": f"{ti * 100 + i:064x}", "restored_state_sha256": f"{ti * 100 + i + 3000:064x}",
                "init_state_bank_sha256": f"{ti + 7000:064x}"} for i, bank in enumerate(part["init_state_indices"])]
            pairs.extend({"task": task, **r} for r in resets)
            # Preserve and exclude the simulator's terminal automatic reset.
            text = "synthetic fixture only\n" + "".join("FP4VLA_EPISODE_RESET " + json.dumps(r) + "\n"
                for r in [*resets, {**resets[-1], "episode_index": len(resets)}])
            offset = (*archive.ARMS, "gptq").index(arm)
            outcomes = [(ti + i + offset) % 4 != 0 for i in range(len(resets))]
            text += f"results: ('libero_sim/{task}', {outcomes})\n"
            log = folder / (task + ".log")
            log.write_text(text, encoding="utf-8")
            results[task] = {**parse_log(log), "returncode": 0, "seed": seed}
            raw[task] = rid(log)
            successes += sum(outcomes)
            server = folder / (task + ".server.log")
            text = "synthetic BF16 server ready\n"
            if quantized:
                report = {"mode": "all", "ste": False, "ordinary_total": 469, "ordinary_w4a4": 469,
                    "ordinary_bf16": 0, "category_total": 7, "category_w4a4": 7, "category_bf16": 0,
                    "records": [{"name": str(i), "format": "W4A4"} for i in range(476)]}
                text = "[fp4vla] activation-only " + json.dumps(report) + "\n"
            server.write_text(text, encoding="utf-8")
            servers.append({"path": str(server), "sha256": rid(server)["sha256"]})
        count = len(TASKS) * part["episodes_per_task"]
        macro = sum(row["success_rate"] for row in results.values()) / len(TASKS)
        write(folder / "eval_manifest.json", manifest)
        write(folder / "task_results.json", results)
        write(folder / "summary.json", {"tasks_complete": 10, "total_successes": successes, "total_episodes": count,
            "macro_success_rate": macro, "micro_success_rate": successes / count, "purpose": purpose, "seed": part["seed"]})
        return {"evaluation_identity": {name: rid(folder / name) for name in archive.JSON_NAMES},
            "raw_log_identities": raw, "activation_installation": {"server_logs": servers},
            "pairing_sha256": hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest(),
            "successes": successes, "episodes": count, "macro_success_rate": macro, "micro_success_rate": successes / count}


class CollectGPTQReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="synthetic-gptq-archive-")
        cls.fixture = Fixture(Path(cls.temporary.name))
        with patch.object(archive, "ROOT", cls.fixture.private), \
             patch.object(live, "compare", return_value=cls.fixture.report) as gate:
            cls.collected = archive.collect(cls.fixture.report_path, cls.fixture.output, cls.fixture.main)
        if gate.call_count != 1:
            raise AssertionError("Collection must perform exactly one live verification")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="relocated-gptq-archive-")
        self.addCleanup(temporary.cleanup)
        self.main = Path(temporary.name) / "evidence"
        shutil.copytree(self.fixture.main, self.main)
        self.folder = self.main / "gptq_reference"

    def verify(self):
        return archive.verify(self.folder, self.main)

    def rehash(self, relative):
        path = self.folder / "evidence_manifest.json"
        manifest = archive.read(path)
        row = next(row for row in manifest["files"] if row["published_path"] == relative)
        row.update(archive.identity(self.folder / relative))
        if relative == "comparison.json":
            manifest["source_report"].update(archive.identity(self.folder / relative))
        write(path, manifest)

    def update_report(self, change):
        path = self.folder / "comparison.json"
        data = archive.read(path)
        change(data)
        write(path, data)
        self.rehash("comparison.json")

    def test_relocated_archive_verifies_when_all_original_paths_are_absent(self):
        hidden = self.fixture.private.with_name("hidden-private")
        self.fixture.private.rename(hidden)
        try:
            report = self.verify()
        finally:
            hidden.rename(self.fixture.private)
        self.assertEqual(report["comparison"], self.fixture.report)
        self.assertTrue(report["raw_logs_replayed"])
        self.assertFalse(report["tensor_contents_reverified_offline"])
        self.assertTrue(all(path.is_absolute() and path.is_file() for path in report["files"]))
        self.assertFalse(any(path.suffix in (".pt", ".safetensors") for path in report["files"]))
        self.assertEqual(report["calibration_costs"]["category_bake_wall_seconds"], 4.5)
        self.assertEqual(report["calibration_costs"]["ordinary_collection_seconds"], 1.5)
        self.assertEqual(report["calibration_costs"]["ordinary_bake_seconds"], 3.25)

    def test_replay_uses_snapshot_functions_and_restores_imports(self):
        before = {name: module for name, module in sys.modules.items() if name in
                  ("paper", "eval", "exp", "run_recovery_eval", "compare_recovery", "paper.paired_uncertainty")}
        old_path = sys.path[:]
        with patch.object(live, "summarize", side_effect=AssertionError("live workspace algorithm used")), \
             patch.object(live, "compare", side_effect=AssertionError("live model validation used")):
            self.verify()
        self.assertEqual(sys.path, old_path)
        self.assertTrue(all(sys.modules[name] is value for name, value in before.items()))
        self.assertFalse(list(self.folder.rglob("__pycache__")))

    def test_archived_cli_replays_in_an_isolated_python_without_the_original_tree(self):
        hidden = self.fixture.private.with_name("hidden-private")
        self.fixture.private.rename(hidden)
        try:
            result = subprocess.run([sys.executable, "-I", "-B",
                str(self.folder / "source/paper/collect_gptq_reference.py"),
                "--verify", str(self.folder), "--main-evidence", str(self.main)],
                cwd=self.main.parent, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                text=True, capture_output=True, timeout=30)
        finally:
            hidden.rename(self.fixture.private)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["paired_statistics_recomputed"])

    def test_collect_rejects_failed_live_tensor_gate_without_publishing(self):
        out = self.main / "rejected"
        with patch.object(archive, "ROOT", self.fixture.private), \
             patch.object(live, "compare", side_effect=ValueError("synthetic live tensor drift")) as gate:
            with self.assertRaisesRegex(ValueError, "tensor drift"):
                archive.collect(self.fixture.report_path, out, self.fixture.main)
        self.assertEqual(gate.call_count, 1)
        self.assertFalse(out.exists())

    def test_collect_refuses_existing_destination_before_live_validation(self):
        with patch.object(live, "compare", side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError, "Refusing existing"):
                archive.collect(self.fixture.report_path, self.folder, self.main)

    def test_atomic_publication_refuses_even_an_existing_empty_directory(self):
        stage, target = self.main / "stage", self.main / "racing-destination"
        stage.mkdir()
        target.mkdir()
        (stage / "receipt.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            archive.publish_new(stage, target)
        self.assertTrue((stage / "receipt.json").is_file())
        self.assertEqual(list(target.iterdir()), [])

    def test_missing_or_unlisted_archive_file_is_rejected(self):
        path = self.folder / ("heldout_gptq/" + TASKS[0] + ".log")
        path.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.verify()

    def test_unregistered_file_is_rejected(self):
        (self.folder / "unlisted.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            self.verify()

    def test_manifest_cannot_escape_archive_root(self):
        path = self.folder / "evidence_manifest.json"
        data = archive.read(path)
        data["files"][0]["published_path"] = "../outside.json"
        write(path, data)
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            self.verify()

    def test_rehashed_source_drift_is_rejected_by_original_producer_identity(self):
        relative = "source/eval/run_recovery_eval.py"
        path = self.folder / relative
        path.write_text(path.read_text() + "\n# changed parser\n")
        self.rehash(relative)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.verify()

    def test_rehashed_ordinary_recipe_must_match_category_parent_receipt(self):
        relative = "artifacts/ordinary_parent/ptq_recipe.json"
        write(self.folder / relative, {"another": "recipe"})
        self.rehash(relative)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.verify()

    def test_rehashed_capture_reset_log_must_match_the_frozen_capture_manifest(self):
        relative = "calibration/teacher/observations/" + TASKS[0] + "/reset_events.jsonl"
        (self.folder / relative).write_text("{}\n")
        self.rehash(relative)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.verify()

    def test_rehashed_statistics_are_recomputed_from_real_logs(self):
        self.update_report(lambda data: data["contrasts"]["gptq_vs_rtn"].update(difference_pp=99.0))
        with self.assertRaisesRegex(ValueError, "statistics do not reproduce"):
            self.verify()

    def test_rehashed_task_json_cannot_override_rollout_results(self):
        relative = "heldout_gptq/task_results.json"
        path = self.folder / relative
        data = archive.read(path)
        data[TASKS[0]]["results"][0] = not data[TASKS[0]]["results"][0]
        write(path, data)
        self.rehash(relative)
        self.update_report(lambda report: report["raw_evaluation_audits"]["gptq"]["evaluation_identity"]["task_results.json"].update(archive.identity(path)))
        with self.assertRaisesRegex(ValueError, "Raw log differs"):
            self.verify()

    def test_rehashed_server_still_requires_full_activation_installation(self):
        relative = "heldout_gptq/" + TASKS[0] + ".server.log"
        path = self.folder / relative
        path.write_text("synthetic server requested W4A4 but did not install it\n")
        self.rehash(relative)
        self.update_report(lambda report: report["raw_evaluation_audits"]["gptq"]["activation_installation"]["server_logs"][0].update(sha256=archive.identity(path)["sha256"]))
        with self.assertRaisesRegex(RuntimeError, "one activation"):
            self.verify()

    def test_rehashed_log_and_result_cannot_hide_a_wrong_reset(self):
        log_relative = "heldout_gptq/" + TASKS[0] + ".log"
        log = self.folder / log_relative
        text = log.read_text()
        seed = self.fixture.data["partitions"]["heldout"]["seed"]
        log.write_text(text.replace('"seed": ' + str(seed), '"seed": ' + str(seed + 99), 1))
        results_path = self.folder / "heldout_gptq/task_results.json"
        results = archive.read(results_path)
        results[TASKS[0]].update(parse_log(log))
        write(results_path, results)
        self.rehash(log_relative)
        self.rehash("heldout_gptq/task_results.json")

        def update(report):
            audit = report["raw_evaluation_audits"]["gptq"]
            audit["raw_log_identities"][TASKS[0]].update(archive.identity(log))
            audit["evaluation_identity"]["task_results.json"].update(archive.identity(results_path))

        self.update_report(update)
        with self.assertRaisesRegex(ValueError, "reset differs from declared"):
            self.verify()

    def test_main_evidence_must_be_the_original_five_arm_bytes(self):
        path = self.main / "heldout_qad/summary.json"
        path.write_text(path.read_text() + " \n")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.verify()

    def test_missing_timing_fields_are_null_not_zero(self):
        for relative in ("calibration/ordinary/calib_meta.json", "calibration/category/calib_meta.json",
                         "artifacts/ordinary_parent/bake_manifest.json", "artifacts/category_bake_invocation.json"):
            write(self.folder / relative, {})
        result = archive.calibration_costs(self.folder)
        self.assertTrue(all(value is None for key, value in result.items() if key != "scopes"))

    def test_invalid_timing_is_rejected(self):
        write(self.folder / "calibration/ordinary/calib_meta.json", {"elapsed_seconds": -1})
        with self.assertRaisesRegex(ValueError, "elapsed_seconds"):
            archive.calibration_costs(self.folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
