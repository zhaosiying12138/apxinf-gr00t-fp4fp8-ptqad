"""CPU evidence-chain tests using two real, tiny safetensors shards."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from eval import gptq_reference_evidence as evidence


CATEGORY = "action_head.fixture.W"
LINEAR = "ordinary"
NONLINEAR = "action_head.position_embedding"
CHANGED_SHARD = "model-00001-of-00002.safetensors"
ORDINARY_SHARD = "model-00002-of-00002.safetensors"
CATEGORY_IMPL = importlib.import_module(evidence.validate_parent.__module__)
SOURCE_FILES = (
    "quant/ptq/collector.py", "quant/ptq/collector_category.py",
    "quant/ptq/bake.py", "quant/ptq/quantizers.py", "quant/torch_fp4.py",
    "quant/ptq/bake_category.py",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def tiny_categories(entries):
    return {CATEGORY: entries[CATEGORY]}


class EvidenceFixture:
    """Real file, tensor and cache identities; only model size is synthetic."""

    def __init__(self, root, *, relative_calib=False):
        self.root = root
        self.teacher = root / "teacher"
        output = self.output = root / "supplement"
        self.parent = output / "ordinary_parent"
        self.final = output / "w4a4_category"
        self.ordinary_h = output / "ordinary_h"
        self.category_h = output / "category_h"
        for folder in (self.teacher, self.parent, self.final, self.ordinary_h, self.category_h):
            folder.mkdir(parents=True)
        self.parent_tensors = {
            LINEAR + ".weight": torch.arange(32, dtype=torch.float32).reshape(2, 16).to(torch.bfloat16),
            CATEGORY: torch.arange(48, dtype=torch.float32).reshape(3, 16, 1).to(torch.bfloat16),
            "ordinary.bias": torch.tensor([1., 2.], dtype=torch.bfloat16),
            NONLINEAR + ".weight": torch.ones((2, 16), dtype=torch.bfloat16),
        }
        weight_map = {key: ORDINARY_SHARD if key == NONLINEAR + ".weight" else CHANGED_SHARD
                      for key in self.parent_tensors}
        for folder in (self.teacher, self.parent):
            for shard in (CHANGED_SHARD, ORDINARY_SHARD):
                tensors = {key: value.clone() for key, value in self.parent_tensors.items()
                           if weight_map[key] == shard}
                if folder == self.teacher:
                    # The ordinary parent may quantize eligible tensors but must retain categories.
                    tensors = {key: value + .5 if key.endswith(".weight") else value
                               for key, value in tensors.items()}
                save_file(tensors, str(folder / shard))
            for name in evidence.METADATA:
                write_json(folder / name, {"libero_sim": 2} if name == "embodiment_id.json" else {"fixture": name})
            write_json(folder / "model.safetensors.index.json", {"weight_map": weight_map})
        teacher_id, _ = evidence.checkpoint_identity(self.teacher)
        self.entries, _, _ = evidence.inventory(self.parent)
        self.plan = {
            "execution": {"output_root": "supplement"},
            "main_run": "main_run",
            "teacher_weights": evidence.weights_only(teacher_id["weights"]),
            "teacher_metadata": {name: teacher_id["metadata"][name]["sha256"] for name in evidence.METADATA},
            "preparation_source_files": {name: {"sha256": "fixture source " + name} for name in SOURCE_FILES},
            "calibration": {"batch": 1, "seed": 123, "windows": 2, "gptq_damp": .01,
                            "clip_grid": list(evidence.CLIP_GRID), "category_method": "gptq_active"},
            "quantization": {"eligible_weight_tensors": 3},
        }
        capture = {
            "source_audit": {"teacher": str(self.teacher), "teacher_weights": self.plan["teacher_weights"]},
            "teacher_metadata": self.plan["teacher_metadata"],
            "protocol_file": str(root / "protocol.json"), "order": "fixture capture order",
            "calibration_forward": "training velocity forward", "implementation_sha256": "fixture capture source",
        }
        write_json(root / "capture.json", capture)
        capture_id = evidence.identity(root / "capture.json")
        self.plan["capture_manifest"] = {**capture_id, "path": "capture.json"}
        self.capture_record = {**capture_id, **{key: capture[key] for key in (
            "source_audit", "protocol_file", "order", "calibration_forward", "implementation_sha256")}}
        h = {"H": torch.eye(16), "abs": torch.ones(16), "n": 4, "calls": 2}
        torch.save({LINEAR: h}, self.ordinary_h / "calib.pt")
        ordinary_meta = self.common_meta("full-model-cpu-hessian-v2", "quant/ptq/collector.py")
        ordinary_meta.update(
            base=str(self.teacher), base_weight_files=self.plan["teacher_weights"],
            base_config_sha256=self.plan["teacher_metadata"]["config.json"],
            base_statistics_sha256=self.plan["teacher_metadata"]["statistics.json"],
            recipe_targets="calib", tied_weight_aliases={}, nonlinear_targets={NONLINEAR: "Embedding"},
            rows_per_layer={LINEAR: 4}, calls_per_layer={LINEAR: 2}, n_layers=1,
            cache_sha256=evidence.identity(self.ordinary_h / "calib.pt")["sha256"],
        )
        write_json(self.ordinary_h / "calib_meta.json", ordinary_meta)
        planned, excluded, _ = evidence.make_plan(self.entries, "calib", False)
        planned[LINEAR].update(method="nvfp4_gptq", actual_method="nvfp4_gptq", calibration_rows=4, clip=.9)
        planned[NONLINEAR].update(method="nvfp4_rtn", actual_method="nvfp4_rtn",
                                  fallback_reason="documented_non_linear_target")
        recipe = {
            "base": str(self.teacher), "recipe": "calib", "recipe_version": evidence.RECIPE_VERSION,
            "calibration_mode": "required", "head_bf16": False, "gptq_damp": .01, "rtn_clip": 1.,
            "calib": str((self.ordinary_h / "calib.pt").relative_to(root)) if relative_calib else str(self.ordinary_h / "calib.pt"),
            "calibration_provenance": {
                "path": str(self.ordinary_h / "calib.pt"), "sha256": ordinary_meta["cache_sha256"], "layers": 1,
                "metadata": ordinary_meta, "metadata_sha256": evidence.identity(self.ordinary_h / "calib_meta.json")["sha256"],
            },
            "layers": planned, "excluded_tensors": excluded, "tied_weight_aliases": {},
            "actual_method_tensor_counts": {"nvfp4_gptq": 1, "nvfp4_rtn": 1}, "n_weights_edited": 2,
        }
        write_json(self.parent / "ptq_recipe.json", recipe)
        parent_id, _ = evidence.checkpoint_identity(self.parent)
        write_json(self.parent / "bake_manifest.json", {
            "status": "complete", "recipe_version": evidence.RECIPE_VERSION, "base": str(self.teacher),
            "source_weight_files": self.plan["teacher_weights"],
            "output_weight_files": evidence.weights_only(parent_id["weights"]),
            "implementation_sha256": {name: self.plan["preparation_source_files"][name]["sha256"]
                                      for name in SOURCE_FILES[2:5]},
        })
        self.source, _, _ = evidence.validate_parent(self.parent)
        torch.save({CATEGORY: {"2": h}}, self.category_h / "calib.pt")
        category_meta = self.common_meta(evidence.CACHE_VERSION, "quant/ptq/collector_category.py")
        category_meta.update(
            parent=str(self.parent), active_bank=2,
            source_category_sha256=self.source["category_source_sha256"],
            provenance={"parent": self.source,
                        "parent_recipe_sha256": evidence.identity(self.parent / "ptq_recipe.json")["sha256"],
                        "parent_bake_sha256": evidence.identity(self.parent / "bake_manifest.json")["sha256"]},
            cache_sha256=evidence.identity(self.category_h / "calib.pt")["sha256"],
            layers={CATEGORY: {"2": {"rows": 4, "calls": 2,
                                     "H_sha256": evidence.tensor_hash(h["H"]),
                                     "abs_sha256": evidence.tensor_hash(h["abs"])}}},
        )
        write_json(self.category_h / "calib_meta.json", category_meta)
        self.category_ids = {"cache": evidence.identity(self.category_h / "calib.pt"),
                             "metadata": evidence.identity(self.category_h / "calib_meta.json")}
        for path in self.parent.iterdir():
            shutil.copyfile(path, self.final / path.name)
        final_tensors = load_file(str(self.final / CHANGED_SHARD))
        final_tensors[CATEGORY] += 1
        save_file(final_tensors, str(self.final / CHANGED_SHARD))
        banks = []
        for bank in range(3):
            active = bank == 2
            banks.append({
                "bank": bank, "source_key": CATEGORY,
                "source_sha256": evidence.tensor_hash(self.parent_tensors[CATEGORY][bank]),
                "output_sha256": evidence.tensor_hash(final_tensors[CATEGORY][bank]),
                "source_dtype": "torch.bfloat16", "output_dtype": "torch.bfloat16",
                "inactive_bank": not active, "calibrated": active,
                "method": "nvfp4_gptq" if active else "nvfp4_rtn",
                "calibration_rows": 4 if active else 0, "calibration_calls": 2 if active else 0,
                "gptq_objective": .5 if active else None, "best_rtn_objective": 1.,
                "selected_objective": .5 if active else 1.,
            })
        self.memory = {"eligible_tensor_count": 3, "fp8_params": 0, "bf16_params": 0}
        category_recipe = {
            "version": evidence.VERSION, "recipe": "category_nvfp4_extension", "parent": str(self.parent),
            "parent_recipe_sha256": evidence.identity(self.parent / "ptq_recipe.json")["sha256"],
            "parent_bake_manifest_sha256": evidence.identity(self.parent / "bake_manifest.json")["sha256"],
            "source_base": teacher_id, "active_libero_bank": 2, "active_bank_method": "gptq_active", "rtn_clip": 1.,
            "source_category_sha256": self.source["category_source_sha256"],
            "calibration": self.category_ids, "calibration_metadata": category_meta,
            "implementation_sha256": self.plan["preparation_source_files"][SOURCE_FILES[-1]]["sha256"],
            "categories": {CATEGORY: {"shape": [3, 16, 1], "banks": banks}}, "memory": self.memory,
        }
        write_json(self.final / "category_ptq_recipe.json", category_recipe)
        manifest = {
            "status": "complete", "version": evidence.VERSION, "parent": self.source, "root_bf16": teacher_id,
            "parent_recipe": evidence.identity(self.parent / "ptq_recipe.json"),
            "parent_bake_manifest": evidence.identity(self.parent / "bake_manifest.json"),
            "category_recipe": self.staging_identity(self.final / "category_ptq_recipe.json"),
            "active_libero_bank": 2, "active_bank_method": "gptq_active", "rtn_clip": 1.,
            "source_category_sha256": self.source["category_source_sha256"],
            "implementation_sha256": category_recipe["implementation_sha256"],
            "methods": {"nvfp4_rtn": 2, "nvfp4_gptq": 1},
            "output_category_sha256": {CATEGORY: banks[2]["output_sha256"]},
            "output_weights": {CHANGED_SHARD: self.staging_identity(self.final / CHANGED_SHARD)},
            "memory": self.memory,
        }
        write_json(self.final / "category_bake_manifest.json", manifest)
        (root / "exp").mkdir()
        (root / "main_run").mkdir()
        write_json(root / "main_run/run_manifest.json", {"python": "/fixture/venv/bin/python"})
        write_json(root / "main_run/final_manifest.json", {"status": "complete", "fixture": True})
        (root / "exp/bake_gptq_reference_category.py").write_text("# fixture invocation wrapper\n")
        log = output / "category_bake.log"
        log.write_text("fixture category bake completed\n")
        write_json(output / "category_bake_invocation.json", {
            "format": "gptq_category_bake_invocation_v1", "status": "complete", "returncode": 0,
            "protocol_sha256": "pending", "producer_sha256": category_recipe["implementation_sha256"],
            "cwd": str(root), "command": [
                "/fixture/venv/bin/python", str(root / "quant/ptq/bake_category.py"),
                "--parent", str(self.parent), "--calib", str(self.category_h), "--out", str(self.final),
                "--expected-windows", "2", "--method", "gptq_active", "--gptq-damp", "0.01",
            ],
            "wrapper": evidence.identity(root / "exp/bake_gptq_reference_category.py"),
            "final_manifest": evidence.identity(root / "main_run/final_manifest.json"),
            "log": evidence.identity(log),
        })
        self.sync_plan()

    def sync_plan(self):
        path = self.root / "exp/gptq_reference_protocol_v12.json"
        write_json(path, self.plan)
        self.edit(self.output / "category_bake_invocation.json",
                  lambda value: value.update(protocol_sha256=evidence.identity(path)["sha256"]))

    def common_meta(self, version, collector):
        return {
            "version": version, "status": "complete", "dataset": None,
            "architecture": evidence.ARCHITECTURE, "model_dtype": "bfloat16",
            "accumulation_dtype": "float32", "accumulation_device": "cpu", "tf32_matmul": False,
            "batch": 1, "seed": 123, "windows_requested": 2, "windows_consumed": 2, "forwards": 2,
            "implementation_sha256": self.plan["preparation_source_files"][collector]["sha256"],
            "captured_input_provenance": self.capture_record,
            "windows_are_unique": "unique capture files; one pass; not independent episodes",
        }

    def staging_identity(self, path):
        return {**evidence.identity(path), "path": str(self.root / "expired-stage" / path.name)}

    def edit(self, path, mutate):
        document = evidence.read(path)
        mutate(document)
        write_json(path, document)

    def edit_category_recipe(self, mutate):
        path = self.final / "category_ptq_recipe.json"
        self.edit(path, mutate)
        self.edit(self.final / "category_bake_manifest.json",
                  lambda value: value.update(category_recipe=self.staging_identity(path)))

    def mutate_final_tensor(self, shard, key, *, refresh_receipt=False):
        path = self.final / shard
        tensors = load_file(str(path))
        tensors[key] += 1
        save_file(tensors, str(path))
        if refresh_receipt:
            self.edit(self.final / "category_bake_manifest.json",
                      lambda value: value["output_weights"].update({shard: self.staging_identity(path)}))


class GPTQReferenceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Keep actual parent, Hessian, file, tensor, and make_plan validation.
        # The production category allowlist requires full GR00T-sized tensors.
        for module in (evidence, CATEGORY_IMPL):
            mock = patch.object(module, "category_inventory", tiny_categories)
            mock.start()
            self.addCleanup(mock.stop)
        self.fixture = EvidenceFixture(Path(self.temp.name))
        mock = patch.object(evidence, "memory_budget", return_value=self.fixture.memory)
        mock.start()
        self.addCleanup(mock.stop)

    def verify(self):
        return evidence.verify_gptq(self.fixture.plan, self.fixture.root)

    def test_complete_chain_reports_actual_methods_and_real_file_identities(self):
        report = self.verify()
        self.assertEqual(report["actual_method_counts"], {
            "ordinary": {"nvfp4_gptq": 1, "nvfp4_rtn": 1},
            "category_banks": {"nvfp4_gptq": 1, "nvfp4_rtn": 2},
            "active_category_banks": {"nvfp4_gptq": 1},
        })
        self.assertEqual(report["weight_identity"], evidence.checkpoint_identity(self.fixture.final)[0])
        self.assertEqual(report["root_identity"], evidence.checkpoint_identity(self.fixture.teacher)[0])
        self.assertEqual(report["calibration_provenance"]["capture"], self.fixture.capture_record)
        self.assertEqual(report["calibration_provenance"]["category"]["cache"], self.fixture.category_ids["cache"])
        self.assertIn("no model execution", report["verification"])

    def test_calibrated_active_bank_may_select_rtn_including_ties(self):
        for objective in (1., 2.):
            with self.subTest(gptq_objective=objective):
                self.fixture.edit_category_recipe(lambda value: value["categories"][CATEGORY]["banks"][2].update(
                    method="nvfp4_rtn", gptq_objective=objective, selected_objective=1.))
                self.fixture.edit(self.fixture.final / "category_bake_manifest.json",
                                  lambda value: value.update(methods={"nvfp4_rtn": 3}))
                self.assertEqual(self.verify()["actual_method_counts"]["active_category_banks"], {"nvfp4_rtn": 1})

    def test_ordinary_calibration_path_may_be_relative_to_project_root(self):
        fixture = EvidenceFixture(self.fixture.root / "relative", relative_calib=True)
        report = evidence.verify_gptq(fixture.plan, fixture.root)
        self.assertEqual(report["checkpoint"], str(fixture.final))

    def test_frozen_teacher_and_capture_identities_are_required(self):
        original = copy.deepcopy(self.fixture.plan)
        mutations = (
            (lambda plan: plan["teacher_weights"][CHANGED_SHARD].update(sha256="wrong"), "frozen teacher weights"),
            (lambda plan: plan["teacher_metadata"].update({"config.json": "wrong"}), "root metadata differs"),
            (lambda plan: plan["capture_manifest"].update(sha256="wrong"), "Artifact identity differs"),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                self.fixture.plan = copy.deepcopy(original)
                mutate(self.fixture.plan)
                self.fixture.sync_plan()
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()

    def test_capture_teacher_cannot_be_rebound_even_with_new_capture_hash(self):
        path = self.fixture.root / "capture.json"
        self.fixture.edit(path, lambda value: value["source_audit"].update(teacher=str(self.fixture.parent)))
        self.fixture.plan["capture_manifest"] = evidence.identity(path)
        self.fixture.sync_plan()
        with self.assertRaisesRegex(ValueError, "Frozen capture teacher differs"):
            self.verify()

    def test_both_calibrations_must_match_frozen_capture_protocol(self):
        for folder in (self.fixture.ordinary_h, self.fixture.category_h):
            path = folder / "calib_meta.json"
            original = evidence.read(path)
            mutations = (
                ("seed", lambda value: value.update(seed=124)),
                ("implementation_sha256", lambda value: value.update(implementation_sha256="wrong collector")),
                ("captured_input_provenance", lambda value: value["captured_input_provenance"].update(order="other order")),
            )
            for field, mutate in mutations:
                with self.subTest(cache=folder.name, field=field):
                    write_json(path, original)
                    self.fixture.edit(path, mutate)
                    with self.assertRaisesRegex(ValueError, "Captured calibration metadata differs: " + field):
                        self.verify()
            write_json(path, original)

    def test_ordinary_hessian_bytes_must_match_recipe_and_metadata(self):
        with (self.fixture.ordinary_h / "calib.pt").open("ab") as stream:
            stream.write(b"changed cache")
        with self.assertRaisesRegex(ValueError, "Ordinary H cache SHA differs"):
            self.verify()

    def test_category_hessian_bytes_and_parent_provenance_are_required(self):
        path = self.fixture.category_h / "calib_meta.json"
        original = evidence.read(path)
        self.fixture.edit(path, lambda value: value["provenance"].update(parent_recipe_sha256="wrong"))
        with self.assertRaisesRegex(ValueError, "Category cache was collected from another parent"):
            self.verify()
        write_json(path, original)
        with (self.fixture.category_h / "calib.pt").open("ab") as stream:
            stream.write(b"changed cache")
        with self.assertRaisesRegex(ValueError, "Category cache bytes changed"):
            self.verify()

    def test_ordinary_linear_cannot_silently_fall_back_to_rtn(self):
        self.fixture.edit(self.fixture.parent / "ptq_recipe.json", lambda value: value["layers"][LINEAR].update(
            method="nvfp4_rtn", actual_method="nvfp4_rtn", fallback_reason="missing_hessian"))
        with self.assertRaisesRegex(ValueError, "Ordinary layer did not use frozen GPTQ"):
            self.verify()

    def test_documented_embedding_fallback_reason_is_required(self):
        self.fixture.edit(self.fixture.parent / "ptq_recipe.json", lambda value: value["layers"][NONLINEAR].update(
            fallback_reason="missing_hessian"))
        with self.assertRaisesRegex(ValueError, "Unexpected ordinary fallback"):
            self.verify()

    def test_parent_recipe_and_metadata_copies_are_bound(self):
        self.fixture.edit(self.fixture.final / "ptq_recipe.json", lambda value: value.update(gptq_damp=.1))
        with self.assertRaisesRegex(ValueError, "Category copied parent metadata differs"):
            self.verify()

    def test_final_model_metadata_must_retain_teacher_identity(self):
        write_json(self.fixture.final / "processor_config.json", {"fixture": "different processor"})
        with self.assertRaisesRegex(ValueError, "Final metadata differs: processor_config.json"):
            self.verify()

    def test_changed_shard_requires_exact_output_receipt(self):
        self.fixture.mutate_final_tensor(CHANGED_SHARD, CATEGORY)
        with self.assertRaisesRegex(ValueError, "Artifact identity differs"):
            self.verify()

    def test_ordinary_only_shard_is_preserved_byte_for_byte(self):
        self.fixture.mutate_final_tensor(ORDINARY_SHARD, NONLINEAR + ".weight")
        with self.assertRaisesRegex(ValueError, "Category bake changed an ordinary shard"):
            self.verify()

    def test_all_noncategory_tensors_in_changed_shard_are_preserved(self):
        path = self.fixture.final / CHANGED_SHARD
        original = path.read_bytes()
        for key in (LINEAR + ".weight", "ordinary.bias"):
            with self.subTest(key=key):
                path.write_bytes(original)
                self.fixture.mutate_final_tensor(CHANGED_SHARD, key, refresh_receipt=True)
                with self.assertRaisesRegex(ValueError, "Category bake changed non-category tensor"):
                    self.verify()

    def test_bank_output_hashes_are_checked_after_shard_receipt(self):
        self.fixture.mutate_final_tensor(CHANGED_SHARD, CATEGORY, refresh_receipt=True)
        with self.assertRaisesRegex(ValueError, "Category bank bytes differ"):
            self.verify()

    def test_active_method_must_follow_same_hessian_objective_selection(self):
        self.fixture.edit_category_recipe(lambda value: value["categories"][CATEGORY]["banks"][2].update(
            gptq_objective=2.))
        with self.assertRaisesRegex(ValueError, "Active category method does not follow calibration selection"):
            self.verify()

    def test_inactive_bank_cannot_claim_calibration_or_gptq(self):
        path = self.fixture.final / "category_ptq_recipe.json"
        original = evidence.read(path)
        for mutation, message in (
            ({"method": "nvfp4_gptq"}, "Invalid category bank method"),
            ({"calibration_rows": 4}, "Inactive category unexpectedly used calibration"),
        ):
            with self.subTest(mutation=mutation):
                write_json(path, original)
                self.fixture.edit_category_recipe(lambda value: value["categories"][CATEGORY]["banks"][0].update(mutation))
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()

    def test_missing_or_extra_shard_receipts_are_rejected(self):
        path = self.fixture.final / "category_bake_manifest.json"
        original = evidence.read(path)
        for receipts in ({}, {CHANGED_SHARD: original["output_weights"][CHANGED_SHARD],
                              ORDINARY_SHARD: evidence.identity(self.fixture.final / ORDINARY_SHARD)}):
            with self.subTest(receipts=list(receipts)):
                write_json(path, original)
                self.fixture.edit(path, lambda value: value.update(output_weights=receipts))
                with self.assertRaisesRegex(ValueError, "Category output shard coverage differs"):
                    self.verify()

    def test_expired_staging_path_must_still_have_correct_basename(self):
        self.fixture.edit(self.fixture.final / "category_bake_manifest.json", lambda value: value["output_weights"][CHANGED_SHARD].update(
            path=str(self.fixture.root / "expired-stage" / "another.safetensors")))
        with self.assertRaisesRegex(ValueError, "Artifact path differs"):
            self.verify()

    def test_recovery_artifacts_are_rejected(self):
        write_json(self.fixture.final / "recovery_manifest.json", {})
        with self.assertRaisesRegex(ValueError, "must not contain recovery/LoRA"):
            self.verify()

    def test_invocation_damping_method_and_interpreter_are_frozen(self):
        path = self.fixture.output / "category_bake_invocation.json"
        original = evidence.read(path)
        for index, replacement, message in (
            (-1, "0.1", "arguments differ"),
            (-3, "rtn", "arguments differ"),
            (0, "/fixture/other-venv/bin/python", "interpreter differs"),
        ):
            with self.subTest(index=index, replacement=replacement):
                value = copy.deepcopy(original)
                value["command"][index] = replacement
                write_json(path, value)
                with self.assertRaisesRegex(ValueError, message):
                    evidence.verify_category_invocation(self.fixture.plan, self.fixture.root, self.fixture.output)

    def test_invocation_requires_successful_integer_exit_status(self):
        path = self.fixture.output / "category_bake_invocation.json"
        for returncode in (1, "0", False):
            with self.subTest(returncode=returncode):
                self.fixture.edit(path, lambda value: value.update(returncode=returncode))
                with self.assertRaisesRegex(ValueError, "invocation is incomplete"):
                    evidence.verify_category_invocation(self.fixture.plan, self.fixture.root, self.fixture.output)

    def test_invocation_binds_protocol_log_wrapper_and_main_final_manifest(self):
        paths = (
            (self.fixture.root / "exp/gptq_reference_protocol_v12.json", "another supplement protocol"),
            (self.fixture.output / "category_bake.log", "Artifact identity differs"),
            (self.fixture.root / "exp/bake_gptq_reference_category.py", "Artifact identity differs"),
            (self.fixture.root / "main_run/final_manifest.json", "Artifact identity differs"),
        )
        for path, message in paths:
            with self.subTest(artifact=path.name):
                original = path.read_bytes()
                with path.open("ab") as stream:
                    stream.write(b"\n ")
                with self.assertRaisesRegex(ValueError, message):
                    evidence.verify_category_invocation(self.fixture.plan, self.fixture.root, self.fixture.output)
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
