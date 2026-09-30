#!/usr/bin/env python3
"""Read-only release evidence audit; stdout is one JSON object, never a score claim.

This auditor does not import the experiment driver, load a GPU, repair manifests,
rewrite historical hashes, or adopt an incomplete stage.  ``matched`` means an
existing declaration matches the bytes inspected now; ``failed`` means an
observed contradiction; ``unverifiable`` means missing/unstable evidence.  A
current snapshot is never substituted for an absent historical identity.

Run with the training venv for CPU teacher-cache payload checks.  ``--out`` is
optional and must be outside the audited run.  Exit 0 requires all checks to
match; 1 means contradictions, 2 means incomplete/unverifiable evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CORE_METADATA = ("config.json", "statistics.json", "processor_config.json",
                 "embodiment_id.json", "model.safetensors.index.json")
STATES = ("matched", "failed", "unverifiable")


class Audit:
    def __init__(self, repo: Path, payload=True):
        self.repo = repo.resolve()
        self.payload = payload
        self.checks = []
        self.snapshots = {}
        self._hashes = {}

    def add(self, key, status, message, **detail):
        assert status in STATES
        self.checks.append({"id": key, "status": status, "message": message, **detail})
        return status == "matched"

    def equal(self, key, actual, expected, message, remedy=None):
        return self.add(key, "matched" if actual == expected else "failed", message,
                        actual=actual, expected=expected, **({"remedy": remedy} if remedy else {}))

    def missing(self, key, message, **detail):
        return self.add(key, "unverifiable", message, **detail)

    def read(self, path, key):
        path = Path(path)
        try:
            before = path.stat()
            data = json.loads(path.read_text())
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                self.missing(key, "JSON changed during inspection", path=str(path)); return None
            if not isinstance(data, dict):
                raise ValueError("JSON root is not an object")
            return data
        except FileNotFoundError:
            self.missing(key, "Required JSON is unavailable", path=str(path)); return None
        except (OSError, ValueError) as exc:
            self.add(key, "failed", "Cannot read declared JSON", path=str(path), error=str(exc)); return None

    def digest(self, path):
        path = Path(path)
        before = path.stat()
        token = (str(path.resolve()), before.st_size, before.st_mtime_ns)
        if token in self._hashes:
            return self._hashes[token]
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("file changed while hashing")
        row = {"path": str(path.resolve()), "bytes": before.st_size, "sha256": h.hexdigest()}
        self._hashes[token] = row
        return row

    def file(self, path, expected, key):
        if isinstance(expected, str):
            expected = {"sha256": expected}
        if not isinstance(expected, dict) or not expected.get("sha256"):
            return self.missing(key, "Historical file identity was not recorded", path=str(path),
                                remedy="Preserve a producer-recorded SHA; do not infer it from current bytes.")
        try:
            actual = self.digest(path)
        except (OSError, RuntimeError) as exc:
            return self.missing(key, "Declared file is missing or changing", path=str(path), error=str(exc))
        fields = [k for k in ("bytes", "sha256") if k in expected]
        return self.equal(key, {k: actual[k] for k in fields}, {k: expected[k] for k in fields},
                          "Recorded file identity versus current bytes", path_remedy())

    def checkpoint(self, folder, expected, key):
        """Check all recorded identities and expose metadata with no prior hash."""
        folder = Path(folder)
        if not folder.is_dir():
            self.missing(key, "Checkpoint directory is unavailable", path=str(folder)); return
        expected = expected if isinstance(expected, dict) else {}
        weights = expected.get("weights", expected.get("shards", {}))
        if isinstance(weights, list):
            weights = {x["name"]: x for x in weights}
        actual_names = sorted(p.name for p in folder.glob("*.safetensors"))
        if weights:
            self.equal(key + ".shard_set", actual_names, sorted(weights), "Physical checkpoint shard set")
        else:
            self.missing(key + ".shard_set", "No producer-recorded complete shard identity", path=str(folder))
        snapshot = {"path": str(folder.resolve()), "weights": {}, "metadata": {}}
        for name in sorted(set(actual_names) | set(weights)):
            self.file(folder / name, weights.get(name), key + ".weight." + name)
            try:
                snapshot["weights"][name] = self.digest(folder / name)
            except (OSError, RuntimeError):
                pass
        metadata = expected.get("metadata", {})
        names = set(CORE_METADATA) | set(metadata)
        names |= {p.name for p in folder.glob("*.json")}
        for name in sorted(names):
            self.file(folder / name, metadata.get(name), key + ".metadata." + name)
            try:
                snapshot["metadata"][name] = self.digest(folder / name)
            except (OSError, RuntimeError):
                pass
        index = self.read(folder / "model.safetensors.index.json", key + ".index")
        if index:
            self.equal(key + ".indexed_shards", actual_names,
                       sorted(set(index.get("weight_map", {}).values())), "Index references exactly the physical shards")
        self.snapshots[key] = snapshot

    def sources(self, identities, key, required=()):
        identities = identities if isinstance(identities, dict) else {}
        for name in sorted(set(identities) | set(required)):
            self.file(self.repo / name, identities.get(name), key + "." + name)


def path_remedy():
    return "Restore the declared artifact or document a new producer run; never overwrite its historical hash."


def protocol_checks(audit, protocol, run):
    selected = protocol.get("selection", {})
    minimum = selected.get("pressure_rule", {}).get("min_drop_from_bf16")
    text = selected.get("rule", "")
    numbers = re.findall(r"at least\s+([0-9.]+)\s+below\s+BF16", text, re.I)
    if numbers:
        audit.equal("protocol.pressure_text", [float(v) for v in numbers], [minimum] * len(numbers),
                    "Human-readable pressure threshold agrees with structured rule",
                    "Keep the frozen protocol bytes; publish a separate erratum stating which structured value executed.")
    else:
        audit.missing("protocol.pressure_text", "No recognized numerical pressure rule in prose; manual review required")
    mapping = {"train_seed": "train_seed", "qad_steps": "qad_optimizer_steps",
               "continuation_steps": "continuation_optimizer_steps", "lora_scope": "recovery_scope",
               "rank": "rank", "alpha": "alpha", "effective_demo_batch": "effective_demo_batch",
               "opd_every": "opd_every", "qad_learning_rates": "qad_learning_rates", "opd_weights": "opd_weights"}
    for field, source in mapping.items():
        if field not in run or source not in selected:
            audit.missing("configuration." + field, "Run/protocol configuration field was not recorded")
        else:
            audit.equal("configuration." + field, run[field], selected[source], "Run configuration equals protocol")
    parts = protocol.get("partitions", {})
    for left, right in (("development", "collection"), ("development", "heldout"), ("collection", "heldout")):
        overlap = sorted(set(parts.get(left, {}).get("init_state_indices", [])) &
                         set(parts.get(right, {}).get("init_state_indices", [])))
        audit.equal(f"protocol.disjoint.{left}.{right}", overlap, [], "Official initial-state partitions are disjoint")


def format_check(audit, recipe):
    """Check the declared target format; an absent FP8 allocation is explicit."""
    memory = recipe.get("memory", {}) if isinstance(recipe, dict) else {}
    fp8 = memory.get("fraction_of_eligible_params", {}).get("fp8")
    if fp8 is None:
        return audit.missing("quantization.fp8_coverage", "Category recipe has no format coverage declaration")
    if float(fp8) <= 0:
        return audit.add("quantization.fp8_coverage", "failed",
                         "The selected category recipe contains no FP8 eligible parameters; it cannot support the declared FP4/FP8 mixed claim",
                         actual=fp8, remedy="Produce and bind a parent recipe with a nonzero FP8 allocation before publishing a mixed-precision result.")
    return audit.add("quantization.fp8_coverage", "matched", "Selected recipe declares a nonzero FP8 allocation", actual=fp8)


def parent_checks(audit, base):
    """A missing parent is not declared rebuildable merely because a script exists."""
    recipe = audit.read(base / "category_ptq_recipe.json", "category.recipe")
    manifest = audit.read(base / "category_bake_manifest.json", "category.manifest")
    if not recipe or not manifest:
        return
    format_check(audit, recipe)
    audit.file(base / "category_ptq_recipe.json", manifest.get("category_recipe"), "category.recipe_binding")
    audit.file(base / "ptq_recipe.json", recipe.get("parent_recipe_sha256"), "category.parent_recipe_copy")
    audit.file(base / "bake_manifest.json", recipe.get("parent_bake_manifest_sha256"), "category.parent_bake_copy")
    audit.sources({"quant/ptq/bake_category.py": manifest.get("implementation_sha256")}, "category.source",
                  ("quant/ptq/category_fp4.py",))
    # Category manifest records rewritten shards; parent identities bind the copied remainder.
    parent_id = manifest.get("parent", {}).get("parent", {})
    outputs = dict(parent_id.get("weights", {})); outputs.update(manifest.get("output_weights", {}))
    audit.checkpoint(base, {"weights": outputs, "metadata": parent_id.get("metadata", {})}, "category.checkpoint")
    parent = Path(recipe.get("parent", "/nonexistent"))
    if parent.is_dir():
        audit.checkpoint(parent, parent_id, "category.parent")
        audit.add("category.parent_rebuild", "matched", "Original parent checkpoint is available; no rebuild is needed")
        return
    copied = audit.read(base / "ptq_recipe.json", "category.parent_copy")
    if copied is None:
        return
    calib = copied.get("calibration_provenance", {})
    candidates = [Path(x) for x in (copied.get("calib"), calib.get("path")) if x]
    available = [p for p in candidates if p.is_file()]
    if available:
        for p in available:
            audit.file(p, calib.get("sha256"), "category.parent_calibration." + p.name)
    bake = audit.read(base / "bake_manifest.json", "category.parent_bake")
    if bake:
        audit.sources(bake.get("implementation_sha256"), "category.parent_sources")
    audit.missing("category.parent_rebuild", "Parent checkpoint is absent; exact reconstruction has not been demonstrated",
                  parent=str(parent), calibration_candidates=[str(p) for p in candidates],
                  calibration_available=[str(p) for p in available],
                  classification="rebuild_inputs_available_unproven" if available else "missing_calibration_dependency",
                  remedy="Rebuild to a new directory from the recorded base, calibration and source revision, then compare all parent shard SHAs before claiming exact reconstruction.")


def tree_equal(left, right):
    """Tensor-aware exact CPU comparison without model construction."""
    import torch
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and
                left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(tree_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(tree_equal(x, y) for x, y in zip(left, right))
    return type(left) is type(right) and left == right


def cache_checks(audit, folder, teacher, student, protocol):
    meta = audit.read(folder / "teacher_probes.json", "cache.metadata")
    if not meta:
        return
    audit.equal("cache.teacher_path", meta.get("teacher"), str(teacher), "Teacher is the declared BF16 base")
    audit.checkpoint(teacher, {"weights": meta.get("teacher_weights", {}), "metadata": {
        "config.json": meta.get("teacher_config_sha256"),
        "statistics.json": meta.get("teacher_statistics_sha256")}}, "cache.teacher")
    audit.sources({"rl/opd_probe_cache.py": meta.get("labeling_implementation_sha256"),
                   "rl/probe_distill.py": meta.get("replay_implementation_sha256")}, "cache.sources")
    expected_count = 10 * protocol.get("partitions", {}).get("collection", {}).get("episodes_per_task", 0) * 4
    audit.equal("cache.count", [meta.get("count"), meta.get("requested_count")], [expected_count] * 2, "Frozen observation budget")
    sources = meta.get("source_observation_files", [])
    audit.equal("cache.source_count", len({r.get("path") for r in sources}), expected_count, "Unique source observation count")
    for i, source in enumerate(sources):
        audit.file(Path(source.get("path", "/nonexistent")), source, f"cache.observation.{i}")
    audit.missing("cache.student_weight_binding", "Cache metadata does not bind student checkpoint weight bytes",
                  remedy="Use a producer-recorded student shard identity; matching the path/config alone is insufficient.") if not meta.get("student_weights") else audit.checkpoint(student, {"weights": meta["student_weights"]}, "cache.student")
    if not audit.payload:
        audit.missing("cache.payload", "CPU payload inspection explicitly disabled"); return
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        payload = torch.load(folder / "teacher_probes.pt", map_location="cpu", weights_only=True)
        if payload.get("version") != 3 or payload.get("metadata") != meta:
            raise ValueError("Serialized version/metadata differs from sidecar")
        samples = payload.get("samples", [])
        if len(samples) != expected_count or len(samples) != len(sources):
            raise ValueError("Payload sample/source count differs")
        for i, (sample, source_record) in enumerate(zip(samples, sources)):
            raw = torch.load(source_record["path"], map_location="cpu", weights_only=True)
            provenance = sample.get("provenance", {})
            if provenance != {k: v for k, v in raw.items() if k != "inputs"}:
                raise ValueError(f"Sample {i} provenance differs from its recorded source")
            if provenance.get("student_checkpoint") != str(student) or provenance.get("source_kind") != "student_rollout":
                raise ValueError(f"Sample {i} did not use the frozen QAD student")
            inputs = sample["inputs"]
            if not tree_equal(inputs, raw["inputs"]):
                raise ValueError(f"Sample {i} input tensors differ from captured inputs")
            pred, action, mask = sample["pred"], inputs["action"], inputs["action_mask"]
            if (pred.shape != action.shape or mask.shape != pred.shape or action.shape[0] != 1 or
                    not all(torch.isfinite(t).all() for t in (pred, action, mask)) or
                    not ((mask == 0) | (mask == 1)).all() or mask.sum() <= 0 or
                    sample.get("seed") != meta.get("seed", -1) + i):
                raise ValueError(f"Sample {i} velocity/action/mask/seed is invalid")
        audit.add("cache.payload", "matched", "CPU payload matches source observations, provenance and sidecar", samples=len(samples))
    except ImportError as exc:
        audit.missing("cache.payload", "CPU payload verification requires torch in the audit interpreter", error=str(exc))
    except Exception as exc:
        audit.add("cache.payload", "failed", "CPU teacher payload validation failed", error=str(exc))


def audit_run(run_dir, repo=ROOT, payload=True):
    run_dir = Path(run_dir).resolve()
    a = Audit(Path(repo), payload)
    run = a.read(run_dir / "run_manifest.json", "run.manifest")
    if not run:
        return finish(a, run_dir)
    protocol_path = Path(run.get("protocol_file", "/nonexistent"))
    protocol = a.read(protocol_path, "protocol.document")
    a.file(protocol_path, run.get("protocol_sha256"), "protocol.run_binding")
    selection_path = Path(run.get("selection_file", "/nonexistent"))
    selection = a.read(selection_path, "selection.document")
    a.file(selection_path, run.get("selection_sha256"), "selection.run_binding")
    a.sources({"exp/run_high_fp4_v3.py": run.get("implementation_sha256")}, "run.sources",
              ("exp/run_category_recovery.py", "eval/run_recovery_eval.py", "eval/serve_recovery.py", "eval/rollout_seeded.py"))
    if protocol:
        protocol_checks(a, protocol, run)
    teacher = Path(run.get("base", "/nonexistent"))
    base = Path(run.get("selected_ptq_checkpoint", "/nonexistent"))
    a.checkpoint(teacher, run.get("base_identity"), "run.base")
    if selection:
        row = selection.get("arms", {}).get(selection.get("selected_recipe"), {})
        a.equal("selection.checkpoint", row.get("checkpoint"), str(base), "Selected checkpoint equals run input")
        a.equal("selection.recipe", selection.get("selected_recipe"), run.get("selected_recipe"), "Selected recipe equals run input")
        a.equal("selection.heldout", selection.get("selection_uses_heldout"), False, "Selection declares development-only evidence")
        a.file(protocol_path, selection.get("protocol_sha256"), "selection.protocol_binding")
        for arm, rec in selection.get("arms", {}).items():
            eval_path = selection_path.parent / "development" / arm / "eval_manifest.json"
            raw = a.read(eval_path, "selection.raw." + arm)
            if raw and protocol:
                expected = protocol.get("selection", {}).get("source_development_protocol_sha256", run.get("protocol_sha256"))
                a.equal("selection.raw_protocol." + arm, raw.get("protocol_sha256"), expected, "Raw development uses declared source protocol")
    if (base / "category_ptq_recipe.json").is_file():
        a.file(base / "category_ptq_recipe.json", run.get("selected_ptq_recipe_sha256"), "selection.category_recipe_binding")
        parent_checks(a, base)
    else:
        a.file(base / "ptq_recipe.json", run.get("selected_ptq_recipe_sha256"), "selection.ptq_recipe_binding")
    stages = run.get("stages", {})
    for name, record in stages.items():
        disk = a.read(run_dir / "stages" / (name + ".json"), "stage.receipt." + name)
        if disk is not None:
            a.equal("stage.record." + name, disk, record, "Stage receipt equals run manifest record")
        output = Path(record.get("output", "/nonexistent"))
        for field in ("checkpoint_identity", "model_identity"):
            if record.get(field):
                a.checkpoint(Path(record[field]["path"]), record[field], name + "." + field)
        for field in ("selection_identity", "cache_identity", "metadata_identity", "training_request_identity", "runtime_identity"):
            if record.get(field):
                a.file(Path(record[field]["path"]), record[field], name + "." + field)
        if name.startswith("train_"):
            rec = a.read(output / "recovery_manifest.json", name + ".recovery")
            request = a.read(output / "orchestrator_training_request.json", name + ".request")
            if rec:
                a.sources(rec.get("recovery_source_sha256"), name + ".sources")
                a.equal(name + ".base_path", rec.get("base"), str(base), "Training base equals selection")
                a.file(protocol_path, rec.get("protocol_sha256"), name + ".protocol")
            if request:
                a.checkpoint(base, request.get("base_identity"), name + ".base")
                for field in ("initial_adapter_identity",):
                    if request.get(field):
                        a.checkpoint(Path(request[field]["path"]), request[field], name + ".initial_adapter")
                if request.get("cache_identity"):
                    a.file(Path(request["cache_identity"]["path"]), request["cache_identity"], name + ".cache")
        if name.startswith("merge_"):
            merge = a.read(output / "merge_manifest.json", name + ".merge")
            if merge:
                a.file(a.repo / "rl/lora_merge_bake.py", merge.get("implementation_sha256"), name + ".source")
                for field, folder in (("base_weights", base), ("training_weights", Path(merge.get("training_checkpoint", "/nonexistent"))), ("output_weights", output)):
                    a.checkpoint(folder, {"weights": merge.get(field, {})}, name + "." + field)
                for fn in ("category_ptq_recipe.json", "category_bake_manifest.json"):
                    if (base / fn).is_file():
                        a.file(output / fn, a.digest(base / fn), name + ".category_copy." + fn)
    qs = a.read(run_dir / "artifacts/select_qad_lr/selection.json", "qad.selection")
    if qs and protocol:
        lr = qs.get("selected_learning_rate")
        student = run_dir / "artifacts" / ("merge_qad_lr_" + str(float(lr)))
        cache_checks(a, run_dir / "artifacts/teacher_cache", teacher, student, protocol)
    final = a.read(run_dir / "final_manifest.json", "final.manifest")
    if final:
        a.equal("final.required_arms", final.get("required_arms"), ["bf16", "ptq", "qad", "continued_qad", "qad_opd"], "Complete paired comparison arm set")
        a.file(protocol_path, final.get("protocol_sha256"), "final.protocol")
        if final.get("heldout_comparison"):
            a.file(Path(final["heldout_comparison"]["path"]), final["heldout_comparison"], "final.comparison")
    a.equal("run.completion", run.get("status"), "complete", "Run is complete before release") if final else a.missing("run.completion", "Run has no final manifest; interim evidence only")
    return finish(a, run_dir)


def finish(audit, run_dir):
    counts = dict(Counter(row["status"] for row in audit.checks))
    status = "failed" if counts.get("failed") else "unverifiable" if counts.get("unverifiable") else "matched"
    return {"format": "recovery_release_integrity_v1", "status": status,
            "read_only": True, "run_dir": str(run_dir), "repo_root": str(audit.repo),
            "inspected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "meaning": {"matched": "existing declaration matches inspected evidence",
                        "failed": "an observed declaration contradicts inspected evidence",
                        "unverifiable": "missing, unstable, unrecorded or incomplete evidence"},
            "counts": {key: counts.get(key, 0) for key in STATES}, "checks": audit.checks,
            "current_snapshots_not_historical_proof": audit.snapshots}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--skip-cache-payload", action="store_true")
    parser.add_argument("--out", help="Optional JSON report outside the audited run directory")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if args.out:
        out = Path(args.out).resolve()
        if out == run_dir or run_dir in out.parents:
            parser.error("--out must be outside the audited run directory")
        if out.exists():
            parser.error("refusing to overwrite an existing audit report")
    report = audit_run(run_dir, Path(args.repo_root), not args.skip_cache_payload)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x") as stream:
            stream.write(text)
    sys.stdout.write(text)
    return {"matched": 0, "failed": 1, "unverifiable": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
