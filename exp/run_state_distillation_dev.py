#!/usr/bin/env python3
"""Run three fixed, matched continuation arms on the development partition.

Start from an audited stratified-QAD endpoint bundle and its two complete
velocity caches. Reuse the existing training, export, rollout and stage-receipt
implementations. This entry point has no selection, collection or heldout path.
--validate-only performs full CPU input audits without launching these stages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exp.run_high_fp4_v3 import (
    Driver, OrchestrationError, STATE_DISTILLATION_SCOPE, STATE_DISTILLATION_ROLES,
    TASKS, identity, jread, jwrite, require_pairing, sha,
)
from eval.compare_recovery import paired_summary
from rl.endpoint_cache_guard import state_distillation_config

SCHEMA = "fp4vla_state_distillation_development_v1"
ARMS = tuple(STATE_DISTILLATION_ROLES.values())


def require(condition, message):
    if not condition:
        raise OrchestrationError(message)


def external_output(run_dir, bundle, plan, caches):
    """Never place mutable run stages inside their frozen input evidence."""
    require(not Path(run_dir).is_symlink(), "Run directory must not be a symlink")
    output = Path(run_dir).expanduser().resolve()
    protected = [bundle, Path(plan["teacher_checkpoint"]), Path(plan["qad"]["checkpoint"]),
                 Path(plan["qad"]["base"]), Path(plan["qad"]["training_checkpoint"]).parent]
    for source in plan["sources"].values():
        protected.extend((Path(source["source_directory"]), Path(source["view"]).parent))
    require(all(output != path and path not in output.parents and output not in path.parents
                for path in protected), "Run output must be outside frozen sources and checkpoints")
    require(all(output != path and output not in path.parents for path in caches.values()),
            "Run output must not contain its source caches")
    return output


class StateDistillationDev:
    def __init__(self, args):
        self.bundle = Path(args.endpoint_bundle).expanduser().resolve()
        self.caches = {role: Path(getattr(args, role + "_cache")).expanduser().resolve()
                       for role in ("teacher", "student")}
        require(self.caches["teacher"] != self.caches["student"], "Matched roles need distinct velocity caches")
        # These fields only route CPU validation. No checkpoint is loaded until
        # the existing guard independently reaudits this plan, both sources,
        # both endpoint roles and the completed pure-QAD training identity.
        self.plan = jread(self.bundle / "plan.json")
        require(self.plan.get("schema") == "fp4vla_probe_endpoint_plan_v1"
                and self.plan.get("status") == "frozen_cpu_plan", "Expected a frozen endpoint plan")
        routed = argparse.Namespace(**vars(args))
        routed.run_dir = str(external_output(args.run_dir, self.bundle, self.plan, self.caches))
        routed.base = self.plan["teacher_checkpoint"]
        routed.capture_dataset = self.plan["qad"]["training_identity"]["capture_dataset"]
        routed.until = STATE_DISTILLATION_SCOPE
        routed.cleanup_duplicates = False
        routed.allow_opd_nonimprovement = False
        self.driver = Driver(routed, execution_scope=STATE_DISTILLATION_SCOPE)
        self.config = state_distillation_config(self.driver.protocol["data"])
        self.initial = Path(self.plan["qad"]["training_checkpoint"])
        self.base = Path(self.driver.selection["selected_ptq_checkpoint"])
        require(self.driver.protocol["sha256"] == self.plan["protocol_sha256"],
                "Endpoint plan and development protocol differ")
        require(self.base == Path(self.plan["qad"]["base"]),
                "Endpoint QAD uses a different quantized base from the frozen PTQ reference")
        self.binding = self.preflight()
        previous = self.driver.state.get("state_distillation_inputs")
        require(previous is None or previous == self.binding, "State-distillation inputs changed since run creation")
        if previous is None:
            require(not self.driver.state["stages"], "Existing stages lack the matched input binding")
            self.driver.state["state_distillation_inputs"] = self.binding
            self.driver.save()

    def options(self, role):
        return {"initial": self.initial,
                "weight": 0.0 if role == "continued" else self.config["teacher_velocity_weight"],
                "cache": None if role == "continued" else self.caches[role],
                "endpoint_role": role, "endpoint_bundle": self.bundle}

    def preflight(self):
        audits = {}
        for role, arm in STATE_DISTILLATION_ROLES.items():
            options = self.options(role)
            environment = self.driver.training_environment(
                self.driver.art / ("train_" + arm), self.base, self.config["learning_rate"],
                self.config["optimizer_steps"], **options)
            report = self.driver._endpoint_training_identity(options["weight"], environment)
            require(isinstance(report, dict) and report.get("status") == "verified"
                    and report.get("role") == role and report.get("initial_adapter") == str(self.initial),
                    "Matched continuation input audit did not verify " + arm)
            provenance = report["endpoint_bundle"]
            require(provenance["root"] == str(self.bundle)
                    and provenance["plan_sha256"] == sha(self.bundle / "plan.json")
                    and provenance["endpoint_policy_checkpoint"] == self.plan["qad"]["checkpoint"]
                    and provenance["endpoint_policy_files"] == self.plan["qad"]["checkpoint_files"],
                    "Audited endpoint policy differs from the routed starting QAD")
            audits[arm] = report
        return {"schema": SCHEMA, "scope": STATE_DISTILLATION_SCOPE,
                "entrypoint": identity(Path(__file__)),
                "protocol_sha256": self.driver.protocol["sha256"],
                "endpoint_plan": identity(self.bundle / "plan.json"),
                "preflight_audits": audits}

    def paired_report(self):
        """Recompute every score from the audited raw development outputs."""
        require(self.preflight() == self.binding, "Matched inputs changed during development execution")
        audits, arms = {}, {}
        for role, arm in STATE_DISTILLATION_ROLES.items():
            training = self.driver.art / ("train_" + arm)
            model = self.driver.art / ("merge_" + arm)
            evaluation = self.driver.art / ("dev_" + arm)
            training_audit = self.driver.train_verify(training, self.config["optimizer_steps"],
                lr=self.config["learning_rate"], **self.options(role))
            export_audit = self.driver.merge_verify(model)
            merge = jread(model / "merge_manifest.json")
            require(Path(merge["training_checkpoint"]) == training / f"checkpoint-{self.config['optimizer_steps']}",
                    "Development model belongs to another continuation arm")
            audits[arm] = self.driver.eval_verify(evaluation, "development", model)
            results = jread(evaluation / "task_results.json")
            require(identity(evaluation / "task_results.json") == audits[arm]["evaluation_identity"]["task_results.json"],
                    "Development outcomes changed after audit")
            episodes, per_task = [], {}
            for task in TASKS:
                outcomes = results[task]["results"]
                episodes.extend({"task": task, "episode_index": index, "success": outcome}
                                for index, outcome in enumerate(outcomes))
                per_task[task] = {"successes": sum(outcomes), "episodes": len(outcomes),
                                  "success_rate": sum(outcomes) / len(outcomes)}
            arms[arm] = {"role": role, "successes": audits[arm]["successes"],
                "episodes": episodes, "count": len(episodes), "per_task": per_task,
                "macro_success_rate": audits[arm]["macro_success_rate"],
                "micro_success_rate": audits[arm]["micro_success_rate"],
                "training_audit": training_audit, "export_audit": export_audit,
                "evaluation_audit": audits[arm], "evaluation_path": str(evaluation),
                "training_runtime": jread(training / "runtime_metrics.json")}
        require_pairing(audits)
        contrasts = {}
        for baseline, treatment in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])):
            left, right = arms[baseline]["episodes"], arms[treatment]["episodes"]
            contrasts[treatment + "_vs_" + baseline] = {"baseline": baseline, "treatment": treatment,
                **paired_summary(left, right), "per_task": {task: paired_summary(
                    [item for item in left if item["task"] == task],
                    [item for item in right if item["task"] == task]) for task in TASKS}}
        return {"schema": SCHEMA, "status": "complete", "scope": "development_only",
                "input_binding": self.binding, "fixed_arm_order": list(ARMS),
                "starting_qad_checkpoint": self.plan["qad"]["checkpoint"],
                "starting_qad_training_checkpoint": str(self.initial),
                "development_partition": self.driver.protocol["partitions"]["development"],
                "environment_pairing_verified": True, "selection_performed": False,
                "heldout_executed": False, "arms": arms, "contrasts": contrasts,
                "paired_scope": "Official environment reset identities; policy noise is seeded per task.",
                "interpretation": "Development evidence only. All three preregistered arms are reported without winner selection. "
                    "Exact McNemar p-values are exploratory; ten shared tasks and one training seed do not establish broader generalization."}

    def write_summary(self, destination):
        report = self.paired_report()
        jwrite(destination / "paired_comparison.json", report)
        return {"comparison_identity": identity(destination / "paired_comparison.json")}

    def verify_summary(self, destination):
        path = destination / "paired_comparison.json"
        require(jread(path) == self.paired_report(), "Development summary differs from recomputed source evidence")
        return {"comparison_identity": identity(path)}

    def run(self):
        if self.driver.a.validate_only:
            return {"schema": SCHEMA, "status": "validated_only", "scope": "development_only",
                    "fixed_arm_order": list(ARMS), "input_binding": self.binding}
        for offset, (role, arm) in enumerate(STATE_DISTILLATION_ROLES.items()):
            options = self.options(role)
            steps, lr = self.config["optimizer_steps"], self.config["learning_rate"]
            training = self.driver.stage("train_" + arm,
                lambda path, arm=arm, options=options, lr=lr, steps=steps: self.driver.train(
                    path, "train_" + arm, self.base, lr, steps, **options),
                lambda path, options=options, steps=steps, lr=lr: self.driver.train_verify(path, steps, lr=lr, **options))
            checkpoint = training / f"checkpoint-{steps}"

            def verify_export(path, checkpoint=checkpoint):
                report = self.driver.merge_verify(path)
                require(Path(jread(path / "merge_manifest.json")["training_checkpoint"]) == checkpoint,
                        "Export belongs to another continuation arm")
                return report

            model = self.driver.stage("merge_" + arm,
                lambda path, arm=arm, checkpoint=checkpoint, steps=steps: self.driver.merge(
                    path, "merge_" + arm, self.base, checkpoint, steps), verify_export)
            self.driver.stage("dev_" + arm,
                lambda path, arm=arm, model=model, offset=offset: self.driver.evaluate(
                    path, "dev_" + arm, model, "development", offset),
                lambda path, model=model: self.driver.eval_verify(path, "development", model))
        summary = self.driver.stage("paired_summary", self.write_summary, self.verify_summary)
        self.driver.state["status"] = "complete"
        self.driver.state["development_comparison"] = str(summary / "paired_comparison.json")
        self.driver.save()
        return jread(summary / "paired_comparison.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "protocol-file", "ptq-selection", "endpoint-bundle", "teacher-cache", "student-cache"):
        parser.add_argument("--" + name, required=True)
    for name in ("gr00t-repo", "python", "rollout-python", "dataset"):
        parser.add_argument("--" + name)
    parser.add_argument("--port-base", type=int, default=5890)
    parser.add_argument("--adopt-complete", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        result = StateDistillationDev(parse_args(argv)).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OrchestrationError, OSError, ValueError, KeyError) as exc:
        print("[state-distillation-dev] ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
