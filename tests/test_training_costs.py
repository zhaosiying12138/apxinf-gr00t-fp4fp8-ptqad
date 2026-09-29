"""CPU-only failures must not publish incomplete formal cost evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import shutil
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("training_costs", ROOT / "paper/collect_training_costs.py")
cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost)
PROTOCOL = json.loads((ROOT / "exp/recovery_protocol.json").read_text())
PROTOCOL_SHA = cost.identity(ROOT / "exp/recovery_protocol.json")["sha256"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def stage(root):
    folder = root / "train_qad"
    manifest = {"rank":32, "alpha":64, "scope":"head+lang_all", "architecture":cost.ARCH,
        "protocol_sha256":PROTOCOL_SHA, "optimizer_resumed":False, "initial_adapter":None,
        "micro_batch":1, "gradient_accumulation_steps":16, "effective_global_batch":16,
        "upstream_global_batch_size_cli":1, "compute_dtype":"bfloat16", "activation_checkpointing":{"blocks":52},
        "probe_weight":0, "probe_every":4, "trainable_parameters":32, "lora_linear_modules":1,
        "recovery_source_sha256":{n:cost.identity(ROOT/n)["sha256"] for n in cost.TRAIN_SOURCES}}
    peaks = [{"device_index":0, "device_name":"fixture only", "max_memory_allocated_bytes":100,
              "max_memory_reserved_bytes":120}]
    metrics = {"status":"completed", "error":None, "global_steps":500, "requested_optimizer_steps":500,
        "micro_batch":1, "gradient_accumulation_steps":16, "effective_global_batch":16,
        "compute_dtype":"bfloat16", "activation_checkpointing_enabled":True,
        "wall_seconds":12.5, "timing_scope":cost.TRAIN_SCOPE, "cuda_peak_memory":peaks, "cuda_peak_scope":"fixture",
        "parameter_storage_by_dtype":{"float32":{"tensors":2,"parameters":64,"bytes":256,"trainable_parameters":32}}}
    write(folder/"runtime_metrics.json", metrics)
    write(folder/"recovery_manifest.json", manifest)
    write(folder/"checkpoint-500/recovery_manifest.json", manifest)
    write(folder/"checkpoint-500/trainer_state.json", {"global_step":500,"max_steps":500})
    write(folder/"checkpoint-500/model.safetensors.index.json", {"weight_map":{"weight":"model.safetensors"}})
    return folder, metrics, manifest


def synthetic_package(root):
    """Synthetic metadata fixture, never a model/experiment or published result."""
    private = root/"private"; private.mkdir()
    base, teacher, student, round_dir = (private/n for n in ("ptq","bf16","qad","round"))
    idx = {"weight_map":{"weight":"model.safetensors"}}
    for folder in (base,teacher,student):
        folder.mkdir()
        write(folder/"config.json",{"fixture_only":True})
        write(folder/"statistics.json",{"fixture_only":True})
        write(folder/"model.safetensors.index.json",idx)
        (folder/"model.safetensors").write_bytes((folder.name+" opaque fixture").encode())
    write(base/"ptq_recipe.json",{"base":str(teacher)})
    qad_dir, qmetrics, qmanifest = stage(private)
    qmanifest.update({"base":str(base),"base_config_sha256":cost.identity(base/"config.json")["sha256"],
        "base_statistics_sha256":cost.identity(base/"statistics.json")["sha256"],
        "base_recipe_sha256":cost.identity(base/"ptq_recipe.json")["sha256"],"parameter_dtype":"torch.float32","seed":7})
    adapter=qad_dir/"checkpoint-500"
    mask={"horizon":16,"dimensions":7}
    evaluation=cost.evaluation_helpers(ROOT/"eval/run_recovery_eval.py")
    collection=round_dir/"collection";collection.mkdir(parents=True)
    results, sources, samples={},[],[]
    for index,task in enumerate(evaluation.TASKS):
        resets=[{"episode_index":i,"seed":110000+index*1000+i,"init_state_index":i+2,"settle_steps":10,
                 "initial_state_sha256":"a"*64,"restored_state_sha256":"b"*64,"init_state_bank_sha256":"c"*64} for i in range(2)]
        log=collection/(task+".log")
        log.write_text("".join("FP4VLA_EPISODE_RESET "+json.dumps(r)+"\n" for r in resets)+"results: ('fixture', [True, False])\n")
        results[task]=evaluation.parse_log(log)|{"returncode":0}
        obs=collection/"observations"/task;obs.mkdir(parents=True)
        path=obs/"sample_000000.pt";path.write_bytes(b"opaque observation fixture")
        write(obs/"capture_manifest.json",{"student_config_sha256":qmanifest["base_config_sha256"],
            "source_kind":"student_rollout","student_checkpoint":str(student),
            "student_statistics_sha256":qmanifest["base_statistics_sha256"],"action_mask":mask})
        write(obs/"capture_counts.json",{"saved":1,"per_task":{task:1},"calls":1})
        sources.append({"path":str(path),**cost.identity(path)})
        samples.append({"provenance":{"source_kind":"student_rollout","student_checkpoint":str(student),
            "student_statistics_sha256":qmanifest["base_statistics_sha256"]}})
    write(collection/"eval_manifest.json",{"purpose":"collection","checkpoint":str(student),"seed":110000,
        "episodes":2,"tasks":evaluation.TASKS,"init_state_indices":[2,3],"n_envs":1,
        "initial_state_protocol":PROTOCOL["initial_states"]["protocol"],"protocol_sha256":PROTOCOL_SHA})
    write(collection/"task_results.json",results)
    write(collection/"summary.json",{"tasks_complete":10,"total_episodes":20,"total_successes":10,
        "macro_success_rate":.5,"wall_seconds_including_server_loads":123.5})
    teacher_meta={"teacher":str(teacher),"teacher_config_sha256":cost.identity(teacher/"config.json")["sha256"],
        "teacher_statistics_sha256":cost.identity(teacher/"statistics.json")["sha256"],
        "teacher_weights":{"model.safetensors":cost.identity(teacher/"model.safetensors")},
        "source_kind":"student_rollout","requested_count":160,"count":10,"source_observation_files":sources,
        "action_mask":mask,"architecture":cost.ARCH,"model_dtype":"float32","autocast_dtype":"bfloat16","eval_mode":True,
        "labeling_implementation_sha256":cost.identity(ROOT/"rl/opd_probe_cache.py")["sha256"],
        "replay_implementation_sha256":cost.identity(ROOT/"rl/probe_distill.py")["sha256"],
        "timing_scope":cost.TEACHER_SCOPE,"elapsed_seconds":13.5,"cuda_peak_memory":qmetrics["cuda_peak_memory"],"cuda_peak_scope":"fixture"}
    write(round_dir/"teacher_probes.json",teacher_meta)
    (round_dir/"teacher_probes.pt").write_bytes(b"opaque mocked cache fixture")
    cache={"version":3,"metadata":teacher_meta,"samples":samples}
    for name in ("qad","continued_qad","qad_opd"):
        folder=qad_dir if name=="qad" else round_dir/name
        steps=500 if name=="qad" else 100
        manifest=copy.deepcopy(qmanifest);metrics=copy.deepcopy(qmetrics)
        metrics.update({"global_steps":steps,"requested_optimizer_steps":steps,"wall_seconds":20.0 if name=="qad_opd" else 12.5})
        if name!="qad": manifest["initial_adapter"]=str(adapter)
        if name=="qad_opd":
            manifest.update({"probe_weight":1.0,"probe_cache":str(round_dir/"teacher_probes.pt"),
                "probe_cache_sha256":cost.identity(round_dir/"teacher_probes.pt")["sha256"],"probe_action_mask":mask})
        write(folder/"runtime_metrics.json",metrics);write(folder/"recovery_manifest.json",manifest)
        checkpoint=folder/("checkpoint-"+str(steps))
        write(checkpoint/"recovery_manifest.json",manifest)
        write(checkpoint/"trainer_state.json",{"global_step":steps,"max_steps":steps})
        write(checkpoint/"model.safetensors.index.json",idx)
        (checkpoint/"model.safetensors").write_bytes((name+" adapter fixture").encode())
        merged=student if name=="qad" else round_dir/(name+"_merged");merged.mkdir(exist_ok=True)
        if name!="qad": (merged/"model.safetensors").write_bytes((name+" output fixture").encode())
        write(merged/"merge_manifest.json",{"status":"complete","base":str(base),"training_checkpoint":str(checkpoint),
            "rank":32,"alpha":64,"recovery_manifest":manifest,
            "recovery_manifest_sha256":cost.identity(checkpoint/"recovery_manifest.json")["sha256"],
            "base_weights":{"model.safetensors":cost.identity(base/"model.safetensors")},
            "training_weights":{"model.safetensors":cost.identity(checkpoint/"model.safetensors")},
            "output_weights":{"model.safetensors":cost.identity(merged/"model.safetensors")},"elapsed_seconds":1.5})
        if name!="qad":
            text="" if name!="qad_opd" else "".join(f"[opd] step={step} probe=0 mse=0.01 weight=1.0 microbatch=1\n" for step in (4,24,44,64,84) for _ in range(16))
            (round_dir/(name+".train.log")).write_text(text)
    out=root/"evidence"
    # The serialized tensor loader alone is mocked; identity checks and all
    # source, JSON, raw-log, byte-copy and public recomputation code are real.
    with patch.dict(sys.modules,{"torch":types.SimpleNamespace(load=lambda *a,**kw:cache)}):
        cost.collect(qad_dir,student,round_dir,out,ROOT/"exp/recovery_protocol.json")
    return out, private


class TrainingCostTest(unittest.TestCase):
    def audit(self, folder):
        return cost.audit_stage(folder,"qad",500,PROTOCOL,PROTOCOL_SHA,cost.Evidence())

    def test_completed_stage_uses_actual_time_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _, _ = stage(Path(tmp))
            row, _, _ = self.audit(folder)
            self.assertEqual(row["wall_seconds"],12.5)
            self.assertEqual(row["demonstration_window_draws"],8000)

    def test_incomplete_and_wrong_steps_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, metrics, _ = stage(Path(tmp))
            metrics["status"]="failed"
            write(folder/"runtime_metrics.json",metrics)
            with self.assertRaisesRegex(ValueError,"incomplete training"):
                self.audit(folder)
            metrics["status"]="completed";metrics["global_steps"]=499
            write(folder/"runtime_metrics.json",metrics)
            with self.assertRaisesRegex(ValueError,"optimizer budget"):
                self.audit(folder)

    def test_equal_looking_but_wrong_demo_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, metrics, manifest = stage(Path(tmp))
            metrics["gradient_accumulation_steps"]=manifest["gradient_accumulation_steps"]=8
            metrics["effective_global_batch"]=manifest["effective_global_batch"]=8
            write(folder/"runtime_metrics.json",metrics)
            write(folder/"recovery_manifest.json",manifest)
            write(folder/"checkpoint-500/recovery_manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"demonstration budget"):
                self.audit(folder)

    def test_producer_source_sha_is_not_just_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _, manifest = stage(Path(tmp))
            manifest["recovery_source_sha256"][cost.TRAIN_SOURCES[0]]="0"*64
            write(folder/"recovery_manifest.json",manifest)
            write(folder/"checkpoint-500/recovery_manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"producer source SHA"):
                self.audit(folder)

    def test_empty_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _, _ = stage(Path(tmp))
            write(folder/"checkpoint-500/model.safetensors.index.json", {"weight_map":{}})
            with self.assertRaisesRegex(ValueError,"checkpoint index"):
                self.audit(folder)

    def test_weight_inventory_and_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);path=folder/"model.safetensors";path.write_bytes(b"opaque fixture")
            index={"weight_map":{"w":"model.safetensors"}}
            records={"model.safetensors":cost.identity(path)}
            cost.verify_weight_files(folder,records,index)
            path.write_bytes(b"changed fixture")
            with self.assertRaisesRegex(ValueError,"weight identity"):
                cost.verify_weight_files(folder,records,index)
            with self.assertRaisesRegex(ValueError,"shard inventory"):
                cost.verify_weight_files(folder,{},index)

    def test_false_timing_scope_is_rejected(self):
        for scope in ("", "kernel-only excludes loading and saves"):
            with self.subTest(scope=scope), tempfile.TemporaryDirectory() as tmp:
                folder, metrics, _ = stage(Path(tmp))
                metrics["timing_scope"]=scope
                write(folder/"runtime_metrics.json",metrics)
                with self.assertRaisesRegex(ValueError,"timing scope"):
                    self.audit(folder)

    def test_invalid_parameter_storage_is_rejected(self):
        for key,value in (("bytes",-1),("bytes",255),("trainable_parameters",65),("tensors",0)):
            with self.subTest(key=key,value=value), tempfile.TemporaryDirectory() as tmp:
                folder, metrics, _ = stage(Path(tmp))
                metrics["parameter_storage_by_dtype"]["float32"][key]=value
                write(folder/"runtime_metrics.json",metrics)
                with self.assertRaisesRegex(ValueError,"[Dd]type storage"):
                    self.audit(folder)

    def test_teacher_cannot_use_another_students_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obs=root/"observations/task";obs.mkdir(parents=True)
            sample=obs/"sample_000000.pt";sample.write_bytes(b"opaque fixture")
            student=root/"qad";student.mkdir()
            metadata={"source_kind":"student_rollout", "requested_count":160, "count":1,
                "teacher_statistics_sha256":"a"*64, "action_mask":{"horizon":16,"dimensions":7},
                "source_observation_files":[{"path":str(sample),**cost.identity(sample)}]}
            captures={"task":{"files":[sample],"manifest":{"student_statistics_sha256":"a"*64,
                "action_mask":metadata["action_mask"]}}}
            cache={"version":3,"metadata":copy.deepcopy(metadata),"samples":[{"provenance":{
                "source_kind":"student_rollout","student_checkpoint":str(root/"different_student"),
                "student_statistics_sha256":"a"*64}}]}
            with self.assertRaisesRegex(ValueError,"Teacher source student"):
                cost.validate_teacher_source(metadata,cache,root/"observations",student,captures)
            cache["samples"][0]["provenance"]["student_checkpoint"]=str(student)
            self.assertEqual(cost.validate_teacher_source(metadata,cache,root/"observations",student,captures),{"task":1})
            cache["metadata"]["count"]=2
            with self.assertRaisesRegex(ValueError,"JSON metadata differ"):
                cost.validate_teacher_source(metadata,cache,root/"observations",student,captures)

    def test_byte_copy_keeps_original_and_combines_source_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"record.json";data=b'{ "actual": 1 }\n';source.write_bytes(data)
            evidence=cost.Evidence()
            evidence.add(source,"source/record.json","training")
            evidence.add(source,"source/record.json","teacher")
            row,copied=evidence.files["source/record.json"]
            self.assertEqual(copied,data)
            self.assertEqual(source.read_bytes(),data)
            self.assertEqual(row["role"],["teacher","training"])

    def test_public_verifier_runs_standalone_without_private_paths_or_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out,private=synthetic_package(root)
            standalone=root/"standalone.py";shutil.copy2(ROOT/"paper/collect_training_costs.py",standalone)
            private.rename(root/"private_hidden")
            # -I -S removes user PYTHONPATH/site-packages; cwd is outside repo.
            command=[sys.executable,"-I","-S",str(standalone),"--verify-published",str(out)]
            result=subprocess.run(command,cwd=root,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertFalse(json.loads(result.stdout)["private_tensors_opened"])
            # A repeated verification must not create unmapped bytecode files.
            self.assertEqual(subprocess.run(command,cwd=root,capture_output=True).returncode,0)
            verified=cost.verify_published(out)
            self.assertEqual(verified["training"]["qad_opd"]["logged_teacher_backward_passes"],80)
            self.assertEqual(verified["training"]["qad_opd"]["scheduled_teacher_backward_passes"],400)

    def test_public_verifier_rejects_summary_and_collection_identity_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,_=synthetic_package(Path(tmp))
            costs=json.loads((out/"costs.json").read_text())
            changed=copy.deepcopy(costs);changed["training"]["qad"]["wall_seconds"]=.01
            write(out/"costs.json",changed)
            with self.assertRaisesRegex(ValueError,"training costs disagree"):
                cost.verify_published(out)
            write(out/"costs.json",costs)
            manifest=json.loads((out/"evidence_manifest.json").read_text())
            row=next(r for r in manifest["files"] if r["published_path"]=="exports/qad/merge_manifest.json")
            row["original_absolute_path"]="/different/student/merge_manifest.json"
            write(out/"evidence_manifest.json",manifest)
            with self.assertRaisesRegex(ValueError,"not the initial QAD export"):
                cost.verify_published(out)

    def test_existing_output_is_refused_before_input_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cost,"Evidence",side_effect=AssertionError("must not inspect")), self.assertRaises(FileExistsError):
                cost.collect("missing","missing","missing",tmp,"missing")


if __name__=="__main__":
    unittest.main()
