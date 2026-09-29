#!/usr/bin/env python3
"""Audit completed formal recovery costs and byte-copy lightweight evidence.

Use the recovery Python (torch is needed only to mmap the teacher cache on CPU).
No GPU calls, model loads, checkpoint tensor loads, or experiment subprocesses.
Smoke is not a formal arm. Missing/incomplete evidence prevents publication.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from fractions import Fraction
import types
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCES = tuple("rl/" + name for name in (
    "lora_qad.py", "probe_distill.py", "lora_scope.py", "recovery_batch.py",
    "gr00t_runtime.py", "activation_checkpoint.py", "runtime_metrics.py"))
ARCH = {"language_layers": 16, "dit_layers": 32, "vl_layers": 4}
TRAIN_SCOPE = "script import through training and checkpoint serialization"
TEACHER_SCOPE = "checkpoint loading and teacher labeling, before cache serialization"
DTYPE_BYTES = {"bfloat16":2,"float16":2,"float32":4,"float64":8,"int8":1,"uint8":1,"bool":1,"int16":2,"int32":4,"int64":8}
SHARED_KEYS = ("base", "rank", "alpha", "scope", "base_config_sha256", "base_statistics_sha256", "base_recipe_sha256",
    "parameter_dtype", "compute_dtype", "seed", "micro_batch", "effective_global_batch", "gradient_accumulation_steps",
    "recovery_source_sha256", "protocol_sha256", "activation_checkpointing", "trainable_parameters", "lora_linear_modules")
EXCLUDED = ["initial QAD smoke", "OPD smoke", "calibration", "baking", "heldout evaluation", "collector audit"]

ACCOUNTING_NOTES = ['Training, collection and labeling retain their producer timing scopes; do not sum them as a measured end-to-end wall clock.', 'Teacher labeling elapsed includes checkpoint/input hashing, configuration and observation reading, model loading and labeling, but ends before cache serialization. Export elapsed is a producer field, not process end-to-end timing.', 'Training timing includes checkpoint saving; the three arms can have different save counts. This is measured stage cost, not a pure per-update compute benchmark.', 'Allocator peaks are process-local PyTorch measurements; collection peaks were not recorded and are null.', 'Both continuation manifests name the same adapter; current adapter bytes match the completed QAD export. Training did not record a separate initial-adapter weight digest at start.', 'Training and teacher source hashes are producer-attested and verified. Collection records protocol SHA but not its own source SHA; archived wrapper source is a collector-time snapshot.', 'Weights, observation tensors and teacher cache are hashed/checked but are not copied into this lightweight evidence package.', 'Demonstration window draws are derived from completed optimizer steps times configured effective batch, not an independent microbatch counter or a count of distinct samples.', 'Teacher backward fields count student backward passes against cached teacher velocity targets; the BF16 teacher labels under no_grad and is never updated.', 'Public verification checks copied metadata/logs and recorded identities without opening private model, observation or cache tensor files; the collector-time tensor hash checks cannot be independently repeated from this lightweight package.']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def is_orchestrated_protocol(protocol):
    """Recognize the orchestrator contract from its schema, not its version number.

    The protocol version is a provenance label. A later frozen round may keep
    the same partition/selection contract while incrementing that label; the
    evidence gate must still audit it instead of silently falling back to the
    old protocol.
    """
    partitions = protocol.get('partitions')
    selection = protocol.get('selection')
    return (isinstance(partitions, dict) and
            all(isinstance(partitions.get(name), dict)
                for name in ('development', 'collection', 'heldout')) and
            isinstance(selection, dict) and
            isinstance(selection.get('qad_learning_rates'), list) and
            isinstance(selection.get('opd_weights'), list) and
            isinstance(selection.get('pressure_candidates'), list))


def orchestrator_source(protocol, state=None):
    """Return the producer source named by a run, with a compatibility default.

    New run manifests should record ``implementation_path``. For the first
    published orchestrator that field was absent, so derive its v3 filename
    from the frozen protocol id; this does not select a different protocol.
    """
    state = state or {}
    relative = state.get('implementation_path')
    if relative is None:
        match = re.search(r'v(\d+)', str(protocol.get('id', '')))
        relative = 'exp/run_high_fp4_v%s.py' % (match.group(1) if match else '3')
    path = Path(relative)
    need(not path.is_absolute() and '..' not in path.parts and path.parts[:1] == ('exp',),
         'Invalid orchestrator implementation path')
    return path.as_posix()


def normalized_protocol(protocol, final=None):
    """Derive audit settings without rewriting the archived protocol bytes."""
    if not is_orchestrated_protocol(protocol):return protocol
    need(isinstance(final,dict),'v3 costs require the completed orchestrator final manifest')
    selection=protocol['selection'];lr=final['selected_qad_learning_rate'];weight=final['selected_opd_weight']
    need(lr in selection['qad_learning_rates'] and weight in selection['opd_weights'],
         'Selected recovery setting is outside the frozen v3 search')
    return {**protocol,**protocol['partitions'],
        'initial_states':{'protocol':protocol['initial_state_protocol']},
        'recovery':{'initial_qad_optimizer_steps':selection['qad_optimizer_steps'],
            'effective_demo_batch':selection['effective_demo_batch'],'rank':selection['rank'],
            'alpha':selection['alpha'],'scope':selection['recovery_scope'],'activation_checkpointing':True},
        'continuation':{'optimizer_steps':selection['continuation_optimizer_steps'],
            'effective_demo_batch':selection['effective_demo_batch'],'opd_weight':weight,
            'probe_every_optimizer_steps':selection['opd_every']},
        'selected_learning_rate':lr,'train_seed':selection['train_seed']}


def audit_recovery_selection(record,kind,protocol,protocol_sha,resolve,evaluation):
    """Recompute recovery choice from archived development raw logs, never heldout."""
    field='learning_rate' if kind=='qad' else 'opd_weight'
    declared=protocol['selection']['qad_learning_rates' if kind=='qad' else 'opd_weights']
    need(record.get('protocol_sha256')==protocol_sha and record.get('selection_uses_heldout') is False and
         record.get('environment_pairing_verified') is True,'Recovery choice lacks frozen development provenance')
    rows=list(record['candidates'].values())
    need(len(rows)==len(declared) and {row[field] for row in rows}==set(declared),'Recovery candidate set differs')
    expected=protocol['partitions']['development'];signatures=[]
    for row in rows:
        need(row.get('selection_source')=='development_only','Recovery choice uses non-development data')
        ids=row['evaluation_identity'];base=Path(row['evaluation_path'])
        need(set(ids)=={'eval_manifest.json','task_results.json','summary.json'},'Incomplete selection evaluation identity')
        for name,item in ids.items():need(item['path']==str(base/name),'Selection input path differs')
        manifest,tasks,summary=[json.loads(resolve(ids[name]).read_text()) for name in
                               ('eval_manifest.json','task_results.json','summary.json')]
        values={'purpose':'development','seed':expected['seed'],'episodes':expected['episodes_per_task'],
            'init_state_indices':expected['init_state_indices'],'tasks':evaluation.TASKS,
            'n_envs':1,'settle_steps':10,'n_action_steps':8,'max_episode_steps':720,
            'protocol_sha256':protocol_sha,'initial_state_protocol':'libero10_official_bank_v1'}
        need(all(manifest.get(k)==v for k,v in values.items()),'Selection evaluation protocol differs')
        need(set(tasks)==set(evaluation.TASKS) and set(row['raw_log_identities'])==set(evaluation.TASKS),'Selection task/log set incomplete')
        count=successes=0;signature=[]
        for ti,task in enumerate(evaluation.TASKS):
            result=tasks[task];outcomes=result['results'];log=row['raw_log_identities'][task]
            need(log['path']==str(base/(task+'.log')),'Selection raw log path differs')
            parsed=evaluation.parse_log(resolve(log))
            need(type(result.get('returncode')) is int and result['returncode']==0 and
                 len(outcomes)==expected['episodes_per_task'] and all(type(x) is bool for x in outcomes) and
                 all(result.get(k)==v for k,v in parsed.items()),'Selection task differs from raw log')
            need(result.get('seed')==expected['seed']+1000*ti,'Selection task seed differs')
            resets=evaluation.validate_resets(result,expected['seed']+1000*ti,expected['init_state_indices'])
            signature.extend({'task':task,**{k:z[k] for k in ('episode_index','seed','init_state_index','settle_steps',
                'initial_state_sha256','restored_state_sha256','init_state_bank_sha256')}} for z in resets)
            successes+=sum(outcomes);count+=len(outcomes)
        digest=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest();signatures.append(digest)
        need(row.get('pairing_sha256')==digest,'Selection pairing digest differs')
        need(summary.get('tasks_complete')==10 and summary.get('total_successes')==successes and
             summary.get('total_episodes')==count and summary.get('purpose')=='development' and
             summary.get('seed')==expected['seed'] and
             math.isclose(summary['macro_success_rate'],successes/count,rel_tol=0,abs_tol=1e-12),
             'Selection summary differs from raw outcomes')
        need(row.get('successes')==successes and row.get('episodes')==count and
             math.isclose(row['macro_success_rate'],successes/count,rel_tol=0,abs_tol=1e-12),
             'Selection score differs from raw outcomes')
    need(len(set(signatures))==1,'Recovery development candidate initial states do not pair')
    winner=sorted(rows,key=lambda row:(-Fraction(row['successes'],row['episodes']),row[field]))[0][field]
    need(record['selected_'+field]==winner,'Recovery selection violates frozen score/tie-break rule')
    return winner


def evaluation_helpers(source):
    """Load only the stdlib evaluation parser, lazily; public callers verify its bytes first."""
    source = Path(source).resolve(strict=True)
    module = types.ModuleType("training_cost_eval_parser")
    module.__file__ = str(source)
    # Do not create __pycache__ inside a byte-verified evidence tree.
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return module


def positive(value, label):
    need(type(value) in (int, float) and math.isfinite(value) and value > 0, "Invalid " + label)
    return value


def identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Input changed: " + str(path))
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


class Evidence:
    def __init__(self):
        self.files = {}

    def add(self, source, target, role):
        source = Path(source).resolve(strict=True)
        need(not Path(target).is_absolute() and ".." not in Path(target).parts, "Invalid archive path")
        need(source.stat().st_size < 64 * 1024 * 1024, "Not lightweight evidence: " + str(source))
        before = identity(source)
        data = source.read_bytes()
        need(before == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, "Input changed while reading")
        row = {"original_absolute_path": str(source), "published_path": target, **before, "role": role}
        if target in self.files:
            previous = self.files[target][0]
            need({k:v for k,v in previous.items() if k != "role"} ==
                 {k:v for k,v in row.items() if k != "role"}, "Conflicting archive destination")
            roles = previous["role"] if isinstance(previous["role"], list) else [previous["role"]]
            row["role"] = sorted(set([*roles, role]))
        self.files[target] = (row, data)
        return data

    def json(self, source, target, role="raw_metadata"):
        return json.loads(self.add(source, target, role))


def validate_peaks(rows):
    need(isinstance(rows, list) and rows, "Missing measured CUDA allocator peaks")
    for row in rows:
        need(type(row["device_index"]) is int and row["device_index"] >= 0 and row["device_name"], "Invalid peak device")
        a = positive(row["max_memory_allocated_bytes"], "allocated peak")
        b = positive(row["max_memory_reserved_bytes"], "reserved peak")
        need(type(a) is int and type(b) is int and b >= a, "Invalid allocator peak bytes")
    return rows


def index_shards(index):
    values = index.get("weight_map")
    need(isinstance(values, dict) and values and all(isinstance(k,str) and k for k in values), "Empty/invalid checkpoint index")
    names = set(values.values())
    need(all(isinstance(n,str) and Path(n).name == n and n.endswith(".safetensors") for n in names), "Invalid checkpoint shard names")
    return names


def verify_weight_files(folder, records, index):
    folder = Path(folder)
    need(index_shards(index) == set(records) == {p.name for p in folder.glob("*.safetensors")}, "Incomplete checkpoint shard inventory")
    for filename, row in records.items():
        need(identity(folder/filename) == row, "Checkpoint weight identity differs: " + str(folder/filename))


def validate_storage(storage):
    need(isinstance(storage, dict) and storage, "No measured parameter dtype inventory")
    for dtype, row in storage.items():
        need(dtype in DTYPE_BYTES and all(type(row.get(k)) is int and row[k] >= 0 for k in
             ("tensors", "parameters", "bytes", "trainable_parameters")), "Invalid dtype storage counters")
        need(0 < row["tensors"] <= row["parameters"] and row["trainable_parameters"] <= row["parameters"] and
             row["bytes"] == row["parameters"] * DTYPE_BYTES[dtype], "Dtype storage size/count differs")
    return storage


def audit_stage(folder, name, steps, protocol, protocol_sha, evidence, expected_initial=None,
                source_root=ROOT, checkpoint_subdir=None, logical_checkpoint=None):
    folder = Path(folder).resolve(strict=True)
    need("smoke" not in folder.name.lower(), "Smoke cannot be a formal training arm")
    prefix = "stages/" + name
    metrics = evidence.json(folder / "runtime_metrics.json", prefix + "/runtime_metrics.json")
    manifest = evidence.json(folder / "recovery_manifest.json", prefix + "/recovery_manifest.json")
    need(metrics.get("status") == "completed" and metrics.get("error") is None, name + ": incomplete training")
    need(metrics.get("global_steps") == steps and metrics.get("requested_optimizer_steps") == steps,
         name + ": optimizer budget differs from the formal protocol")
    checkpoint = folder / (checkpoint_subdir or ("checkpoint-" + str(steps)))
    saved = evidence.json(checkpoint / "recovery_manifest.json", prefix + "/checkpoint/recovery_manifest.json")
    state = evidence.json(checkpoint / "trainer_state.json", prefix + "/checkpoint/trainer_state.json")
    need(saved == manifest and state.get("global_step") == steps and state.get("max_steps") == steps,
         name + ": final checkpoint is missing or does not match completed training")
    index_shards(evidence.json(checkpoint / "model.safetensors.index.json", prefix + "/checkpoint/model.safetensors.index.json"))
    recovery = protocol["recovery"]
    for key in ("rank", "alpha", "scope"):
        need(manifest.get(key) == recovery[key], name + ": recovery " + key + " differs")
    need(manifest.get("architecture") == ARCH and manifest.get("protocol_sha256") == protocol_sha,
         name + ": model/protocol identity differs")
    need(manifest.get("optimizer_resumed") is False, name + ": optimizer was resumed")
    initial = manifest.get("initial_adapter")
    need((initial is None and expected_initial is None) or
         (initial is not None and expected_initial is not None and Path(initial).resolve() == expected_initial),
         name + ": initial adapter differs")
    for key in ("micro_batch", "gradient_accumulation_steps", "effective_global_batch"):
        need(type(manifest.get(key)) is int and manifest[key] > 0 and metrics.get(key) == manifest[key],
             name + ": batch metadata differs")
    need(manifest["micro_batch"] * manifest["gradient_accumulation_steps"] == manifest["effective_global_batch"] ==
         protocol["recovery" if name == "qad" else "continuation"]["effective_demo_batch"],
         name + ": effective demonstration budget differs")
    need(manifest.get("upstream_global_batch_size_cli") == manifest["micro_batch"], name + ": upstream batch semantics differ")
    need(metrics.get("compute_dtype") == manifest.get("compute_dtype") == "bfloat16", name + ": compute dtype differs")
    need(metrics.get("activation_checkpointing_enabled") is recovery["activation_checkpointing"] and
         bool(manifest.get("activation_checkpointing")) is recovery["activation_checkpointing"], name + ": checkpointing differs")
    expected_weight = protocol["continuation"]["opd_weight"] if name == "qad_opd" else 0
    need(manifest.get("probe_weight") == expected_weight, name + ": teacher objective differs")
    need(manifest.get("probe_every") == protocol["continuation"]["probe_every_optimizer_steps"], name + ": teacher schedule differs")
    if is_orchestrated_protocol(protocol):
        request=evidence.json(folder/'orchestrator_training_request.json',prefix+'/orchestrator_training_request.json')
        env=request['environment'];logical_folder=Path(logical_checkpoint).parent if logical_checkpoint else folder
        expected_env={'QAD_OUT':str(logical_folder),'QAD_STEPS':str(steps),
            'QAD_LR':str(float(protocol['selected_learning_rate'])),'TRAIN_SEED':str(protocol['train_seed']),
            'QAD_OPD_MSE_W':str(float(expected_weight)),'QAD_ACTIVATION_CHECKPOINTING':'1'}
        need(request.get('protocol_sha256')==protocol_sha and all(env.get(k)==v for k,v in expected_env.items()),
             name+': training request differs from selected v3 settings')
        need(manifest.get('train_seed')==manifest.get('seed')==protocol['train_seed'],name+': training seed differs')
    need(set(manifest.get("recovery_source_sha256", {})) == set(TRAIN_SOURCES), name + ": incomplete source SHA coverage")
    for relative, digest in manifest["recovery_source_sha256"].items():
        data = evidence.add(Path(source_root) / relative, "source/" + relative, "verified_training_producer")
        need(hashlib.sha256(data).hexdigest() == digest, name + ": producer source SHA differs: " + relative)
    storage = validate_storage(metrics.get("parameter_storage_by_dtype"))
    need(metrics.get("timing_scope") == TRAIN_SCOPE, name + ": unsupported training timing scope")
    need(sum(row["trainable_parameters"] for row in storage.values()) == manifest["trainable_parameters"] > 0,
         name + ": trainable parameter count differs")
    result = {"status": "completed", "optimizer_steps": steps,
              "demonstration_window_draws": steps * manifest["effective_global_batch"],
              "wall_seconds": positive(metrics["wall_seconds"], "training wall time"),
              "timing_scope": metrics["timing_scope"], "cuda_peak_memory": validate_peaks(metrics["cuda_peak_memory"]),
              "cuda_peak_scope": metrics["cuda_peak_scope"], "parameter_storage_by_dtype": storage,
              "compute_dtype": metrics["compute_dtype"], "micro_batch": manifest["micro_batch"],
              "gradient_accumulation_steps": manifest["gradient_accumulation_steps"],
              "effective_global_batch": manifest["effective_global_batch"],
              "trainable_parameters": manifest["trainable_parameters"], "lora_linear_modules": manifest["lora_linear_modules"],
              "checkpoint": str(logical_checkpoint or checkpoint), "initial_adapter": initial}
    return result, manifest, checkpoint


def audit_merge(folder, checkpoint, manifest, evidence, name, logical_checkpoint=None):
    path = Path(folder).resolve(strict=True)
    merge = evidence.json(path / "merge_manifest.json", "exports/" + name + "/merge_manifest.json")
    need(merge.get("status") == "complete" and Path(merge["training_checkpoint"]).resolve() == (logical_checkpoint or checkpoint) and
         Path(merge["base"]).resolve() == Path(manifest["base"]).resolve(), name + ": incomplete/wrong export")
    need(merge["recovery_manifest"] == manifest and merge["recovery_manifest_sha256"] ==
         identity(checkpoint / "recovery_manifest.json")["sha256"], name + ": export recovery identity differs")
    for key in ("rank", "alpha"):
        need(merge[key] == manifest[key], name + ": export scaling differs")
    index = json.loads((checkpoint/"model.safetensors.index.json").read_text())
    need(index_shards(index) == set(merge["training_weights"]), name + ": merge training shard index differs")
    return merge


def audit_collection(folder, qad_merged, protocol, protocol_sha, evidence, published=False, student_config_sha=None, evaluation=None):
    folder = Path(folder).resolve(strict=True)
    manifest = evidence.json(folder / "eval_manifest.json", "collection/eval_manifest.json")
    summary = evidence.json(folder / "summary.json", "collection/summary.json")
    tasks = evidence.json(folder / "task_results.json", "collection/task_results.json")
    expected = protocol["collection"]
    evaluation = evaluation or evaluation_helpers(ROOT/"eval/run_recovery_eval.py")
    TASKS = evaluation.TASKS
    need(manifest["purpose"] == "collection" and manifest["seed"] == expected["seed"] and
         manifest["episodes"] == expected["episodes_per_task"] and manifest["tasks"] == TASKS and
         manifest["init_state_indices"] == expected["init_state_indices"] and manifest["n_envs"] == 1 and
         manifest["initial_state_protocol"] == protocol["initial_states"]["protocol"] and
         manifest["protocol_sha256"] == protocol_sha and Path(manifest["checkpoint"]).resolve() == qad_merged,
         "Collection does not match QAD student/protocol")
    need(set(tasks) == set(TASKS) and summary["tasks_complete"] == 10, "Incomplete collection")
    total, success, captures = 0, 0, {}
    for index, task in enumerate(TASKS):
        row = tasks[task]
        need(row["returncode"] == 0 and row["episodes"] == len(row["results"]) == expected["episodes_per_task"] and
             all(type(v) is bool for v in row["results"]) and row["successes"] == sum(row["results"]), "Incomplete collection task: " + task)
        evaluation.validate_resets(row, expected["seed"] + 1000 * index, expected["init_state_indices"])
        raw = evaluation.parse_log(folder / (task + ".log"))
        need(all(raw[k] == row[k] for k in raw), "Collection raw log disagrees: " + task)
        evidence.add(folder / (task + ".log"), "collection/" + task + ".log", "raw_collection_log")
        total += row["episodes"]; success += row["successes"]
        obs = folder / "observations" / task
        capture = evidence.json(obs / "capture_manifest.json", "collection/observations/" + task + "/capture_manifest.json")
        counts = evidence.json(obs / "capture_counts.json", "collection/observations/" + task + "/capture_counts.json")
        need(capture["student_config_sha256"] == (student_config_sha or identity(qad_merged/"config.json")["sha256"]) and
             capture["source_kind"] == "student_rollout" and Path(capture["student_checkpoint"]).resolve() == qad_merged,
             "Capture uses another student")
        files = sorted(obs.glob("sample_*.pt")) if not published else None
        need(type(counts["saved"]) is int and counts["saved"] > 0 and
             all(type(v) is int and v > 0 for v in counts["per_task"].values()) and
             counts["saved"] == sum(counts["per_task"].values()), "Capture count differs")
        if not published:
            need(counts["saved"] == len(files), "Capture file count differs")
        captures[task] = {"manifest": capture, "counts": counts, "files": files}
    need(summary["total_episodes"] == total == 10*expected['episodes_per_task'] and summary["total_successes"] == success and
         abs(summary["macro_success_rate"] - success/total) < 1e-12, "Collection summary differs")
    return {"wall_seconds": positive(summary["wall_seconds_including_server_loads"], "collection wall time"),
            "timing_scope": "serial ten-task collection loop including server loads, rollouts and server shutdown",
            "episodes": total, "tasks": 10, "captured_observations": sum(v["counts"]["saved"] for v in captures.values()),
            "cuda_peak_memory": None, "cuda_peak_note": "Collection wrapper does not record allocator peaks"}, captures


def validate_teacher_source(metadata, cache, observations_root, student, captures):
    need(cache.get("version") == 3 and cache.get("metadata") == metadata, "Teacher cache and JSON metadata differ")
    samples = cache.get("samples", [])
    need(metadata.get("source_kind") == "student_rollout" and metadata.get("requested_count") == 160 and
         type(metadata.get("count")) is int and 0 < metadata["count"] <= 160 and
         metadata["count"] == len(samples) == len(metadata["source_observation_files"]), "Incomplete/non-on-policy teacher cache")
    seen, used_tasks = set(), Counter()
    for record, sample in zip(metadata["source_observation_files"], samples):
        path = Path(record["path"]).resolve(strict=True)
        need(path.is_relative_to(observations_root) and path.parent.name in captures and path not in seen,
             "Teacher source observations are not unique members of this collection")
        need(path in captures[path.parent.name]["files"] and identity(path) == {k:record[k] for k in ("bytes", "sha256")},
             "Teacher source observation identity differs")
        capture = captures[path.parent.name]["manifest"]
        provenance = sample["provenance"]
        need(provenance.get("source_kind") == "student_rollout" and
             Path(provenance["student_checkpoint"]).resolve() == student and
             provenance["student_statistics_sha256"] == metadata["teacher_statistics_sha256"] == capture["student_statistics_sha256"],
             "Teacher source student/statistics identity differs")
        need(capture["action_mask"] == metadata["action_mask"], "Teacher/capture action masks differ")
        seen.add(path); used_tasks[path.parent.name] += 1
    return dict(used_tasks)


def probe_cost_fields(text, name, steps, opd):
    every, accumulation = opd["probe_every"], opd["gradient_accumulation_steps"]
    records = re.findall(r"\[opd\] step=(\d+) probe=(\d+) mse=(\S+) weight=(\S+) microbatch=1", text)
    expected = Counter({step:accumulation for step in range(1, steps+1)
        if step % every == 0 and (step-1) % max(every, 20) == every-1}) if name == "qad_opd" else Counter()
    need(Counter(int(row[0]) for row in records) == expected, "Formal teacher log schedule differs: " + name)
    need(all(math.isfinite(float(row[2])) and float(row[2]) >= 0 and float(row[3]) == opd["probe_weight"] for row in records), "Nonfinite or wrong teacher loss log")
    return {"logged_teacher_backward_passes": len(records),
            "scheduled_teacher_backward_passes": (steps//every)*accumulation if name == "qad_opd" else 0,
            "teacher_schedule_note": "Scheduled count derives from completed updates and verified source; logger records only selected updates, not all probe calls."}


def orchestrator_layout(run_dir,protocol,protocol_sha,evidence):
    run=Path(run_dir).resolve(strict=True)
    final=evidence.json(run/'final_manifest.json','orchestration/final_manifest.json','completed_orchestrator')
    state=evidence.json(run/'run_manifest.json','orchestration/run_manifest.json','completed_orchestrator')
    need(isinstance(final.get('format'), str) and re.fullmatch(r'high_fp4_[a-z0-9_]+_final_manifest', final['format']) and
         state.get('status')=='complete' and
         state.get('output_layout')=='stable_paths_v2' and
         final.get('protocol_sha256')==state.get('protocol_sha256')==protocol_sha and
         final.get('selection_uses_heldout') is False and state.get('selection_uses_heldout') is False,
         'Orchestrator is incomplete or uses another protocol')
    need(final.get('required_arms')==['bf16','ptq','qad','continued_qad','qad_opd'],'Orchestrator final arm set differs')
    implementation_relative = orchestrator_source(protocol, state)
    need(state.get('implementation_sha256')==identity(ROOT/implementation_relative)['sha256'],
         'Orchestrator source differs from completed run')
    def resolve(record):
        path=Path(record['path']).resolve(strict=True)
        need(identity(path)=={k:record[k] for k in ('bytes','sha256')},'Orchestrator source identity differs')
        target='orchestration/inputs/'+hashlib.sha256(record['path'].encode()).hexdigest()[:20]+'-'+path.name
        evidence.add(path,target,'recovery_selection_input');return path
    evaluation=evaluation_helpers(ROOT/'eval/run_recovery_eval.py')
    lr=audit_recovery_selection(final['qad_selection'],'qad',protocol,protocol_sha,resolve,evaluation)
    weight=audit_recovery_selection(final['opd_selection'],'opd',protocol,protocol_sha,resolve,evaluation)
    need(lr==final['selected_qad_learning_rate'] and weight==final['selected_opd_weight'],'Final settings differ from selected recovery')
    comparison=json.loads(resolve(final['heldout_comparison']).read_text())
    need(comparison.get('environment_pairing_verified') is True and set(comparison['arms'])==set(final['required_arms']) and
         all(row['count']==100 for row in comparison['arms'].values()),'Orchestrator final evaluation is incomplete')
    merged={name:Path(final[key]['path']).resolve(strict=True) for name,key in (
        ('qad','selected_qad_model_identity'),('continued_qad','selected_continued_model_identity'),('qad_opd','selected_opd_model_identity'))}
    training={};logs={};round_dir=Path(final['heldout_round']).resolve(strict=True)
    for name,folder in merged.items():
        export=json.loads((folder/'merge_manifest.json').read_text());training[name]=Path(export['training_checkpoint']).parent.resolve(strict=True)
        matches=[(key,row) for key,row in state['stages'].items() if row.get('output')==str(training[name])]
        need(len(matches)==1,'Selected training stage missing or ambiguous')
        key,marker=matches[0]
        actual=evidence.json(run/'stages'/(key+'.json'),'orchestration/stages/'+key+'.json','completed_stage')
        need(actual==marker and marker.get('status')=='complete' and marker.get('protocol_sha256')==protocol_sha,
             'Selected training stage receipt differs')
        logs[name]=run/'logs'/(key+'.log')
        heldout=json.loads((round_dir/('heldout_'+name)/'eval_manifest.json').read_text())
        need(Path(heldout['checkpoint']).resolve()==folder,'Final heldout uses another recovery export')
        key={'qad':'selected_qad_model_identity','continued_qad':'selected_continued_model_identity',
             'qad_opd':'selected_opd_model_identity'}[name]
        model=final[key]
        need(export['output_weights']=={row['name']:{k:row[k] for k in ('bytes','sha256')} for row in model['shards']},
             'Final model shards differ from selected export')
    need(Path(final['selected_qad_checkpoint_identity']['path']).resolve()==
         training['qad']/('checkpoint-'+str(protocol['selection']['qad_optimizer_steps'])),'Selected QAD adapter differs')
    heldout=evidence.json(round_dir/'heldout_bf16/eval_manifest.json','orchestration/heldout_bf16_manifest.json')
    collection=Path(heldout['collection_manifest']).parent.resolve(strict=True)
    opd=json.loads((training['qad_opd']/'recovery_manifest.json').read_text())
    cache=Path(opd['probe_cache']).resolve(strict=True)
    need(cache.name=='teacher_probes.pt','Unexpected orchestrator teacher cache filename')
    evidence.add(ROOT/implementation_relative,'source/'+implementation_relative,'collector_time_orchestrator_snapshot')
    # The orchestrator validates teacher probes through this CPU verifier. Its
    # bytes are part of the producer provenance even though the cost collector
    # never imports or executes it.
    verifier = ROOT/'exp/verify_teacher_cache_cpu.py'
    if verifier.is_file():
        evidence.add(verifier,'source/exp/verify_teacher_cache_cpu.py','collector_time_orchestrator_dependency')
    return {'final':final,'training':training,'merged':merged,'logs':logs,'collection':collection,'teacher':cache.parent}


def collect(qad_dir, qad_merged, round_dir, out, protocol_path, qad_log=None,orchestrator_run=None):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Refusing existing evidence directory: " + str(out))
    evidence = Evidence()
    raw_protocol = evidence.json(protocol_path, "protocol/recovery_protocol.json", "fixed_protocol")
    protocol_sha = identity(protocol_path)["sha256"]
    layout=orchestrator_layout(orchestrator_run,raw_protocol,protocol_sha,evidence) if orchestrator_run else None
    protocol=normalized_protocol(raw_protocol,layout['final'] if layout else None)
    if layout:
        train_dirs=layout['training'];merged_dirs=layout['merged'];log_paths=layout['logs']
        qad_dir=train_dirs['qad'];qad_merged=merged_dirs['qad'];qad_log=log_paths['qad']
        collection_dir=layout['collection'];teacher_dir=layout['teacher']
        round_dir=Path(layout['final']['heldout_round'])
    else:
        round_dir, qad_merged = Path(round_dir).resolve(strict=True), Path(qad_merged).resolve(strict=True)
        train_dirs={name:round_dir/name for name in ('continued_qad','qad_opd')}
        merged_dirs={name:round_dir/(name+'_merged') for name in ('continued_qad','qad_opd')}
        log_paths={name:round_dir/(name+'.train.log') for name in ('continued_qad','qad_opd')}
        collection_dir=round_dir/'collection';teacher_dir=round_dir
    initial_steps = protocol["recovery"]["initial_qad_optimizer_steps"]
    continuation_steps = protocol["continuation"]["optimizer_steps"]
    qad, initial, adapter = audit_stage(qad_dir, "qad", initial_steps, protocol, protocol_sha, evidence)
    stages, manifests, exports = {"qad": qad}, {"qad": initial}, {}
    exports["qad"] = audit_merge(qad_merged, adapter, initial, evidence, "qad")
    for name in ("continued_qad", "qad_opd"):
        row, manifest, checkpoint = audit_stage(train_dirs[name], name, continuation_steps, protocol, protocol_sha, evidence, adapter)
        for key in SHARED_KEYS:
            need(manifest[key] == initial[key], "Continuation origin/budget differs: " + name + "/" + key)
        stages[name], manifests[name] = row, manifest
        exports[name] = audit_merge(merged_dirs[name], checkpoint, manifest, evidence, name)
        evidence.add(log_paths[name], "stages/"+name+"/train.log", "raw_training_log")
    if qad_log:
        evidence.add(qad_log, "stages/qad/train.log", "raw_training_log")
    base = Path(initial["base"]).resolve(strict=True)
    for field, filename in (("base_config_sha256", "config.json"), ("base_statistics_sha256", "statistics.json"), ("base_recipe_sha256", "ptq_recipe.json")):
        evidence.add(base/filename, "base/"+filename, "verified_base_metadata")
        need(initial[field] == identity(base/filename)["sha256"], "Training base metadata changed")
    # Close the merge/base/collection-student identities before copying evidence.
    base_index = evidence.json(base/"model.safetensors.index.json", "base/model.safetensors.index.json")
    verify_weight_files(base, exports["qad"]["base_weights"], base_index)
    adapter_weights = exports["qad"]["training_weights"]
    for name, export in exports.items():
        need(export["base_weights"] == exports["qad"]["base_weights"], "Three exports do not share the same frozen base bytes")
        checkpoint = Path(stages[name]["checkpoint"])
        verify_weight_files(checkpoint, export["training_weights"], json.loads((checkpoint/"model.safetensors.index.json").read_text()))
    student_index = evidence.json(qad_merged/"model.safetensors.index.json", "exports/qad/model.safetensors.index.json")
    verify_weight_files(qad_merged, exports["qad"]["output_weights"], student_index)
    for filename in ("config.json", "statistics.json"):
        evidence.add(qad_merged/filename, "exports/qad/"+filename, "verified_collection_student_metadata")
        need(identity(qad_merged/filename) == identity(base/filename), "Collection student metadata differs from frozen base")
    collection, captures = audit_collection(collection_dir, qad_merged, protocol, protocol_sha, evidence)
    for filename, target in (("teacher_labeling.log", "teacher/labeling.log"), ("collection.log", "collection/driver.log")):
        if (round_dir/filename).is_file():
            evidence.add(round_dir/filename, target, "raw_stage_driver_log")
    metadata = evidence.json(teacher_dir/"teacher_probes.json", "teacher/teacher_probes.json")
    cache_path = teacher_dir/"teacher_probes.pt"
    cache_identity = identity(cache_path)
    opd = manifests["qad_opd"]
    need(Path(opd["probe_cache"]).resolve() == cache_path and opd["probe_cache_sha256"] == cache_identity["sha256"], "OPD trained against another teacher cache")
    need(opd["probe_action_mask"] == metadata["action_mask"] and metadata["architecture"] == ARCH and
         metadata["action_mask"]["horizon"] == 16 and metadata["action_mask"]["dimensions"] == 7,
         "Teacher/student model or mask differs")
    need(metadata["model_dtype"] == opd["parameter_dtype"].removeprefix("torch.") and
         metadata["autocast_dtype"] == opd["compute_dtype"] and metadata["eval_mode"] is True,
         "Teacher/student precision or mode differs")
    for key, relative in (("labeling_implementation_sha256", "rl/opd_probe_cache.py"), ("replay_implementation_sha256", "rl/probe_distill.py")):
        data = evidence.add(ROOT/relative, "source/"+relative, "verified_teacher_producer")
        need(hashlib.sha256(data).hexdigest() == metadata[key], "Teacher producer source SHA differs")
    teacher = Path(metadata["teacher"]).resolve(strict=True)
    recipe = json.loads((base/"ptq_recipe.json").read_text())
    need(teacher == Path(recipe["base"]).resolve(), "Teacher is not the original BF16 source of the PTQ base")
    for field, filename in (("teacher_config_sha256", "config.json"), ("teacher_statistics_sha256", "statistics.json")):
        evidence.add(teacher/filename, "teacher/base_"+filename, "verified_teacher_metadata")
        need(metadata[field] == identity(teacher/filename)["sha256"], "Teacher metadata changed")
    need(metadata["teacher_statistics_sha256"] == initial["base_statistics_sha256"], "Teacher and frozen base statistics differ")
    teacher_index = evidence.json(teacher/"model.safetensors.index.json", "teacher/model.safetensors.index.json")
    verify_weight_files(teacher, metadata["teacher_weights"], teacher_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    used_tasks = validate_teacher_source(metadata, cache, (collection_dir/"observations").resolve(), qad_merged, captures)
    del cache
    need(set(used_tasks) == set(evaluation_helpers(ROOT/"eval/run_recovery_eval.py").TASKS), "Formal teacher cache does not cover all ten collection tasks")
    need(identity(cache_path) == cache_identity, "Teacher cache changed during audit")
    need(metadata.get("timing_scope") == TEACHER_SCOPE, "Unsupported teacher labeling timing scope")
    teacher_cost = {"elapsed_seconds": positive(metadata["elapsed_seconds"], "teacher wall time"),
                    "timing_scope": metadata["timing_scope"], "cuda_peak_memory": validate_peaks(metadata["cuda_peak_memory"]),
                    "cuda_peak_scope": metadata["cuda_peak_scope"], "model_dtype": metadata["model_dtype"],
                    "autocast_dtype": metadata["autocast_dtype"], "requested_probes": metadata["requested_count"],
                    "actual_probes": metadata["count"], "source_observations_per_task": used_tasks,
                    "cache_identity": cache_identity}
    if layout:
        need(metadata['count']==metadata['requested_count']==160 and
             all(used_tasks[task]==16 for task in used_tasks),
             'v3 requires all 160 declared teacher observations, 16 per task')
    for name in ("continued_qad", "qad_opd"):
        text = evidence.files["stages/"+name+"/train.log"][1].decode(errors="replace")
        stages[name].update(probe_cost_fields(text, name, continuation_steps, opd))
    # The wrapper does not attest its own source SHA at runtime; archive a snapshot
    # without mislabelling it as producer-attested execution evidence.
    for relative in ("eval/run_recovery_eval.py", "rl/capture_onpolicy.py", "rl/run_onpolicy_round.sh", "paper/collect_training_costs.py"):
        evidence.add(ROOT/relative, "source/"+relative, "collector_time_source_snapshot")
    result = {"version": 1, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha, "training": stages, "collection": collection, "teacher_labeling": teacher_cost,
        "export_reported_seconds": {name: positive(value["elapsed_seconds"], "merge elapsed") for name,value in exports.items()},
        "comparison": {"same_initial_adapter": str(adapter), "initial_adapter_weight_files": adapter_weights,
                       "same_frozen_base": str(base), "same_demo_and_update_budget": True,
                       "continuation_wall_ratio_opd_over_continued": stages["qad_opd"]["wall_seconds"]/stages["continued_qad"]["wall_seconds"],
                       "equal_total_compute": False},
        "excluded_from_formal_cost_comparison": EXCLUDED,
        "accounting_notes": ACCOUNTING_NOTES,
        "evidence_manifest": "evidence_manifest.json"}
    if layout:
        result['selected_recovery_settings']={'learning_rate':protocol['selected_learning_rate'],
            'opd_weight':protocol['continuation']['opd_weight'],'train_seed':protocol['train_seed']}
        result['cost_scope']='Selected final three training arms. Hyperparameter-search training and development evaluations are separate and excluded from these stage costs.'
    # All audit checks finish before creating the output. Existing paths are never reused.
    out.mkdir(parents=True, exist_ok=False)
    for row, data in evidence.files.values():
        target = out/row["published_path"]; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream: stream.write(data)
        need(identity(target) == {k:row[k] for k in ("bytes", "sha256")}, "Archive copy identity differs")
    mapping = {"status":"complete", "files":[row for row,_ in evidence.files.values()],
               "omitted_large_artifacts":"model weights, observation tensors and teacher cache; identities remain in raw metadata"}
    (out/"evidence_manifest.json").write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+"\n")
    (out/"costs.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def verify_published(directory):
    """Recompute a lightweight cost package without original/private paths.

    Return the verified costs dict. Tensor bytes are not re-opened here; their
    recorded identities are checked for internal consistency and bound by the
    publication guard to runtime/closed-loop artifacts.
    """
    root = Path(directory).resolve(strict=True)
    mapping = json.loads((root/"evidence_manifest.json").read_text())
    costs = json.loads((root/"costs.json").read_text())
    need(mapping.get("status") == costs.get("status") == "complete" and costs.get("version") == 1, "Incomplete published costs")
    records = mapping.get("files", [])
    need(isinstance(records,list) and records, "Empty cost evidence map")
    indexed = {}
    for row in records:
        relative = Path(row["published_path"])
        need(not relative.is_absolute() and ".." not in relative.parts and relative.as_posix() not in indexed, "Invalid/duplicate cost evidence path")
        path = (root/relative).resolve(strict=True)
        need(path.is_relative_to(root) and path.is_file() and Path(row["original_absolute_path"]).is_absolute(), "Invalid cost evidence source path")
        need(identity(path) == {k:row[k] for k in ("bytes","sha256")}, "Published evidence hash differs: " + str(relative))
        indexed[relative.as_posix()] = row
    need({str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} ==
         set(indexed) | {"costs.json","evidence_manifest.json"}, "Unmapped files in published costs")
    read = lambda relative: json.loads((root/relative).read_text())
    raw_protocol = read("protocol/recovery_protocol.json")
    protocol_sha = identity(root/"protocol/recovery_protocol.json")["sha256"]
    need(costs["protocol_sha256"] == protocol_sha, "Published cost protocol differs")
    final=None
    if is_orchestrated_protocol(raw_protocol):
        final=read('orchestration/final_manifest.json');state=read('orchestration/run_manifest.json')
        need(isinstance(final.get('format'), str) and re.fullmatch(r'high_fp4_[a-z0-9_]+_final_manifest', final['format']) and
             state.get('status')=='complete' and
             state.get('output_layout')=='stable_paths_v2' and
             final.get('protocol_sha256')==state.get('protocol_sha256')==protocol_sha and
             final.get('selection_uses_heldout') is False and state.get('selection_uses_heldout') is False,
             'Published orchestrator is incomplete or uses another protocol')
        def resolve(record):
            matches=[row for row in indexed.values() if row['original_absolute_path']==record['path'] and
                     all(row[k]==record[k] for k in ('bytes','sha256'))]
            need(bool(matches),'Missing/mismatched published selection input: '+record['path'])
            return root/matches[0]['published_path']
        evaluation=evaluation_helpers(root/'source/eval/run_recovery_eval.py')
        lr=audit_recovery_selection(final['qad_selection'],'qad',raw_protocol,protocol_sha,resolve,evaluation)
        weight=audit_recovery_selection(final['opd_selection'],'opd',raw_protocol,protocol_sha,resolve,evaluation)
        need(lr==final['selected_qad_learning_rate'] and weight==final['selected_opd_weight'],
             'Published final settings differ from development selection')
        implementation_relative = orchestrator_source(raw_protocol, state)
        need(identity(root/('source/'+implementation_relative))['sha256']==state['implementation_sha256'],
             'Published orchestrator source differs from completed run')
    protocol=normalized_protocol(raw_protocol,final)
    if final:
        need(costs.get('selected_recovery_settings')=={'learning_rate':protocol['selected_learning_rate'],
             'opd_weight':protocol['continuation']['opd_weight'],'train_seed':protocol['train_seed']},
             'Published selected recovery settings differ')
        need(costs.get('cost_scope')=='Selected final three training arms. Hyperparameter-search training and development evaluations are separate and excluded from these stage costs.',
             'Published v3 cost scope differs')
    evidence = Evidence()
    exports = {n:read("exports/"+n+"/merge_manifest.json") for n in ("qad","continued_qad","qad_opd")}
    adapter = Path(exports["qad"]["training_checkpoint"]).resolve()
    computed, manifests = {}, {}
    need(set(costs["training"]) == set(exports), "Published formal arm set differs")
    for name in exports:
        steps = protocol["recovery"]["initial_qad_optimizer_steps"] if name=="qad" else protocol["continuation"]["optimizer_steps"]
        logical = Path(exports[name]["training_checkpoint"]).resolve()
        row, manifest, checkpoint = audit_stage(root/"stages"/name,name,steps,protocol,protocol_sha,evidence,
            expected_initial=None if name=="qad" else adapter, source_root=root/"source",
            checkpoint_subdir="checkpoint",logical_checkpoint=logical)
        audit_merge(root/"exports"/name,checkpoint,manifest,evidence,name,logical_checkpoint=logical)
        computed[name], manifests[name] = row, manifest
    initial, opd = manifests["qad"], manifests["qad_opd"]
    if final:
        need(initial['base']==final['selected_pressure_checkpoint'] and
             str(adapter)==final['selected_qad_checkpoint_identity']['path'],
             'Published selected PTQ/adapter differs from final manifest')
        for arm,key in (('qad','selected_qad_model_identity'),('continued_qad','selected_continued_model_identity'),
                        ('qad_opd','selected_opd_model_identity')):
            record=final[key];weights={row['name']:{k:row[k] for k in ('bytes','sha256')} for row in record['shards']}
            need(exports[arm]['output_weights']==weights and Path(indexed['exports/'+arm+'/merge_manifest.json']['original_absolute_path']).parent==Path(record['path']),
                 'Published selected model identity differs: '+arm)
    for name in ("continued_qad","qad_opd"):
        need(all(manifests[name][k] == initial[k] for k in SHARED_KEYS), "Published continuation origin/budget differs")
        computed[name].update(probe_cost_fields((root/"stages"/name/"train.log").read_text(errors="replace"),name,
                             protocol["continuation"]["optimizer_steps"],opd))
    need(computed == costs["training"], "Published training costs disagree with raw metadata/logs")
    base_weights = exports["qad"]["base_weights"]
    need(all(export["base_weights"] == base_weights for export in exports.values()), "Published merge base weights differ")
    need(index_shards(read("base/model.safetensors.index.json")) == set(base_weights), "Published base shard inventory differs")
    need(index_shards(read("exports/qad/model.safetensors.index.json")) == set(exports["qad"]["output_weights"]), "Published collection student shard inventory differs")
    for export in exports.values():
        for key in ("base_weights","training_weights","output_weights"):
            need(export[key] and all(type(v.get("bytes")) is int and v["bytes"]>0 and re.fullmatch(r"[0-9a-f]{64}",v.get("sha256",""))
                                    for v in export[key].values()), "Malformed recorded tensor identity")
    for field,filename in (("base_config_sha256","config.json"),("base_statistics_sha256","statistics.json"),("base_recipe_sha256","ptq_recipe.json")):
        need(initial[field] == identity(root/"base"/filename)["sha256"], "Published base metadata differs")
    for filename in ("config.json","statistics.json"):
        need(identity(root/"base"/filename) == identity(root/"exports/qad"/filename), "Published student metadata differs")
    collection_manifest = read("collection/eval_manifest.json")
    student = Path(collection_manifest["checkpoint"]).resolve()
    need(student == Path(indexed["exports/qad/merge_manifest.json"]["original_absolute_path"]).parent,
         "Published collection student is not the initial QAD export")
    evaluation = evaluation_helpers(root/"source/eval/run_recovery_eval.py")
    collection, captures = audit_collection(root/"collection",student,protocol,protocol_sha,evidence,published=True,
                                            student_config_sha=initial["base_config_sha256"], evaluation=evaluation)
    need(collection == costs["collection"], "Published collection costs disagree with raw metadata/logs")
    metadata = read("teacher/teacher_probes.json")
    need(metadata.get("timing_scope") == TEACHER_SCOPE and metadata.get("source_kind") == "student_rollout" and
         metadata.get("requested_count") == 160 and 0 < metadata.get("count",0) <= 160 and
         metadata["count"] == len(metadata["source_observation_files"]), "Invalid published teacher metadata")
    need(opd["probe_action_mask"] == metadata["action_mask"] and metadata["architecture"] == ARCH and
         metadata["action_mask"]["horizon"] == 16 and metadata["action_mask"]["dimensions"] == 7 and
         metadata["model_dtype"] == opd["parameter_dtype"].removeprefix("torch.") and
         metadata["autocast_dtype"] == opd["compute_dtype"] and metadata["eval_mode"] is True, "Published teacher/student semantics differ")
    for key,relative in (("labeling_implementation_sha256","rl/opd_probe_cache.py"),("replay_implementation_sha256","rl/probe_distill.py")):
        need(metadata[key] == identity(root/"source"/relative)["sha256"], "Published teacher source SHA differs")
    for field,filename in (("teacher_config_sha256","config.json"),("teacher_statistics_sha256","statistics.json")):
        need(metadata[field] == identity(root/"teacher"/("base_"+filename))["sha256"], "Published teacher metadata identity differs")
    need(metadata["teacher_statistics_sha256"] == initial["base_statistics_sha256"] and
         Path(metadata["teacher"]).resolve() == Path(read("base/ptq_recipe.json")["base"]).resolve(), "Published teacher origin differs")
    need(index_shards(read("teacher/model.safetensors.index.json")) == set(metadata["teacher_weights"]),
         "Published teacher shard inventory differs")
    need(all(type(v.get("bytes")) is int and v["bytes"] > 0 and re.fullmatch(r"[0-9a-f]{64}", v.get("sha256", ""))
             for v in metadata["teacher_weights"].values()), "Malformed teacher weight identity")
    original_collection = Path(indexed["collection/eval_manifest.json"]["original_absolute_path"]).parent
    seen, used = set(), Counter()
    for record in metadata["source_observation_files"]:
        path = Path(record["path"])
        need(path.is_absolute() and path.is_relative_to(original_collection/"observations") and path.parent.name in captures and
             str(path) not in seen and type(record["bytes"]) is int and record["bytes"]>0 and
             re.fullmatch(r"[0-9a-f]{64}",record["sha256"]), "Published teacher source observation differs")
        capture = captures[path.parent.name]["manifest"]
        need(Path(capture["student_checkpoint"]).resolve()==student and
             capture["student_statistics_sha256"]==metadata["teacher_statistics_sha256"] and
             capture["action_mask"]==metadata["action_mask"], "Published capture/teacher source differs")
        seen.add(str(path));used[path.parent.name]+=1
    need(set(used)==set(evaluation.TASKS) and all(used[k]<=captures[k]["counts"]["saved"] for k in used), "Published teacher task coverage differs")
    if final:need(metadata['count']==metadata['requested_count']==160 and all(used[task]==16 for task in used),
                  'Published v3 teacher cache is incomplete')
    cache = costs["teacher_labeling"]["cache_identity"]
    need(cache["sha256"]==opd["probe_cache_sha256"] and type(cache["bytes"]) is int and cache["bytes"]>0, "Published teacher cache identity differs")
    teacher_cost = {"elapsed_seconds":positive(metadata["elapsed_seconds"],"teacher elapsed"),"timing_scope":metadata["timing_scope"],
        "cuda_peak_memory":validate_peaks(metadata["cuda_peak_memory"]),"cuda_peak_scope":metadata["cuda_peak_scope"],
        "model_dtype":metadata["model_dtype"],"autocast_dtype":metadata["autocast_dtype"],"requested_probes":metadata["requested_count"],
        "actual_probes":metadata["count"],"source_observations_per_task":dict(used),"cache_identity":cache}
    need(teacher_cost==costs["teacher_labeling"], "Published teacher costs disagree with raw metadata")
    expected_comparison = {"same_initial_adapter":str(adapter),"initial_adapter_weight_files":exports["qad"]["training_weights"],
        "same_frozen_base":str(Path(initial["base"]).resolve()),"same_demo_and_update_budget":True,
        "continuation_wall_ratio_opd_over_continued":computed["qad_opd"]["wall_seconds"]/computed["continued_qad"]["wall_seconds"],
        "equal_total_compute":False}
    need(costs["comparison"]==expected_comparison and costs["export_reported_seconds"]==
         {n:positive(v["elapsed_seconds"],"merge elapsed") for n,v in exports.items()}, "Published ratios/export costs differ")
    need(costs["excluded_from_formal_cost_comparison"]==EXCLUDED and costs["accounting_notes"]==ACCOUNTING_NOTES,
         "Published cost accounting scope differs")
    return costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qad-dir")
    parser.add_argument("--qad-merged")
    parser.add_argument("--round")
    parser.add_argument("--out", default=str(ROOT/"paper/evidence/training"))
    parser.add_argument("--protocol", default=str(ROOT/"exp/recovery_protocol.json"))
    parser.add_argument("--qad-log")
    parser.add_argument('--orchestrator-run',help='Completed v3 run directory; derive selected actual stages from final_manifest.json')
    parser.add_argument("--verify-published", metavar="DIRECTORY", help="Read-only public verification; no private weights/cache or torch required")
    args = parser.parse_args()
    if args.verify_published:
        need(not any((args.qad_dir,args.qad_merged,args.round,args.qad_log,args.orchestrator_run)), "Do not combine verification and collection")
        result = verify_published(args.verify_published)
        print(json.dumps({"status":"verified", "formal_training_arms":list(result["training"]),"private_tensors_opened":False}))
        return
    need(bool(args.orchestrator_run) or all((args.qad_dir,args.qad_merged,args.round)), "Collection requires --orchestrator-run or --qad-dir, --qad-merged and --round")
    need(not args.orchestrator_run or not any((args.qad_dir,args.qad_merged,args.round,args.qad_log)),
         'Do not mix orchestrator-derived and manually specified stage paths')
    result = collect(args.qad_dir,args.qad_merged,args.round,args.out,args.protocol,args.qad_log,args.orchestrator_run)
    print(json.dumps({"status":result["status"],"out":str(Path(args.out).resolve()),
                     "formal_training_arms":list(result["training"])}))


if __name__ == "__main__":
    main()
