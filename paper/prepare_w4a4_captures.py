#!/usr/bin/env python3
"""Prepare, but never run, the seven v12 W4A4 screenshot commands.

The default verify-completed mode reads finalized original evidence without
training, quantization, rollout or GPU execution. The original smoke renderer is
retained only for provenance reconstruction via explicit legacy-smoke mode.
Both preparation modes require a completed v12 final manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("bf16", "ptq", "qad", "continued_qad", "qad_opd")
TASK = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
FINAL_FORMAT = "w4a4_recovery_v12_final_manifest"
PROTOCOL_ID = "w4a4-recovery-v12-rtn"
PROTOCOL_VERSION = 12
COMMON_SCRIPT = "common_v12.sh"
INFERENCE_SWITCHES = ("FP4VLA_QUANT", "FP4VLA_W4A4", "FP4VLA_W4A4_ADAPTER",
                      "FP4VLA_SATURATE_F16_ACTIVATIONS")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def regular(path, label):
    value = Path(path).expanduser().resolve(strict=True)
    need(value.is_file() and not value.is_symlink(), f"{label} is not a regular file: {value}")
    return value


def directory(path, label):
    value = Path(path).expanduser().resolve(strict=True)
    need(value.is_dir() and not value.is_symlink(), f"{label} is not a regular directory: {value}")
    return value


def executable(path, label):
    """Resolve the venv launcher; virtualenv Python is commonly a symlink."""
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = Path.cwd() / value
    value = Path(os.path.abspath(value))
    target = value.resolve(strict=True)
    need(value.is_file() and target.is_file() and os.access(target, os.X_OK),
         f"{label} is not executable: {value}")
    # Keep the venv launcher path in generated commands. Returning ``target``
    # would silently replace it with the system interpreter behind the symlink.
    return value


def load_json(path, label):
    value = json.loads(regular(path, label).read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def identity_matches(record, path, label):
    need(record.get("path") == str(path), f"{label} path differs from manifest")
    need(type(record.get("bytes")) is int and record["bytes"] == path.stat().st_size,
         f"{label} byte identity differs")
    need(record.get("sha256") == digest(path), f"{label} SHA-256 differs")


def model_identity(record, label):
    need(isinstance(record, dict) and isinstance(record.get("path"), str),
         f"{label} identity is missing")
    root = directory(record["path"], label)
    shards = record.get("shards")
    need(isinstance(shards, list) and shards, f"{label} has no recorded shards")
    for shard in shards:
        path = regular(root / shard["name"], f"{label} shard")
        identity_matches({**shard, "path": str(path)}, path, f"{label} shard {path.name}")
    return root


def validate_final(final_path):
    final_path = regular(final_path, "final_manifest")
    need(final_path.name == "final_manifest.json", "--final-manifest must name final_manifest.json")
    final = load_json(final_path, "final_manifest")
    need(final.get("format") == FINAL_FORMAT,
         "capture preparation requires the completed v12 W4A4 final manifest")
    need(final.get("selection_uses_heldout") is False, "final selection must be development-only")
    need(final.get("required_arms") == list(ARMS), "final manifest does not contain five frozen arms")
    protocol_path = regular(final.get("protocol_file"), "v12 protocol")
    protocol = load_json(protocol_path, "v12 protocol")
    need(protocol.get("id") == PROTOCOL_ID and protocol.get("version") == PROTOCOL_VERSION and
         protocol.get("w4a4") is True, "protocol is not frozen v12 RTN W4A4")
    need(final.get("protocol_sha256") == digest(protocol_path), "final protocol SHA-256 differs")
    run = load_json(final_path.parent / "run_manifest.json", "run_manifest")
    need(run.get("status") == "complete" and run.get("output_layout") == "stable_paths_v2",
         "v12 run_manifest is not complete")
    need(run.get("protocol_sha256") == final["protocol_sha256"], "run/final protocol identities differ")
    comparison = final.get("heldout_comparison")
    need(isinstance(comparison, dict), "heldout comparison identity is missing")
    comparison_path = regular(comparison.get("path"), "heldout comparison")
    identity_matches(comparison, comparison_path, "heldout comparison")
    compared = load_json(comparison_path, "heldout comparison")
    need(compared.get("environment_pairing_verified") is True and
         compared.get("protocol_consistency_verified") is True and
         set(compared.get("arms", {})) == set(ARMS), "heldout comparison is incomplete")
    return {"final": final, "protocol": protocol, "protocol_path": protocol_path,
            "run": run, "final_path": final_path, "comparison": compared,
            "comparison_path": comparison_path}


def inference_contract(bundle, arm, model):
    """Read inference switches from the hash-bound evaluation, never training."""
    need(arm in ("qad", "qad_opd"), "capture inference requires a selected recovery arm")
    round_dir = directory(bundle["final"].get("heldout_round"), "heldout round")
    need(bundle["comparison_path"].parent == round_dir,
         "heldout comparison is outside the final heldout round")
    relative = f"heldout_{arm}/eval_manifest.json"
    path = regular(round_dir / relative, f"{arm} evaluation manifest")
    record = bundle["comparison"].get("source_files", {}).get(relative)
    need(isinstance(record, dict), f"{arm} evaluation identity is missing from comparison")
    identity_matches({**record, "path": str(path)}, path, f"{arm} evaluation manifest")
    manifest = load_json(path, f"{arm} evaluation manifest")
    need(manifest.get("checkpoint") == str(model),
         f"{arm} evaluation checkpoint differs from selected model")
    need(manifest.get("purpose") == "heldout" and
         manifest.get("protocol_sha256") == bundle["final"]["protocol_sha256"],
         f"{arm} evaluation does not use the final heldout protocol")
    summary = manifest.get("environment_summary")
    env = summary.get("variables") if isinstance(summary, dict) else None
    need(isinstance(env, dict) and all(env.get(key) in ("0", "1") for key in INFERENCE_SWITCHES),
         f"{arm} evaluation lacks explicit binary inference switches")
    need(env["FP4VLA_QUANT"] == "0" and env["FP4VLA_W4A4"] == "1" and
         env["FP4VLA_W4A4_ADAPTER"] == "1",
         f"{arm} evaluation does not use the W4A4 adapter inference path")
    return {"arm": arm, "checkpoint": str(model),
            "eval_manifest": {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)},
            "environment": {key: env[key] for key in INFERENCE_SWITCHES}}


def validate_inputs(bundle):
    final, protocol, run = bundle["final"], bundle["protocol"], bundle["run"]
    pressure = directory(final.get("selected_pressure_checkpoint"), "selected W4A4 PTQ checkpoint")
    need((pressure / "category_ptq_recipe.json").is_file() and
         (pressure / "category_bake_manifest.json").is_file(),
         "selected pressure checkpoint lacks category W4A4 provenance")
    q_model = model_identity(final["selected_qad_model_identity"], "selected QAD model")
    op_model = model_identity(final["selected_opd_model_identity"], "selected OPD model")
    q_adapter = directory(final["selected_qad_checkpoint_identity"]["path"], "selected QAD adapter checkpoint")
    need((q_adapter / "recovery_manifest.json").is_file(), "selected QAD adapter lacks recovery manifest")
    q_merge = load_json(q_model / "merge_manifest.json", "selected QAD merge manifest")
    op_merge = load_json(op_model / "merge_manifest.json", "selected OPD merge manifest")
    need(q_merge.get("status") == op_merge.get("status") == "complete", "selected merge is incomplete")
    need(Path(q_merge.get("base", "")).resolve() == pressure and
         Path(op_merge.get("base", "")).resolve() == pressure,
         "selected recovery models do not use selected W4A4 PTQ base")
    need(Path(q_merge.get("training_checkpoint", "")).resolve() == q_adapter,
         "selected QAD model is not built from the selected QAD adapter checkpoint")
    q_rec, op_rec = q_merge.get("recovery_manifest"), op_merge.get("recovery_manifest")
    need(isinstance(q_rec, dict) and isinstance(op_rec, dict), "merge lacks recovery provenance")
    for name, rec in (("QAD", q_rec), ("OPD", op_rec)):
        need(rec.get("w4a4_enabled") is True and rec.get("scope") == "all_ordinary_linear",
             f"selected {name} recovery does not use all-ordinary-linear W4A4")
        need(int(rec.get("rank")) == int(protocol["selection"]["rank"]) and
             float(rec.get("alpha")) == float(protocol["selection"]["alpha"]),
             f"selected {name} rank/alpha differs from protocol")
        need(rec.get("compute_dtype") == "bfloat16" and rec.get("parameter_dtype") == "torch.float32",
             f"selected {name} dtype contract differs")
    need(float(q_rec.get("probe_weight", 0)) == 0 and float(op_rec.get("probe_weight", 0)) > 0,
         "selected QAD/OPD probe weights do not identify the stages")
    cache = regular(op_rec.get("probe_cache"), "selected OPD teacher cache")
    teacher = directory(run.get("base"), "BF16 teacher checkpoint")
    for name, root in (("BF16 teacher", teacher), ("selected PTQ", pressure), ("selected QAD model", q_model)):
        for filename in ("config.json", "statistics.json"):
            need((root / filename).is_file(), f"{name} lacks {filename}")
    groot = directory(run.get("gr00t_repo"), "GR00T repository")
    server_python = executable(run.get("python"), "training/server Python")
    rollout_python = executable(run.get("rollout_python"), "LIBERO rollout Python")
    dataset = directory(run.get("dataset"), "GR00T LIBERO dataset")
    capture_dataset = directory(q_rec.get("capture_dataset"), "QAD capture dataset")
    capture_sha = q_rec.get("capture_dataset_sha256")
    need(isinstance(capture_sha, str) and len(capture_sha) == 64, "QAD capture dataset identity is missing")
    base_config = load_json(pressure / "config.json", "selected PTQ config")
    return {"pressure": pressure, "q_model": q_model, "op_model": op_model, "q_adapter": q_adapter,
            "q_rec": q_rec, "op_rec": op_rec, "cache": cache, "teacher": teacher,
            "inference": {"qad": inference_contract(bundle, "qad", q_model),
                          "qad_opd": inference_contract(bundle, "qad_opd", op_model)},
            "groot": groot, "server_python": server_python, "rollout_python": rollout_python,
            "dataset": dataset, "capture_dataset": capture_dataset, "capture_sha": capture_sha,
            "backbone": base_config.get("model_name", "nvidia/Cosmos-Reason2-2B"),
            "task_seed": int(protocol["partitions"]["development"]["seed"]),
            "state_seed": int(run["train_seed"]), "protocol_path": bundle["protocol_path"],
            "final_path": bundle["final_path"]}


def q(value):
    return shlex.quote(str(value))


def write_script(path, text):
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def render_scripts(out, bundle, values):
    out.mkdir(parents=True, exist_ok=False)
    # Frozen protocol copies may live under results/ or outside the checkout.
    # Source commands belong to this generator's repository, not that copy.
    project = ROOT
    # Production v12 manifests always carry the explicit protocol version.
    # The fallback keeps the CPU-only unit fixtures that exercise the renderer
    # without a protocol document readable; it is never reachable through
    # ``validate_final`` and is not part of a v12 capture plan.
    common_script = (COMMON_SCRIPT if bundle.get("protocol", {}).get("version") == PROTOCOL_VERSION
                     else "common_v11.sh")
    need((project / "paper").is_dir() and (project / "rl").is_dir() and
         (project / "eval").is_dir(), "generator is not inside the fp4vla repository root")
    common = r'''#!/usr/bin/env bash
set -euo pipefail
PROJECT=@PROJECT@
GR00T_REPO=@GR00T_REPO@
PTQAD_PYTHON=@PTQAD_PYTHON@
LIBERO_PYTHON=@LIBERO_PYTHON@
BF16_TEACHER=@BF16_TEACHER@
PTQ_BASE=@PTQ_BASE@
QAD_MODEL=@QAD_MODEL@
QAD_ADAPTER=@QAD_ADAPTER@
OPD_MODEL=@OPD_MODEL@
DATASET=@DATASET@
CAPTURE_DATASET=@CAPTURE_DATASET@
CAPTURE_DATASET_SHA256=@CAPTURE_DATASET_SHA256@
PROTOCOL_FILE=@PROTOCOL_FILE@
BACKBONE_MODEL=@BACKBONE_MODEL@
CAPTURE_ROOT=@CAPTURE_ROOT@
SCRATCH_ROOT="$CAPTURE_ROOT/scratch"
export PROJECT GR00T_REPO PTQAD_PYTHON LIBERO_PYTHON BF16_TEACHER PTQ_BASE QAD_MODEL QAD_ADAPTER OPD_MODEL DATASET CAPTURE_DATASET CAPTURE_DATASET_SHA256 PROTOCOL_FILE BACKBONE_MODEL CAPTURE_ROOT SCRATCH_ROOT
export GR00T_BACKBONE_MODEL="$BACKBONE_MODEL"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 NO_ALBUMENTATIONS_UPDATE=1
export PTQAD_ZMQ_TIMEOUT_MS="${PTQAD_ZMQ_TIMEOUT_MS:-120000}"
export PTQAD_MEDIA_LIB="${PTQAD_MEDIA_LIB:-$HOME/miniforge3/envs/media7/lib}"
export LD_LIBRARY_PATH="$PTQAD_MEDIA_LIB:/usr/local/cuda/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="${PTQAD_MEDIA_LIB%/lib}/bin:/usr/local/cuda/bin:$PATH" PYTHONPATH="$PROJECT:$GR00T_REPO"
require_sources() {
  for executable in "$PTQAD_PYTHON" "$LIBERO_PYTHON"; do test -x "$executable" || { echo "missing executable: $executable" >&2; exit 2; }; done
  for directory in "$BF16_TEACHER" "$PTQ_BASE" "$QAD_MODEL" "$QAD_ADAPTER" "$OPD_MODEL" "$DATASET" "$CAPTURE_DATASET"; do test -d "$directory" || { echo "missing source: $directory" >&2; exit 2; }; done
  test -f "$PROTOCOL_FILE" && test -f "$PTQ_BASE/category_ptq_recipe.json" && test -f "$QAD_MODEL/merge_manifest.json"
}
prepare_scratch() {
  local figure="$1"; require_sources; local out="$SCRATCH_ROOT/$figure"
  if [[ -e "$out" ]]; then echo "refusing to overwrite existing scratch: $out" >&2; exit 2; fi
  mkdir -p "$out"; printf '%s\n' "$(date -u +%FT%TZ)" > "$out/started_utc.txt"; printf '%s\n' "$out"
}
assert_no_compute_apps() {
  local rows
  if ! rows="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)"; then
    echo "nvidia-smi query failed; refusing to start capture" >&2; exit 2
  fi
  if [[ -n "${rows//[[:space:]]/}" ]]; then echo "CUDA process exists; do not start capture: $rows" >&2; exit 2; fi
}
wait_for_port() {
  local pid="$1" port="$2"; python3 - "$pid" "$port" <<'PY'
import os, socket, sys, time
pid, port = map(int, sys.argv[1:]); deadline = time.monotonic() + 240
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1): break
    except OSError: time.sleep(1)
else: raise TimeoutError("W4A4 recovery server did not bind")
PY
}
'''
    replacements = {
        "@PROJECT@": q(project), "@GR00T_REPO@": q(values["groot"]),
        "@PTQAD_PYTHON@": q(values["server_python"]), "@LIBERO_PYTHON@": q(values["rollout_python"]),
        "@BF16_TEACHER@": q(values["teacher"]), "@PTQ_BASE@": q(values["pressure"]),
        "@QAD_MODEL@": q(values["q_model"]), "@QAD_ADAPTER@": q(values["q_adapter"]),
        "@OPD_MODEL@": q(values["op_model"]), "@DATASET@": q(values["dataset"]),
        "@CAPTURE_DATASET@": q(values["capture_dataset"]), "@CAPTURE_DATASET_SHA256@": q(values["capture_sha"]),
        "@PROTOCOL_FILE@": q(values["protocol_path"]), "@BACKBONE_MODEL@": q(values["backbone"]),
        "@CAPTURE_ROOT@": q(out),
    }
    for key, value in replacements.items():
        common = common.replace(key, value)
    write_script(out / common_script, common)

    final = bundle["final"]
    protocol = bundle["protocol"]
    scope = values["q_rec"]["scope"]
    rank, alpha = int(values["q_rec"]["rank"]), float(values["q_rec"]["alpha"])
    lr = float(final["selected_qad_learning_rate"])
    weight = float(final["selected_opd_weight"])
    every = int(protocol["selection"]["opd_every"])
    q_train_sat = "1" if values["q_rec"].get("f16_activation_saturation") else "0"
    op_train_sat = "1" if values["op_rec"].get("f16_activation_saturation") else "0"
    inference = values["inference"]
    common_train = (f"QAD_LORA_R={q(rank)} QAD_LORA_ALPHA={q(alpha)} "
                    f"QAD_LORA_SCOPE={q(scope)} QAD_LR={q(lr)} QAD_W4A4=1 "
                    "FP4VLA_QUANT=0 FP4VLA_W4A4=1 FP4VLA_SATURATE_F16_ACTIVATIONS=")
    scripts = {
        "shot_bake": '''#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/@COMMON_SCRIPT@"
OUT="$(prepare_scratch shot_bake)"; assert_no_compute_apps
QAD_RECOVERY="$QAD_ADAPTER/recovery_manifest.json"; test -f "$QAD_RECOVERY"
cd "$PROJECT"
"$PTQAD_PYTHON" -u "$PROJECT/paper/build_recipe_inventory_v12.py" \
  --checkpoint "$PTQ_BASE" \
  --recovery-manifest "$QAD_RECOVERY" \
  --out "$OUT/recipe_inventory.json" 2>&1 | tee "$OUT/recipe_inventory.raw.log"
"$PTQAD_PYTHON" - "$OUT/recipe_inventory.json" "$PROTOCOL_FILE" "$PTQ_BASE" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    inventory = json.load(stream)
with open(sys.argv[2]) as stream:
    protocol = json.load(stream)
base = sys.argv[3]
recipe_name = inventory.get("recipe")
recipe = inventory.get("recipes", {}).get(recipe_name, {})
scope = protocol.get("quantization_scope", {})
assert protocol.get("version") == 12 and protocol.get("w4a4") is True
assert recipe_name and recipe.get("nvfp4_params") == recipe.get("linear_params")
assert recipe.get("fp8_params", 0) == 0 and recipe.get("bf16_params", 0) == 0
print(json.dumps({
    "protocol": protocol.get("id"),
    "calibration_mode": scope.get("calibration_mode"),
    "weight_format": scope.get("weight_format"),
    "eligible_tensors": scope.get("eligible_tensor_count"),
    "recipe": recipe_name,
    "nvfp4_params": recipe.get("nvfp4_params"),
    "linear_params": recipe.get("linear_params"),
    "residual_bytes": inventory.get("recovery_residual", {}).get("target_bytes"),
    "ptq_base": base,
}, ensure_ascii=False))
PY
sha256sum "$PROTOCOL_FILE" "$PTQ_BASE/category_ptq_recipe.json" "$PTQ_BASE/category_bake_manifest.json" "$QAD_RECOVERY"
printf '%s\\n' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
''',
        "shot_collect": '''#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/@COMMON_SCRIPT@"
OUT="$(prepare_scratch shot_collect)"; assert_no_compute_apps
test -f "$CAPTURE_DATASET/eval_manifest.json"
test -d "$CAPTURE_DATASET/observations"
cd "$PROJECT"
"$PTQAD_PYTHON" -u "$PROJECT/exp/verify_teacher_replay.py" \
  --root "$CAPTURE_DATASET" \
  --protocol-file "$PROTOCOL_FILE" \
  --teacher "$BF16_TEACHER" \
  --minimum-episodes 2 2>&1 | tee "$OUT/teacher_replay_audit.raw.log"
SAMPLES="$(find "$CAPTURE_DATASET/observations" -name 'sample_*.pt' -type f | wc -l)"
TASKS="$(find "$CAPTURE_DATASET/observations" -mindepth 1 -maxdepth 1 -type d | wc -l)"
printf '%s\\n' "teacher_replay_verified=1 samples=$SAMPLES tasks=$TASKS" | tee "$OUT/teacher_replay_summary.txt"
printf '%s\\n' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
''',
        "shot_qad": '''#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/@COMMON_SCRIPT@"
OUT="$(prepare_scratch shot_qad)"; assert_no_compute_apps
cd "$GR00T_REPO"
export GR00T_BASE_CKPT="$PTQ_BASE" QAD_OUT="$OUT/qad" QAD_DATASET="$DATASET" QAD_CAPTURE_DATASET="$CAPTURE_DATASET" QAD_CAPTURE_DATASET_SHA256="$CAPTURE_DATASET_SHA256"
export QAD_STEPS=2 QAD_SAVE_STEPS=2 QAD_SAVE_TOTAL_LIMIT=2 QAD_MICRO_BATCH=1 QAD_GLOBAL_BATCH=1 QAD_ACTIVATION_CHECKPOINTING=1 QAD_OPD_MSE_W=0 TRAIN_SEED=@SEED@ PROTOCOL_FILE="$PROTOCOL_FILE"
export @COMMON_TRAIN@@Q_SAT@
unset QAD_INIT_ADAPTER QAD_MAX_GRAD_NORM OPD_CACHE_PATH || true
"$PTQAD_PYTHON" -u "$PROJECT/rl/lora_qad.py" 2>&1 | tee "$OUT/qad_2step.raw.log"
test -f "$OUT/qad/checkpoint-2/recovery_manifest.json"
printf '%s\n' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
''',
        "shot_opdcache": '''#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/@COMMON_SCRIPT@"
OUT="$(prepare_scratch shot_opdcache)"; assert_no_compute_apps
ROLLOUT="$SCRATCH_ROOT/shot_rollout"; test -d "$ROLLOUT/observations"
test "$(find "$ROLLOUT/observations" -name 'sample_*.pt' | wc -l)" -eq 2
cd "$GR00T_REPO"
export FP4VLA_QUANT=0 FP4VLA_W4A4=0 FP4VLA_W4A4_ADAPTER=0 FP4VLA_SATURATE_F16_ACTIVATIONS=0
"$PTQAD_PYTHON" -u "$PROJECT/rl/opd_probe_cache.py" --teacher "$BF16_TEACHER" --input-dir "$ROLLOUT/observations" --dataset "$DATASET" --count 2 --seed @SEED@ --model-dtype float32 --autocast-dtype bfloat16 --device cuda --out "$OUT/teacher_probes.pt" 2>&1 | tee "$OUT/opdcache_2sample.raw.log"
test -f "$OUT/teacher_probes.pt"
printf '%s\n' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
''',
        "shot_opd": '''#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/@COMMON_SCRIPT@"
OUT="$(prepare_scratch shot_opd)"; assert_no_compute_apps
CACHE="$SCRATCH_ROOT/shot_opdcache/teacher_probes.pt"; test -f "$CACHE"
cd "$GR00T_REPO"
export GR00T_BASE_CKPT="$PTQ_BASE" QAD_INIT_ADAPTER="$QAD_ADAPTER" QAD_OUT="$OUT/opd" QAD_DATASET="$DATASET" QAD_CAPTURE_DATASET="$CAPTURE_DATASET" QAD_CAPTURE_DATASET_SHA256="$CAPTURE_DATASET_SHA256" OPD_CACHE_PATH="$CACHE" OPD_EVERY=@EVERY@
export QAD_STEPS=4 QAD_SAVE_STEPS=4 QAD_SAVE_TOTAL_LIMIT=2 QAD_MICRO_BATCH=1 QAD_GLOBAL_BATCH=1 QAD_ACTIVATION_CHECKPOINTING=1 QAD_OPD_MSE_W=@WEIGHT@ QAD_MAX_GRAD_NORM=0.25 TRAIN_SEED=@SEED@ PROTOCOL_FILE="$PROTOCOL_FILE"
export @COMMON_TRAIN@@OP_SAT@ FP4VLA_W4A4_ADAPTER=1
"$PTQAD_PYTHON" -u "$PROJECT/rl/lora_qad.py" 2>&1 | tee "$OUT/opd_4step.raw.log"
test -f "$OUT/opd/checkpoint-4/recovery_manifest.json"
printf '%s\n' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
''',
    }
    # The v11 CPU fixtures still exercise the renderer with a minimal bundle.
    # Only a validated v12 bundle receives the two new provenance-only stages;
    # production v12 plans therefore always contain all seven entrypoints.
    if common_script != COMMON_SCRIPT:
        scripts.pop("shot_bake")
        scripts.pop("shot_collect")
    for name, text in scripts.items():
        text = text.replace("@COMMON_SCRIPT@", common_script)
        text = text.replace("@SEED@", q(values["state_seed"])).replace("@COMMON_TRAIN@", common_train)
        text = text.replace("@Q_SAT@", q(q_train_sat)).replace("@OP_SAT@", q(op_train_sat))
        text = text.replace("@EVERY@", q(every)).replace("@WEIGHT@", q(weight))
        write_script(out / f"{name}.sh", text)

    server_tail = '''export GR00T_EVAL_SEED=@SEED@ @INFERENCE_ENV@ GR00T_BACKBONE_MODEL="$BACKBONE_MODEL"
PORT=@PORT@
server_pid=""
cleanup() { if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" || true; fi; }
trap cleanup EXIT
cd "$GR00T_REPO"
"$PTQAD_PYTHON" -u "$PROJECT/eval/serve_recovery.py" --model-path @MODEL@ --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper --host 127.0.0.1 --port "$PORT" > >(tee "$OUT/server.raw.log") 2>&1 &
server_pid=$!; wait_for_port "$server_pid" "$PORT"
'''
    server_tail = server_tail.replace("@SEED@", q(values["state_seed"]))
    scripts["shot_evalserver"] = '''#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/@COMMON_SCRIPT@"
OUT="$(prepare_scratch shot_evalserver)"; assert_no_compute_apps
@SERVER@
"$PTQAD_PYTHON" - "$PORT" <<'PY' 2>&1 | tee "$OUT/ping.raw.log"
import json, sys
from gr00t.policy.server_client import PolicyClient
import os
client = PolicyClient(host="127.0.0.1", port=int(sys.argv[1]), timeout_ms=int(os.environ.get("PTQAD_ZMQ_TIMEOUT_MS", "120000")))
try:
    response = client.call_endpoint("ping", requires_input=False)
    print(json.dumps({"endpoint":"ping", "scope":"health RPC only", "response":response}, ensure_ascii=False), flush=True)
    if response.get("status") != "ok": raise RuntimeError(response)
finally: client.close()
PY
kill -TERM "$server_pid"; wait "$server_pid" || true; server_pid=""
printf '%s\n' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
'''
    scripts["shot_rollout"] = '''#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/@COMMON_SCRIPT@"
OUT="$(prepare_scratch shot_rollout)"; assert_no_compute_apps
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPD_CAPTURE_DIR="$OUT/observations" OPD_CAPTURE_EVERY=1 OPD_CAPTURE_PER_TASK=2 OPD_CAPTURE_LIMIT=2 OPD_CAPTURE_PER_EPISODE=2
export FP4VLA_CAPTURE_PURPOSE=screenshot_smoke FP4VLA_CAPTURE_TASK_NAME="@TASK@" FP4VLA_CAPTURE_SEED=@TASK_SEED@ FP4VLA_CAPTURE_INIT_STATE_INDICES=4 FP4VLA_CAPTURE_PROTOCOL_SHA256="$(sha256sum "$PROTOCOL_FILE" | cut -d' ' -f1)" FP4VLA_CAPTURE_EVENT_FILE="$OUT/reset_events.jsonl"
@SERVER@
"$LIBERO_PYTHON" -u "$PROJECT/eval/rollout_seeded.py" --env-name libero_sim/@TASK@ --n-episodes 1 --n-envs 1 --seed @TASK_SEED@ --init-state-indices 4 --max-episode-steps 720 --n-action-steps 8 --video-dir "$OUT/videos" --policy-client-host 127.0.0.1 --policy-client-port "$PORT" 2>&1 | tee "$OUT/rollout_1task_1episode.raw.log"
test "$(find "$OUT/observations" -name 'sample_*.pt' | wc -l)" -eq 2
python3 - "$OUT" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1]); json.dump({"purpose":"screenshot_smoke","tasks":1,"episodes":1,"captured_observations":2,"not_formal_evaluation":True}, (root / "rollout_smoke_manifest.json").open("w"), indent=2)
PY
"$PTQAD_PYTHON" - <<'PY'
from gr00t.policy.server_client import PolicyClient
import os
client = PolicyClient(host="127.0.0.1", port=5618, timeout_ms=int(os.environ.get("PTQAD_ZMQ_TIMEOUT_MS", "120000")))
try: client.kill_server()
finally: client.close()
PY
wait "$server_pid" || true; server_pid=""
printf '%s\n' "$(date -u +%FT%TZ)" > "$OUT/completed_utc.txt"
'''
    for name, text in (("shot_evalserver", scripts["shot_evalserver"]), ("shot_rollout", scripts["shot_rollout"])):
        if name == "shot_evalserver":
            arm = "qad_opd"
            server = server_tail.replace("@PORT@", "5617").replace("@MODEL@", '"$OPD_MODEL"')
        else:
            arm = "qad"
            server = server_tail.replace("@PORT@", "5618").replace("@MODEL@", '"$QAD_MODEL"')
        env = inference[arm]["environment"]
        server = server.replace("@INFERENCE_ENV@", " ".join(f"{key}={q(env[key])}" for key in INFERENCE_SWITCHES))
        text = text.replace("@SERVER@", server)
        text = text.replace("@COMMON_SCRIPT@", common_script)
        text = text.replace("@TASK_SEED@", q(values["task_seed"])).replace("@TASK@", TASK)
        write_script(out / f"{name}.sh", text)
    script_files = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                    for path in sorted(out.glob("*.sh"))}
    plan = {"version": 3, "scope": "v12_w4a4_capture_preparation", "capture_mode": "legacy-smoke", "gpu_executed": False,
            "final_manifest": str(bundle["final_path"]), "final_manifest_sha256": digest(bundle["final_path"]),
            "protocol": str(bundle["protocol_path"]), "protocol_sha256": digest(bundle["protocol_path"]),
            "scratch_root": str(out / "scratch"), "selected_qad_learning_rate": lr,
            "selected_opd_weight": weight, "lora_scope": scope, "rank": rank, "alpha": alpha,
            "training_activation_saturation": {"qad": q_train_sat, "qad_opd": op_train_sat},
            "inference_contracts": inference,
            "script_files": script_files,
            "script_dependencies": {"all_entrypoints": [common_script],
                                    "publication": f"Retain {common_script} and plan.json beside all seven v12 entrypoints; publishing entrypoints alone is incomplete."},
            "stages": {"shot_bake": "CPU-only recipe, residual-budget and protocol identity audit",
                        "shot_collect": "CPU-only verification of the hash-bound successful BF16 teacher replay archive",
                        "shot_qad": "two optimizer steps from selected W4A4 PTQ base; path smoke only",
                        "shot_rollout": "one LIBERO task and one episode using selected QAD model and hash-verified heldout inference switches; exactly two student observations",
                        "shot_opdcache": "labels those two observations with the BF16 teacher",
                        "shot_opd": "four optimizer steps from selected QAD adapter with selected OPD weight",
                        "shot_evalserver": "loads selected OPD W4A4 adapter with hash-verified heldout inference switches and calls health ping only"},
            "execution_order": ["shot_bake", "shot_collect", "shot_qad", "shot_rollout", "shot_opdcache", "shot_opd", "shot_evalserver"],
            "rollout_model": "selected_qad_model_identity",
            "rollout_activation_saturation": inference["qad"]["environment"]["FP4VLA_SATURATE_F16_ACTIVATIONS"],
            "note": "Short smoke output is not formal 2000-step training or held-out success evidence. Every script refuses an existing scratch stage and checks GPU idleness."}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(f"""# v12 W4A4 screenshot command set

Generated from the completed v12 `final_manifest.json`. This directory contains
seven real shell commands and does not execute training, evaluation, or desktop
capture. Run each script through the approved Ubuntu screenshot skill wrapper.
Retain `{common_script}` and `plan.json` beside all seven entrypoints when archiving
or publishing this command set. Every entrypoint sources that common file;
`plan.json` records all eight shell-file hashes and the evaluation manifest
identities used to select inference switches. Entry scripts alone are incomplete.
The scripts write only below `scratch/`, refuse existing stage directories,
check `nvidia-smi` before starting, and preserve raw command output with `set -x`
and `tee`.

`shot_bake.sh` and `shot_collect.sh` are CPU-only provenance checks. They do not
re-bake weights or recollect trajectories. `shot_qad.sh` and `shot_opd.sh` are
short path checks (2 and 4 optimizer steps);
their output cannot be reported as the formal 2000-step recovery result. Run
`shot_rollout.sh` before `shot_opdcache.sh`, then `shot_opd.sh`. The server
health screenshot reports only a `ping` RPC and is not a closed-loop score.
Training and inference contracts are separate in `plan.json`: the training
short checks follow their recovery records, while rollout and server ping use
the selected models' hash-verified heldout evaluation environments. A strict
initial QAD training run does not imply strict activation conversion at inference.

`shot_rollout.sh` sets `FP4VLA_CAPTURE_TASK_NAME` and
`FP4VLA_CAPTURE_EVENT_FILE` before starting the server. The event file is
written inside that run's scratch directory and is required when
`OPD_CAPTURE_PER_EPISODE=2`; do not remove it or reuse a file from another run.
""", encoding="utf-8")
    return plan


def render_completed_scripts(out, bundle, evidence_root):
    """Prepare seven read-only commands; never invoke the legacy smoke path."""
    helper = Path(__file__).with_name("verify_completed_capture.py")
    spec = importlib.util.spec_from_file_location("completed_capture", helper)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    contract = audit.source_contract(evidence_root, digest(bundle["final_path"]))
    need(contract["protocol_sha256"] == bundle["final"]["protocol_sha256"],
         "capture evidence belongs to another protocol")
    plan = {"version": 4, "scope": "v12_w4a4_capture_preparation",
            "capture_mode": audit.MODE, "gpu_executed": False,
            "final_manifest": str(bundle["final_path"]),
            "final_manifest_sha256": digest(bundle["final_path"]),
            "protocol": str(bundle["protocol_path"]),
            **contract, "execution_order": list(audit.FIGURES),
            "stages": audit.CAPTIONS,
            "script_dependencies": {"all_entrypoints": [COMMON_SCRIPT, helper.name]},
            "note": "Real CPU verification of completed original evidence; no live experiment replay. All 17 publication screenshot slots remain required."}
    # Reject invalid completed stage records before writing any command set.
    for figure in audit.FIGURES:
        audit.audit(plan, figure)
    out.mkdir(parents=True, exist_ok=False)
    (out / helper.name).write_bytes(helper.read_bytes())
    common = """#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
CAPTURE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"""
    write_script(out / COMMON_SCRIPT, common)
    for figure in audit.FIGURES:
        write_script(out / (figure + ".sh"), f"""#!/usr/bin/env bash
set -euxo pipefail
source "$(dirname "$0")/{COMMON_SCRIPT}"
python3 -B "$CAPTURE_ROOT/{helper.name}" --plan "$CAPTURE_ROOT/plan_v12_completed.json" --figure {q(figure)}
""")
    plan["script_files"] = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
                            for path in sorted(out.iterdir()) if path.suffix in (".sh", ".py")}
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    (out / "plan.json").write_text(text, encoding="utf-8")
    (out / "plan_v12_completed.json").write_text(text, encoding="utf-8")
    (out / "README.md").write_text("""# v12 completed-evidence screenshot commands

Mode: `verify-completed`. These seven commands verify finalized original log and
metadata bytes on CPU. They do not train, quantize, collect trajectories, build a
cache, launch a policy server, or repeat an evaluation. No GPU status is polled.
Screenshots document execution of this verification, not live model execution.

Run the seven entrypoints through the approved screenshot wrapper. Keep the
unfiltered verification output, window binding and taskbar-crop proof, inspect
the actual screenshot, then register it with `paper/record_capture.py`.
Archive `common_v12.sh`, `verify_completed_capture.py` and
`plan_v12_completed.json` as supporting files for each screenshot. The plan binds
every source file to the final release, and the verifier rejects modified bytes.
All 17 publication screenshot slots remain required; this replaces seven slots.
Large private tensor files are not reopened: relevant figures explicitly verify
their previously recorded identities, not the tensors themselves.
""", encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("verify-completed", "legacy-smoke"), default="verify-completed",
                        help="Default reads completed evidence only; legacy-smoke reconstructs the prior GPU script provenance and must not be run during the frozen release")
    parser.add_argument("--evidence-root", type=Path, default=ROOT / "paper/evidence",
                        help="Installed final evidence, required by verify-completed mode")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing existing preparation output; choose a new --out")
    bundle = validate_final(args.final_manifest)
    if args.mode == "verify-completed":
        plan = render_completed_scripts(args.out.resolve(), bundle, args.evidence_root)
    else:
        values = validate_inputs(bundle)
        plan = render_scripts(args.out.resolve(), bundle, values)
    print(json.dumps({"status": "prepared", "out": str(args.out.resolve()),
                      "gpu_executed": plan["gpu_executed"], "scripts": 7}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
