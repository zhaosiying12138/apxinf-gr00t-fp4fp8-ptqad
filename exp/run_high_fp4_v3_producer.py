#!/usr/bin/env python3
"""Serial, fail-closed orchestration for the v3 high-FP4 recovery study.

The input PTQ selection must be a completed development-only selection.json.
The driver then runs two QAD learning rates, selects on development, collects
the winning QAD student rollouts, labels one frozen teacher cache, runs
continued-QAD and two OPD weights from the exact same QAD A/B checkpoint,
selects the OPD weight on development, and finally evaluates exactly five
heldout arms. Outputs use stable absolute paths; completion markers publish
atomically only after verification. Existing outputs are never overwritten.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from compare_recovery import _same_number, _validate_declared_accounting, compare_round
from run_recovery_eval import parse_log, validate_resets
TASKS = (
"LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
"LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
"KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
"KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
"LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
"STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
"LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
"LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
"KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
"KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)
class OrchestrationError(RuntimeError): pass

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        while c:=f.read(8*1024*1024): h.update(c)
    return h.hexdigest()
def jread(path: Path) -> Any:
    try: return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError) as e: raise OrchestrationError(f"invalid JSON {path}: {e}") from e
def jwrite(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n"); tmp.replace(path)
def need(path: str|Path, label: str) -> Path:
    p=Path(path).expanduser().resolve()
    if not p.exists(): raise OrchestrationError(f"{label} does not exist: {p}")
    return p
def sn(value: str) -> str: return re.sub(r"[^A-Za-z0-9_.-]+","_",value)
def cmdtext(c: list[str]) -> str:
    import shlex
    return shlex.join(c)

def load_protocol(path: Path) -> dict[str,Any]:
    d=jread(path); parts=d.get("partitions",d)
    # Version 8 uses an explicit pressure-window contract.  Keep the older
    # v3 contract below so historical CPU fixtures remain readable, while
    # refusing to silently coerce the new 16-episode heldout partition into
    # the old ten-episode shape.
    if int(d.get("version", 0)) >= 8:
        out={}
        # Recovery v8+ creates a fresh BF16 teacher replay before training.
        # Keep that partition in the normalized protocol so wrappers can
        # construct the teacher evaluation command without reaching into the
        # raw JSON (and without silently falling back to demo_data).
        for name in ("development", "teacher_supervision", "collection", "heldout"):
            x=parts.get(name) if isinstance(parts,dict) else None
            if not isinstance(x,dict):
                raise OrchestrationError(f"protocol has no {name}")
            idx,seed,eps=x.get("init_state_indices"),x.get("seed"),x.get("episodes_per_task")
            if (type(seed) is not int or type(eps) is not int or eps != len(idx or []) or
                    not isinstance(idx,list) or not idx or len(set(idx)) != len(idx) or
                    any(type(i) is not int or i < 0 or i >= 50 for i in idx)):
                raise OrchestrationError(f"invalid {name} partition")
            out[name]={"seed":seed,"episodes_per_task":eps,"init_state_indices":idx}
        s=d.get("selection",{}); r=s.get("pressure_rule",{})
        candidates=s.get("pressure_candidates")
        if (not isinstance(candidates,list) or not candidates or
                any(not isinstance(x,str) or not x for x in candidates) or
                r.get("choose") != "highest_fp4" or
                float(r.get("min_drop_from_bf16", -1)) < 0 or
                float(r.get("min_absolute_success", -1)) < 0 or
                r.get("selection_metric") != "micro_success_rate"):
            raise OrchestrationError("protocol does not match the frozen v8 pressure rule")
        valid_scopes = {"head", "head+lang", "head+lang_all", "all_ordinary_linear"}
        if s.get("recovery_scope") not in valid_scopes:
            raise OrchestrationError(
                "protocol recovery_scope is not registered in rl/lora_scope.py: "
                + repr(s.get("recovery_scope")))
        return {"data":d,"partitions":out,"selection":s,"sha256":sha(path),"path":str(path)}
    out={}
    for name,n in (("development",5),("collection",4),("heldout",10)):
        x=parts.get(name) if isinstance(parts,dict) else None
        if not isinstance(x,dict): raise OrchestrationError(f"protocol has no {name}")
        idx,seed,eps=x.get("init_state_indices"),x.get("seed"),x.get("episodes_per_task")
        if (type(seed)is not int or eps!=n or not isinstance(idx,list) or len(idx)!=eps or
            not idx or any(type(i)is not int for i in idx) or len(set(idx))!=len(idx) or
            any(i<0 or i>=50 for i in idx)): raise OrchestrationError(f"invalid {name} partition")
        out[name]={"seed":seed,"episodes_per_task":eps,"init_state_indices":idx}
    s=d.get("selection",{}); r=s.get("pressure_rule",{})
    if (s.get("pressure_candidates")!=["head_lang_vision","calib"] or
        r.get("choose")!="highest_fp4" or float(r.get("min_drop_from_bf16",-1))!=.20 or
        float(r.get("min_absolute_success",-1))!=.30):
        raise OrchestrationError("protocol does not match frozen high-FP4 v3 rule")
    return {"data":d,"partitions":out,"selection":s,"sha256":sha(path),"path":str(path)}

def frac(row: dict[str,Any], name: str) -> Fraction:
    if type(row.get("successes")) is not int or type(row.get("episodes")) is not int:
        raise OrchestrationError(f"{name} lacks integer score")
    if row["episodes"]<=0 or not 0<=row["successes"]<=row["episodes"]:
        raise OrchestrationError(f"{name} has invalid score")
    return Fraction(row["successes"],row["episodes"])

def validate_ptq(path: Path, protocol: dict[str,Any]) -> dict[str,Any]:
    d=jread(path); c=protocol["selection"]["pressure_candidates"]
    if d.get("selection_uses_heldout") is not False or d.get("protocol_sha256")!=protocol["sha256"]:
        raise OrchestrationError("PTQ selection is not bound to this protocol/development split")
    if d.get("pressure_candidates") not in (None,c): raise OrchestrationError("PTQ candidate list differs")
    arms=d.get("arms")
    if not isinstance(arms,dict): raise OrchestrationError("PTQ selection has no arms")
    for a in ["bf16",*c]:
        row=arms.get(a)
        if not isinstance(row,dict) or row.get("environment_pairing_verified") is not True:
            raise OrchestrationError(f"PTQ arm {a} is incomplete/unpaired")
        frac(row,a)
        if not row.get("checkpoint"): raise OrchestrationError(f"PTQ arm {a} has no checkpoint")
    audited_arms = {}
    for arm in ["bf16", *c]:
        folder = path.parent / arm
        audited = eval_audit(folder, protocol, "development", Path(arms[arm]["checkpoint"]))
        if (arms[arm]["successes"] != audited["successes"] or
                arms[arm]["episodes"] != audited["episodes"] or
                not _same_number(arms[arm].get("macro_success_rate"), audited["macro_success_rate"])):
            raise OrchestrationError(f"PTQ selection score disagrees with raw development evidence: {arm}")
        for filename in ("eval_manifest.json", "task_results.json", "summary.json"):
            key = f"{arm}/{filename}"
            if d.get("source_sha256", {}).get(key) != sha(folder / filename):
                raise OrchestrationError(f"PTQ selection source identity changed: {key}")
        audited_arms[arm] = audited
    require_pairing(audited_arms)
    metric = protocol["selection"].get("pressure_rule", {}).get("selection_metric", "micro_success_rate")
    def score(row: dict[str,Any], name: str) -> Fraction:
        if metric == "micro_success_rate" and row.get("micro_success_rate") is not None:
            return Fraction(str(row["micro_success_rate"]))
        return frac(row, name)
    base=score(arms["bf16"],"bf16")
    pressure_rule = protocol["selection"]["pressure_rule"]
    dr=Fraction(str(pressure_rule["min_drop_from_bf16"]))
    floor=Fraction(str(pressure_rule["min_absolute_success"]))
    window = pressure_rule.get("target_drop_from_bf16")
    maximum_drop = Fraction(str(window[1])) if isinstance(window, list) and len(window) == 2 else None
    qualifying=[a for a in c if base-score(arms[a],a)>=dr
                and (maximum_drop is None or base-score(arms[a],a)<=maximum_drop)
                and score(arms[a],a)>=floor]
    # v8 records the actual FP4 fraction per candidate so selection is based
    # on the declared compression objective rather than candidate list order.
    fractions = d.get("candidate_fp4_fraction", {})
    if not isinstance(fractions, dict):
        fractions = {}
    ranked = sorted(qualifying, key=lambda a: (float(fractions.get(a, -1.0)), -c.index(a)), reverse=True)
    expected_selected = ranked[0] if int(protocol["data"].get("version", 0)) >= 8 else (qualifying[-1] if qualifying else None)
    if not qualifying or d.get("selected_recipe")!=expected_selected:
        raise OrchestrationError("PTQ selection is not highest-FP4 qualifying candidate")
    for arm in ["bf16", *c]:
        ck_arm = need(arms[arm]["checkpoint"], f"{arm} checkpoint")
        recorded_id = arms[arm].get("model_identity")
        if recorded_id is None or recorded_id != model_id(ck_arm):
            raise OrchestrationError(f"PTQ selection model identity changed: {arm}")
        actual_fraction = 0.0 if arm == "bf16" else _candidate_fp4_fraction(ck_arm)
        if arm != "bf16" and abs(float(arms[arm].get("fp4_fraction", -1.0)) - actual_fraction) > 1e-12:
            raise OrchestrationError(f"PTQ selection FP4 fraction changed: {arm}")
    ck=need(arms[d["selected_recipe"]]["checkpoint"],"selected PTQ checkpoint")
    if not (ck/"ptq_recipe.json").is_file(): raise OrchestrationError(f"PTQ recipe missing: {ck}")
    return {**d,"selection_file":str(path),"selection_sha256":sha(path),
            "selected_ptq_checkpoint":str(ck),
            "selected_ptq_recipe_sha256":sha(ck/"ptq_recipe.json"),
            "verified_qualifying_candidates":qualifying}

def identity(path: Path) -> dict[str,Any]:
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)}
def model_id(path: Path) -> dict[str,Any]:
    p=need(path,"checkpoint/model")
    shards=[{"name":x.name,"bytes":x.stat().st_size,"sha256":sha(x)} for x in sorted(p.glob("*.safetensors"))]
    if not shards: raise OrchestrationError(f"no safetensors in {p}")
    meta={n:sha(p/n) for n in ("config.json","statistics.json","ptq_recipe.json","bake_manifest.json",
                                "category_ptq_recipe.json","category_bake_manifest.json",
                                "recovery_manifest.json","merge_manifest.json","runtime_metrics.json",
                                "model.safetensors.index.json") if (p/n).is_file()}
    return {"path":str(p),"metadata":meta,"shards":shards}

def capture_dataset_id(path: Path) -> dict[str,Any]:
    """Content identity for a replay dataset used by QAD/continuation."""
    p=need(path,"captured training dataset")
    files=sorted(x for x in p.rglob("*") if x.is_file())
    samples=[x for x in files if x.name.startswith("sample_") and x.suffix == ".pt"]
    if not samples:
        raise OrchestrationError(f"captured training dataset has no sample_*.pt: {p}")
    return {"root":str(p),"sample_count":len(samples),
            "files":[{"path":str(x.relative_to(p)),"bytes":x.stat().st_size,"sha256":sha(x)} for x in files]}

def validate_teacher_capture(path: Path, protocol_file: Path, teacher: Path, python: Path) -> dict[str,Any]:
    """Audit replay payloads on CPU in the configured training environment."""
    try:
        result = subprocess.run(
            [str(python), str(ROOT / "exp/verify_teacher_replay.py"),
             "--root", str(path), "--protocol-file", str(protocol_file),
             "--teacher", str(teacher)],
            check=True, capture_output=True, text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        report = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        details = getattr(exc, "stderr", str(exc))
        raise OrchestrationError("teacher replay CPU audit failed: " + details[-4000:]) from exc
    if report.get("status") != "verified" or report.get("task_count") != len(TASKS):
        raise OrchestrationError("teacher replay CPU audit did not verify all ten tasks")
    return report


def _candidate_fp4_fraction(path: Path) -> float:
    """Read the final checkpoint's NVFP4 fraction with category precedence."""
    category = path / "category_ptq_recipe.json"
    if category.is_file():
        data = jread(category)
        value = data.get("memory", {}).get("fraction_of_eligible_params", {}).get("nvfp4")
        if value is not None:
            return float(value)
    recipe = jread(path / "ptq_recipe.json")
    value = recipe.get("memory", {}).get("fraction_of_eligible_params", {}).get("nvfp4")
    if value is None:
        raise OrchestrationError(f"checkpoint has no NVFP4 fraction: {path}")
    return float(value)

def eval_audit(path: Path, protocol: dict[str,Any], purpose: str,
               checkpoint: Path | None = None) -> dict[str,Any]:
    """Verify raw logs, strict outcomes, counts and environment reset identities."""
    path = path.resolve()
    part=protocol["partitions"][purpose]; man=jread(path/"eval_manifest.json")
    rows=jread(path/"task_results.json"); summ=jread(path/"summary.json")
    if not all(isinstance(x, dict) for x in (man, rows, summ)):
        raise OrchestrationError(f"{purpose} evidence must contain JSON objects")
    for k,v in {"purpose":purpose,"seed":part["seed"],"episodes":part["episodes_per_task"],
                "init_state_indices":part["init_state_indices"],"tasks":list(TASKS),
                "task_count":10,"n_envs":1,"task_seed_stride":1000,
                "episode_seed_stride":1,"server_seed_offset":10000000,"settle_steps":10,"n_action_steps":8,
                "max_episode_steps":720,"protocol_sha256":protocol["sha256"]}.items():
        if man.get(k)!=v or (type(v) is int and type(man.get(k)) is not int):
            raise OrchestrationError(f"{purpose} manifest mismatch: {k}")
    if (type(man.get("init_state_indices")) is not list or
            any(type(x) is not int for x in man["init_state_indices"])):
        raise OrchestrationError("initial-state indices must be integers")
    if Path(man.get("out", "")).resolve() != path:
        raise OrchestrationError(f"evaluation output path moved or is incorrect: {path}")
    if Path(man.get("video_root", "")).resolve() != path / "videos":
        raise OrchestrationError(f"evaluation video_root is stale: {path}")
    if not Path(man.get("checkpoint", "")).is_dir():
        raise OrchestrationError(f"evaluation checkpoint path is unavailable: {path}")
    if checkpoint is not None and Path(man["checkpoint"]).resolve() != checkpoint.resolve():
        raise OrchestrationError(f"evaluation used another checkpoint: {path}")
    activation_audit = None
    if protocol["data"].get("w4a4") and checkpoint is not None:
        if (checkpoint / "ptq_recipe.json").is_file():
            activation_audit = activation_installation_audit(path, checkpoint, protocol)
        else:
            # A BF16 checkpoint has no PTQ recipe.  It must also have no
            # activation-only installer marker in any server log; otherwise
            # the nominal baseline was silently evaluated through W4A4.
            logs = sorted(path.glob("*.server.log"))
            marker = "[fp4vla] activation-only "
            offenders = [str(log) for log in logs if marker in log.read_text(errors="replace")]
            if offenders:
                raise OrchestrationError(
                    f"BF16 evaluation unexpectedly installed activation QDQ: {offenders[0]}")
            activation_audit = {
                "mode": "bf16",
                "activation_only_reports": 0,
                "server_logs": [{"path": str(log), "sha256": sha(log)} for log in logs],
            }
    if man.get("initial_state_protocol")!="libero10_official_bank_v1" or not man.get("protocol_file"):
        raise OrchestrationError(f"{purpose} manifest lacks official-bank protocol identity")
    protocol_file=Path(man["protocol_file"])
    if not protocol_file.is_file() or sha(protocol_file)!=protocol["sha256"]:
        raise OrchestrationError(f"{purpose} manifest protocol file/SHA is not the frozen protocol")
    if set(rows)!=set(TASKS) or type(summ.get("tasks_complete")) is not int or summ["tasks_complete"]!=10:
        raise OrchestrationError(f"{purpose} is incomplete")
    total_successes=0
    total_episodes=0
    pairing = []
    raw_sources = {}
    for ti,t in enumerate(TASKS):
        row=rows[t]
        if (not isinstance(row, dict) or type(row.get("returncode")) is not int or row["returncode"] != 0):
            raise OrchestrationError(f"{purpose}/{t} did not exit successfully")
        outcomes = row.get("results")
        if (not isinstance(outcomes, list) or len(outcomes) != part["episodes_per_task"] or
                any(type(x) is not bool for x in outcomes)):
            raise OrchestrationError(f"{purpose}/{t} has incomplete outcomes")
        expected_seed = part["seed"] + 1000 * ti
        _validate_declared_accounting(purpose, t, row, outcomes, expected_seed, True)
        log = path / (t + ".log")
        parsed = parse_log(log)
        if parsed["log_sha256"] != row.get("log_sha256"):
            raise OrchestrationError(f"{purpose}/{t} raw log SHA disagrees with task_results")
        for field in ("results", "episodes", "successes", "success_rate", "resets",
                      "raw_reset_count", "scored_reset_count", "ignored_auto_reset_count"):
            if parsed[field] != row.get(field):
                raise OrchestrationError(f"{purpose}/{t} raw log differs from task_results: {field}")
        resets = validate_resets(row, expected_seed, part["init_state_indices"])
        for ei, z in enumerate(resets):
            if any(type(z.get(k)) is not int for k in ("episode_index", "seed", "init_state_index", "settle_steps")):
                raise OrchestrationError(f"{purpose}/{t}/{ei} reset fields must be integers")
            pairing.append({"task": t, **{k:z[k] for k in (
                "episode_index", "seed", "init_state_index", "settle_steps",
                "initial_state_sha256", "restored_state_sha256", "init_state_bank_sha256")}})
        raw_sources[t] = identity(log)
        total_successes += sum(outcomes)
        total_episodes += len(outcomes)
    task_macro_success_rate = sum(
        rows[t]["successes"] / rows[t]["episodes"] for t in TASKS
    ) / 10
    micro_success_rate = total_successes / total_episodes
    # v3/v7 evidence was generated under the old, ambiguous summary field
    # semantics.  Version 8 locks ``macro_success_rate`` to the unweighted
    # mean of the ten task rates and exposes the micro rate separately.
    expected_macro = (task_macro_success_rate if int(protocol["data"].get("version", 0)) >= 8
                      else micro_success_rate)
    for key, expected in {"total_successes": total_successes, "total_episodes": total_episodes,
                          "tasks_complete": 10, "seed": part["seed"]}.items():
        if type(summ.get(key)) is not int or summ[key] != expected:
            raise OrchestrationError(f"{purpose} summary accounting disagrees: {key}")
    if (summ.get("purpose") != purpose or
            not _same_number(summ.get("macro_success_rate"), expected_macro)):
        raise OrchestrationError(f"{purpose} summary purpose/rate disagrees")
    if int(protocol["data"].get("version", 0)) >= 8:
        if not _same_number(summ.get("micro_success_rate"), micro_success_rate):
            raise OrchestrationError(f"{purpose} micro_success_rate disagrees")
        if "score_definition" not in summ:
            raise OrchestrationError(f"{purpose} summary lacks score_definition")
    result = {"evaluation_identity":{n:identity(path/n) for n in ("eval_manifest.json","task_results.json","summary.json")},
            "raw_log_identities": raw_sources,
            "pairing_sha256": hashlib.sha256(json.dumps(pairing, sort_keys=True).encode()).hexdigest(),
            "successes":total_successes,"episodes":total_episodes,
            "macro_success_rate":task_macro_success_rate,
            "micro_success_rate":micro_success_rate}
    if activation_audit is not None:
        result["activation_installation"] = activation_audit
    return result


def require_pairing(arms):
    if not arms or len({row["pairing_sha256"] for row in arms.values()}) != 1:
        raise OrchestrationError("development arms have different environment reset identities")


def activation_installation_audit(path: Path, checkpoint: Path, protocol: dict[str, Any]) -> dict[str, Any] | None:
    """Bind a W4A4 rollout to the server's activation installer report.

    ``eval_manifest.json`` records the requested environment, while the
    server log records what was actually wrapped.  PTQ and recovery arms must
    carry the latter evidence; BF16 checkpoints intentionally return ``None``.
    """
    checkpoint = checkpoint.resolve()
    recipe = checkpoint / "ptq_recipe.json"
    if not recipe.is_file():
        return None
    logs = sorted(path.glob("*.server.log"))
    reports: list[dict[str, Any]] = []
    source_hashes: list[dict[str, str]] = []
    marker = "[fp4vla] activation-only "
    for log in logs:
        digest = sha(log)
        for line in log.read_text(errors="replace").splitlines():
            if marker not in line:
                continue
            payload = line.split(marker, 1)[1].strip()
            try:
                report = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise OrchestrationError(f"invalid activation installer report: {log}: {exc}") from exc
            if not isinstance(report, dict):
                raise OrchestrationError(f"activation installer report is not an object: {log}")
            reports.append(report)
            source_hashes.append({"path": str(log), "sha256": digest})
    if not reports:
        raise OrchestrationError(f"W4A4 checkpoint has no activation-only server report: {path}")
    first = reports[0]
    fields = ("ordinary_total", "ordinary_w4a4", "ordinary_bf16",
              "category_total", "category_w4a4", "category_bf16")
    if any(any(report.get(field) != first.get(field) for field in fields) for report in reports[1:]):
        raise OrchestrationError(f"W4A4 activation installer counts differ across servers: {path}")
    if first.get("ordinary_w4a4", 0) != first.get("ordinary_total", -1):
        raise OrchestrationError(f"ordinary Linear activation coverage is incomplete: {path}")
    scope = protocol["data"].get("quantization_scope", {})
    expected_ordinary = int(scope.get("ordinary_linear_operators", 469))
    expected_category = int(scope.get("category_linear_layers", 7))
    if first.get("ordinary_total") != expected_ordinary or first.get("category_total") != expected_category:
        raise OrchestrationError(
            f"W4A4 activation installer scope differs from protocol: {path}; "
            f"got ordinary={first.get('ordinary_total')} category={first.get('category_total')}, "
            f"expected ordinary={expected_ordinary} category={expected_category}")
    if first.get("mode") != "all" or first.get("ste") is not False:
        raise OrchestrationError(f"W4A4 activation installer mode is not frozen all/STE-off: {path}")
    if (checkpoint / "category_ptq_recipe.json").is_file():
        if first.get("category_w4a4", -1) != first.get("category_total", -2) or first.get("category_bf16", -1) != 0:
            raise OrchestrationError(f"full category W4A4 checkpoint was not fully installed: {path}")
    elif first.get("category_bf16", -1) != first.get("category_total", -2):
        raise OrchestrationError(f"category layers unexpectedly use W4A4 without a category recipe: {path}")
    compact = {field: first.get(field) for field in fields}
    compact.update({"mode": first.get("mode"), "ste": first.get("ste"),
                    "report_count": len(reports), "server_logs": source_hashes})
    return compact

class Driver:
    def __init__(self,a:argparse.Namespace):
        self.a=a; self.protocol_path=need(a.protocol_file,"protocol")
        self.protocol=load_protocol(self.protocol_path); self.selection_path=need(a.ptq_selection,"PTQ selection")
        self.selection=validate_ptq(self.selection_path,self.protocol)
        self.run_dir=Path(a.run_dir).expanduser().resolve()
        self.art=self.run_dir/"artifacts"; self.work=self.run_dir/"work"; self.logs=self.run_dir/"logs"
        self.stages=self.run_dir/"stages"; self.receipts=self.run_dir/"cleanup_receipts"
        self.groot=Path(a.gr00t_repo or os.environ.get("GR00T_REPO",str(Path.home()/ "codebase/groot-fsdp2/Isaac-GR00T"))).expanduser().resolve()
        # Keep the requested venv launcher path intact.  ``Path.resolve()``
        # follows the launcher symlink to uv's bare interpreter, which drops
        # the GR00T/LIBERO environment and makes recovery fail before import.
        self.py=Path(a.python or os.environ.get("PTQAD_PYTHON",str(self.groot/".venv/bin/python"))).expanduser().absolute()
        self.sim=Path(a.rollout_python or os.environ.get("LIBERO_PYTHON",str(self.groot/".venv-libero/bin/python"))).expanduser().absolute()
        if not self.py.is_file(): raise OrchestrationError(f"training Python launcher not found: {self.py}")
        if not self.sim.is_file(): raise OrchestrationError(f"rollout Python launcher not found: {self.sim}")
        self.base=Path(a.base or os.environ.get("PTQAD_BASE",str(ROOT/"weights/GR00T-N1.7-LIBERO/libero_10"))).expanduser().resolve()
        self.dataset=Path(a.dataset or os.environ.get("QAD_DATASET",str(self.groot/"demo_data/libero_demo"))).expanduser().resolve()
        capture_arg = a.capture_dataset or os.environ.get("QAD_CAPTURE_DATASET")
        self.capture_dataset = Path(capture_arg).expanduser().resolve() if capture_arg else None
        if int(self.protocol["data"].get("version", 0)) >= 8 and self.capture_dataset is None:
            raise OrchestrationError(
                "v8 recovery requires --capture-dataset containing finalized BF16 teacher successes"
            )
        self.capture_dataset_identity = capture_dataset_id(self.capture_dataset) if self.capture_dataset else None
        self.teacher_capture_identity = (
            validate_teacher_capture(self.capture_dataset, self.protocol_path, self.base, self.py)
            if int(self.protocol["data"].get("version", 0)) >= 8 else None
        )
        s=self.protocol["selection"]; self.seed=int(s["train_seed"]); self.qsteps=int(s["qad_optimizer_steps"])
        self.csteps=int(s["continuation_optimizer_steps"]); self.rank=int(s["rank"]); self.alpha=float(s["alpha"])
        self.scope=str(s["recovery_scope"]); self.batch=int(s["effective_demo_batch"]); self.every=int(s["opd_every"])
        self.w4a4=bool(self.protocol["data"].get("w4a4", False))
        self.port=int(a.port_base); self.adopt=bool(a.adopt_complete); self.cleanup=bool(a.cleanup_duplicates)
        # Development selection remains strict by default.  This explicit
        # escape hatch only lets both OPD candidates reach the paired
        # held-out evaluation when development data cannot resolve a gain;
        # the selection record retains the actual comparisons and the paper
        # must not claim an OPD improvement unless held-out proves it.
        self.allow_opd_fallback=bool(getattr(a,"allow_opd_nonimprovement",False))
        self.state=self.load_state()

    def save(self): jwrite(self.run_dir/"run_manifest.json",self.state)
    def load_state(self):
        man=self.run_dir/"run_manifest.json"
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not man.exists():
            raise OrchestrationError(f"run directory lacks run_manifest.json: {self.run_dir}")
        self.run_dir.mkdir(parents=True,exist_ok=True)
        if man.exists():
            state=jread(man)
            fixed={"output_layout":"stable_paths_v2","protocol_sha256":self.protocol["sha256"],"selection_sha256":self.selection["selection_sha256"],
                   "base":str(self.base),"train_seed":self.seed,"qad_steps":self.qsteps,
                   "continuation_steps":self.csteps,"lora_scope":self.scope,"rank":self.rank,"alpha":self.alpha,
                   "capture_dataset_identity":self.capture_dataset_identity,
                   "teacher_capture_identity":self.teacher_capture_identity,
                   "w4a4":self.w4a4}
            for k,v in fixed.items():
                if state.get(k)!=v: raise OrchestrationError(f"resume identity changed: {k}")
            for p in (self.art,self.work,self.logs,self.stages,self.receipts):
                p.mkdir(parents=True,exist_ok=True)
            return state
        state={"format":"high_fp4_v3_orchestrator","output_layout":"stable_paths_v2","status":"initialized",
               "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
               "protocol_file":str(self.protocol_path),"protocol_sha256":self.protocol["sha256"],
               "protocol_version":self.protocol["data"].get("version"),
               "selection_file":str(self.selection_path),"selection_sha256":self.selection["selection_sha256"],
               "selected_recipe":self.selection["selected_recipe"],
               "selected_ptq_checkpoint":self.selection["selected_ptq_checkpoint"],
               "selected_ptq_recipe_sha256":self.selection["selected_ptq_recipe_sha256"],
               "base":str(self.base),"base_identity":model_id(self.base),"gr00t_repo":str(self.groot),
               "python":str(self.py),"rollout_python":str(self.sim),"dataset":str(self.dataset),
               "capture_dataset_identity":self.capture_dataset_identity,
               "teacher_capture_identity":self.teacher_capture_identity,
               "w4a4":self.w4a4,
               "train_seed":self.seed,"qad_steps":self.qsteps,"continuation_steps":self.csteps,
               "qad_learning_rates":self.protocol["selection"]["qad_learning_rates"],
               "opd_weights":self.protocol["selection"]["opd_weights"],"lora_scope":self.scope,
               "rank":self.rank,"alpha":self.alpha,"effective_demo_batch":self.batch,"opd_every":self.every,
               "stages":{},"selection_uses_heldout":False,"implementation_sha256":sha(Path(__file__))}
        jwrite(man,state)
        for p in (self.art,self.work,self.logs,self.stages,self.receipts): p.mkdir(parents=True,exist_ok=True)
        return state

    def freeze_check(self):
        if sha(self.protocol_path)!=self.protocol["sha256"]: raise OrchestrationError("protocol changed")
        if validate_ptq(self.selection_path,self.protocol)["selection_sha256"]!=self.selection["selection_sha256"]:
            raise OrchestrationError("PTQ selection changed")

    def mark(self,name:str,payload:dict[str,Any]):
        rec={"stage":name,"status":"complete","completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
             "protocol_file":str(self.protocol_path),"protocol_sha256":self.protocol["sha256"],
             "selection_sha256":self.selection["selection_sha256"],**payload}
        jwrite(self.stages/(sn(name)+".json"),rec); self.state["stages"][name]=rec
        self.state["last_completed_stage"]=name; self.save()

    def stage(self,name:str,action:Callable[[Path],dict[str,Any]],verify:Callable[[Path],dict[str,Any]])->Path:
        return self.stage_at(name, self.art / sn(name), action, verify)

    def stage_at(self, name, final, action, verify):
        """Publish a completion marker; never relocate manifests or payloads.

        GR00T, evaluation and capture manifests embed absolute paths. Moving a
        completed directory invalidates out/video/source-observation identities.
        Stable output paths and an atomic completion marker avoid rewriting raw
        evidence. An unmarked output is incomplete and cannot be reused unless
        --adopt-complete is explicitly supplied and every verifier passes.
        """
        self.freeze_check()
        marker=self.stages/(sn(name)+".json")
        legacy_work=self.work/sn(name)
        if legacy_work.exists():
            raise OrchestrationError(f"legacy moved-layout work output requires manual audit: {legacy_work}")
        if marker.exists():
            record=jread(marker)
            if (not final.exists() or record.get("status")!="complete" or
                    record.get("output") != str(final) or
                    record.get("protocol_sha256") != self.protocol["sha256"] or
                    record.get("selection_sha256") != self.selection["selection_sha256"]):
                raise OrchestrationError(f"stage inconsistent: {name}")
            audited=verify(final)
            if any(record.get(key) != value for key, value in audited.items()):
                raise OrchestrationError(f"completed stage evidence changed: {name}")
            return final
        if final.exists():
            if not self.adopt:
                raise OrchestrationError(f"incomplete stage {final}; inspect then --adopt-complete")
            audited=verify(final)
            self.mark(name,{"output":str(final),"adopted":True,**audited})
            return final
        final.parent.mkdir(parents=True,exist_ok=True)
        produced=action(final); audited=verify(final)
        self.mark(name,{"output":str(final),"adopted":False,**produced,**audited}); return final

    def run_logged(self,name,c, cwd, extra):
        log=self.logs/(sn(name)+".log")
        if log.exists(): raise OrchestrationError(f"refusing existing log: {log}")
        env=os.environ.copy()
        # Do not inherit a prior experiment's adapter, cache or capture budget.
        for key in list(env):
            if key.startswith(("QAD_", "OPD_")) or key in (
                    "GR00T_BASE_CKPT", "TRAIN_SEED", "PROTOCOL_FILE", "PTQAD_PROTOCOL_FILE"):
                env.pop(key)
        env.update(extra); env.setdefault("HF_HUB_OFFLINE","1")
        env.setdefault("TRANSFORMERS_OFFLINE","1")
        env.setdefault("PTQAD_LOCAL_HF_METADATA","1")
        env.setdefault("NO_ALBUMENTATIONS_UPDATE","1")
        with log.open("x") as f:
            f.write(f"UTC {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}\nCWD {cwd}\nCOMMAND {cmdtext(c)}\n")
            r=subprocess.run(c,cwd=str(cwd),env=env,stdout=f,stderr=subprocess.STDOUT)
            f.write(f"RETURN_CODE {r.returncode}\n")
            if r.returncode==0: f.write(f"COMPLETED_UTC {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}\n")
        if r.returncode: raise OrchestrationError(f"{name} failed ({r.returncode}); inspect {log}")

    def train(self,work,name,base,lr,steps,initial=None,weight=0.,cache=None):
        e={"GR00T_BASE_CKPT":str(base),"QAD_DATASET":str(self.dataset),"QAD_OUT":str(work),
           "QAD_STEPS":str(steps),"QAD_SAVE_STEPS":str(min(100, steps)),
           "QAD_SAVE_TOTAL_LIMIT":"2","QAD_GLOBAL_BATCH":str(self.batch),
           "QAD_MICRO_BATCH":"1","QAD_LORA_R":str(self.rank),"QAD_LORA_ALPHA":str(self.alpha),
           "QAD_LORA_SCOPE":self.scope,"QAD_LR":str(lr),"QAD_OPD_MSE_W":str(weight),
           "OPD_EVERY":str(self.every),"TRAIN_SEED":str(self.seed),"PROTOCOL_FILE":str(self.protocol_path),
           "QAD_ACTIVATION_CHECKPOINTING":"1",
           "QAD_W4A4":"1" if self.w4a4 else "0"}
        if self.capture_dataset:
            e["QAD_CAPTURE_DATASET"] = str(self.capture_dataset)
            e["QAD_CAPTURE_DATASET_SHA256"] = hashlib.sha256(
                json.dumps(self.capture_dataset_identity, sort_keys=True).encode()).hexdigest()
        if initial: e["QAD_INIT_ADAPTER"]=str(initial)
        if cache: e["OPD_CACHE_PATH"]=str(cache)
        jwrite(work/"orchestrator_training_request.json",{
            "environment":e,"base_identity":model_id(base),
            "initial_adapter_identity":model_id(initial) if initial else None,
            "cache_identity":identity(cache) if cache else None,
            "capture_dataset_identity":self.capture_dataset_identity,
            "teacher_capture_identity":self.teacher_capture_identity,
            "protocol_sha256":self.protocol["sha256"]})
        self.run_logged(name,[str(self.py),str(ROOT/"rl/lora_qad.py")],self.groot,e)
        return {"base":str(base),"learning_rate":lr,"optimizer_steps":steps,"opd_weight":weight,
                "initial_adapter":str(initial) if initial else None,"teacher_cache":str(cache) if cache else None}

    def train_verify(self,p,steps,initial=None,lr=None,weight=0.,cache=None):
        m=jread(p/"runtime_metrics.json"); rec=jread(p/"recovery_manifest.json")
        if m.get("status")!="completed" or m.get("global_steps")!=steps or m.get("requested_optimizer_steps")!=steps:
            raise OrchestrationError(f"training does not prove {steps} steps: {p}")
        base=Path(self.selection["selected_ptq_checkpoint"])
        expected={"base":str(base),"rank":self.rank,"alpha":self.alpha,"scope":self.scope,
                  "train_seed":self.seed,"protocol_sha256":self.protocol["sha256"],
                  "optimizer_resumed":False,"probe_weight":weight,"probe_every":self.every,
                  "micro_batch":1,"effective_global_batch":self.batch,"gradient_accumulation_steps":self.batch}
        if any(rec.get(k)!=v for k,v in expected.items()) or rec.get("optimizer_resumed") is not False:
            raise OrchestrationError("training manifest differs from frozen recovery settings")
        if self.w4a4 and (rec.get("w4a4_enabled") is not True or
                          rec.get("execution_mode") != "W4A4 numerical QDQ + BF16 LoRA residual"):
            raise OrchestrationError("training activation format differs from frozen protocol")
        if rec.get("initial_adapter") != (str(initial) if initial else None):
            raise OrchestrationError("continuation is not exact-A/B with a fresh optimizer")
        for name,field in (("config.json","base_config_sha256"),("statistics.json","base_statistics_sha256"),
                           ("ptq_recipe.json","base_recipe_sha256"),
                           ("category_ptq_recipe.json","base_category_recipe_sha256"),
                           ("category_bake_manifest.json","base_category_manifest_sha256")):
            if not (base / name).is_file():
                if name.startswith("category_"):
                    continue
                raise OrchestrationError(f"training base metadata is missing: {name}")
            if rec.get(field)!=sha(base/name):
                raise OrchestrationError(f"training base metadata changed: {name}")
        if rec.get("probe_cache")!=(str(cache) if cache else None) or rec.get("probe_cache_sha256")!=(sha(cache) if cache else None):
            raise OrchestrationError("training did not use the frozen teacher cache")
        expected_capture = str(self.capture_dataset) if self.capture_dataset else None
        expected_capture_sha = (hashlib.sha256(
            json.dumps(self.capture_dataset_identity, sort_keys=True).encode()).hexdigest()
            if self.capture_dataset else None)
        if (rec.get("capture_dataset") != expected_capture or
                rec.get("capture_dataset_sha256") != expected_capture_sha):
            raise OrchestrationError("training did not use the frozen capture dataset")
        request=jread(p/"orchestrator_training_request.json")
        if (request.get("protocol_sha256")!=self.protocol["sha256"] or
                request.get("environment",{}).get("QAD_OUT")!=str(p) or
                float(request.get("environment",{}).get("QAD_LR",-1))!=lr or
                request.get("base_identity")!=model_id(base) or
                request.get("initial_adapter_identity")!=(model_id(initial) if initial else None) or
                request.get("cache_identity")!=(identity(cache) if cache else None) or
                request.get("capture_dataset_identity")!=self.capture_dataset_identity):
            raise OrchestrationError("training invocation/path/source identities changed")
        checkpoint=p/f"checkpoint-{steps}"
        if sha(checkpoint/"recovery_manifest.json")!=sha(p/"recovery_manifest.json"):
            raise OrchestrationError("saved checkpoint recovery manifest differs from training")
        return {"checkpoint_identity":model_id(checkpoint),"recovery_manifest_sha256":sha(p/"recovery_manifest.json"),
                "training_request_identity":identity(p/"orchestrator_training_request.json"),
                "runtime_identity":identity(p/"runtime_metrics.json")}

    def merge(self,work,name,base,ckpt,steps):
        self.run_logged(name,[str(self.py),str(ROOT/"rl/lora_merge_bake.py"),"--base",str(base),"--ckpt",str(ckpt),
                             "--out",str(work),"--rank",str(self.rank),"--alpha",str(self.alpha)],ROOT,{})
        return {"base":str(base),"training_checkpoint":str(ckpt),"optimizer_steps":steps}
    def merge_verify(self,p):
        m=jread(p/"merge_manifest.json")
        if m.get("status")!="complete" or not (p/"ptq_recipe.json").is_file(): raise OrchestrationError(f"bad merged model: {p}")
        base=Path(self.selection["selected_ptq_checkpoint"])
        checkpoint=need(m.get("training_checkpoint",""),"merge training checkpoint")
        recovery=need(m.get("recovery_manifest_source",""),"merge recovery manifest")
        if (m.get("base")!=str(base) or recovery != checkpoint/"recovery_manifest.json" or
                m.get("recovery_manifest_sha256")!=sha(recovery) or
                sha(p/"recovery_manifest.json")!=sha(recovery) or
                sha(p/"ptq_recipe.json")!=sha(base/"ptq_recipe.json") or
                m.get("rank")!=self.rank or m.get("alpha")!=self.alpha):
            raise OrchestrationError("merge base/adapter metadata or absolute source paths differ")
        output_weights={f.name:{"bytes":f.stat().st_size,"sha256":sha(f)} for f in p.glob("*.safetensors")}
        if not output_weights or m.get("output_weights")!=output_weights:
            raise OrchestrationError("merge output weight identity differs")
        return {"model_identity":model_id(p),"merge_manifest_sha256":sha(p/"merge_manifest.json")}

    def evaluate(self,work,name,ckpt,purpose,offset=0,collection=None):
        part=self.protocol["partitions"][purpose]
        c=[str(self.py),str(ROOT/"eval/run_recovery_eval.py"),"--checkpoint",str(ckpt),"--out",str(work),
           "--purpose",purpose,"--seed",str(part["seed"]),"--episodes",str(part["episodes_per_task"]),
           "--gr00t",str(self.groot),"--server-python",str(self.py),"--rollout-python",str(self.sim),
           "--port",str(self.port+offset),"--protocol-file",str(self.protocol_path)]
        if collection: c += ["--collection-manifest",str(collection)]
        # The protocol is W4A4 for the quantized arms, but the BF16 baseline
        # is evaluated by the same driver.  Do not let the protocol-wide flag
        # silently install activation QDQ on the baseline: identify the base
        # checkpoint explicitly and force a clean BF16 environment there.
        is_bf16 = Path(ckpt).resolve() == self.base.resolve()
        use_w4a4 = self.w4a4 and not is_bf16
        adapter_mode = use_w4a4 and (Path(ckpt) / "merge_manifest.json").is_file()
        env={"PROTOCOL_FILE":str(self.protocol_path),"QAD_DATASET":str(self.dataset),
             "FP4VLA_QUANT":"0",
             "FP4VLA_W4A4":"1" if use_w4a4 else "0",
             "FP4VLA_W4A4_ADAPTER":"1" if adapter_mode else "0"}
        if purpose == "collection":
            # Four calls per capture provide 16 probes even when a task
            # succeeds before the nominal trajectory call budget.
            env.update(OPD_CAPTURE_EVERY="4",OPD_CAPTURE_PER_TASK="16",OPD_CAPTURE_LIMIT="160")
        self.run_logged(name,c,self.groot,env)
        return {"checkpoint":str(ckpt),"purpose":purpose}
    def eval_verify(self,p,purpose,checkpoint=None): return eval_audit(p,self.protocol,purpose,checkpoint)

    def cache(self,work,collection):
        # The frozen orchestration budget is 16 observations/task, 160 total;
        # v3 supplies ten tasks and four collection episodes/task. Never accept
        # a silently truncated teacher cache if the collector captured fewer.
        count=self.cache_count()
        model=Path(jread(collection/"eval_manifest.json")["checkpoint"])
        self.capture_audit(collection,model)
        self.run_logged("teacher-cache",[str(self.py),str(ROOT/"rl/opd_probe_cache.py"),"--teacher",str(self.base),
            "--input-dir",str(collection/"observations"),"--count",str(count),"--out",str(work/"teacher_probes.pt"),
            "--dataset",str(self.dataset),"--seed",str(self.seed)],self.groot,{})
        return {"collection_observations":str(collection/"observations"),"requested_count":count}
    def cache_count(self):
        return len(TASKS)*self.protocol["partitions"]["collection"]["episodes_per_task"]*4

    def capture_audit(self,collection,model):
        expected_per_task=self.cache_count()//len(TASKS)
        sources={}
        for task in TASKS:
            folder=collection/"observations"/task
            manifest=jread(folder/"capture_manifest.json")
            counts=jread(folder/"capture_counts.json")
            if (manifest.get("source_kind")!="student_rollout" or
                    manifest.get("student_checkpoint")!=str(model) or
                    manifest.get("student_statistics_sha256")!=sha(model/"statistics.json") or
                    manifest.get("student_config_sha256")!=sha(model/"config.json") or
                    manifest.get("every_server_calls")!=4 or
                    manifest.get("per_task_limit")!=16 or manifest.get("total_limit")!=160):
                raise OrchestrationError(f"capture does not use frozen winning QAD/settings: {task}")
            paths=sorted(folder.glob("sample_*.pt"))
            if (len(paths)!=expected_per_task or type(counts.get("saved")) is not int or
                    counts["saved"]!=expected_per_task):
                raise OrchestrationError(f"capture is short: {task}; require {expected_per_task} observations")
            for source in paths:
                sources[str(source.resolve())]=identity(source.resolve())
        return sources

    def cache_verify(self,p,collection,model):
        c,m=p/"teacher_probes.pt",p/"teacher_probes.json"
        if not c.is_file() or not m.is_file(): raise OrchestrationError("teacher cache incomplete")
        d=jread(m)
        expected=self.cache_count()
        if (d.get("source_kind")!="student_rollout" or d.get("teacher")!=str(self.base) or
                type(d.get("count")) is not int or d["count"]!=expected or
                type(d.get("requested_count")) is not int or d["requested_count"]!=expected or
                d.get("seed")!=self.seed):
            raise OrchestrationError("teacher cache provenance invalid")
        sources=self.capture_audit(collection,model)
        actual_sources=d.get("source_observation_files")
        if (not isinstance(actual_sources,list) or len(actual_sources)!=expected or
                len({row["path"] for row in actual_sources})!=expected or
                {row["path"]:row for row in actual_sources}!=sources):
            raise OrchestrationError("teacher cache sources do not match the complete frozen collection")
        for name,field in (("config.json","teacher_config_sha256"),("statistics.json","teacher_statistics_sha256")):
            if d.get(field)!=sha(self.base/name):
                raise OrchestrationError(f"teacher cache metadata differs from BF16 teacher: {name}")
        weights={f.name:{"bytes":f.stat().st_size,"sha256":sha(f)} for f in sorted(self.base.glob("*.safetensors"))}
        if d.get("teacher_weights") != weights:
            raise OrchestrationError("teacher cache weight identity differs from BF16 teacher")
        student_weights = {f.name:{"bytes":f.stat().st_size,"sha256":sha(f)}
                           for f in sorted(model.glob("*.safetensors"))}
        if (d.get("student_checkpoint") != str(model.resolve()) or
                d.get("student_config_sha256") != sha(model/"config.json") or
                d.get("student_statistics_sha256") != sha(model/"statistics.json") or
                d.get("student_weights") != student_weights):
            raise OrchestrationError("teacher cache student checkpoint identity differs from frozen QAD model")
        # Load only on CPU in a separate process. JSON sidecars alone cannot
        # prove the serialized cache contains 160 samples or agrees with them.
        env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=""
        try:
            subprocess.run([str(self.py),str(ROOT/"exp/verify_teacher_cache_cpu.py"),
                            "--cache",str(c),"--metadata",str(m),"--student-checkpoint",str(model),
                            "--seed",str(self.seed),"--expected-count",str(expected)],
                           env=env,check=True,capture_output=True,text=True)
        except subprocess.CalledProcessError as exc:
            raise OrchestrationError("CPU teacher-cache payload verification failed: "+exc.stderr[-4000:]) from exc
        return {"cache_identity":identity(c),"metadata_identity":identity(m),"count":int(d["count"])}

    def cleanup_dupes(self,label,root,ckpt):
        if not self.cleanup: return
        receipt=self.receipts/(sn(label)+".json")
        if receipt.exists():
            prior=jread(receipt)
            for row in prior["deleted_files"]:
                if Path(row["deleted"]).exists() or sha(Path(row["retained"]))!=row["sha256"]:
                    raise OrchestrationError("completed duplicate-cleanup receipt no longer matches disk")
            return
        try:
            running=subprocess.check_output(["ps","-eo","pid=,args="],text=True).splitlines()
        except (OSError,subprocess.CalledProcessError):
            running=[]
        active=[]
        for line in running:
            text=line.strip()
            # Ignore the process-list command itself (for example a user's
            # grep/rg inspection whose search pattern contains a stage name).
            if not text or "grep" in text or "rg " in text or "run_high_fp4_v3.py" in text:
                continue
            if any(x in text for x in ("run_recovery_eval.py", "serve_recovery.py",
                                       "lora_qad.py", "lora_merge_bake.py")):
                active.append(text)
        if active:
            raise OrchestrationError(f"active training/evaluation process prevents cleanup: {active}")
        deleted=[]
        for p in sorted(root.glob("*.safetensors")):
            q=ckpt/p.name
            if p.is_symlink() or not p.is_file() or not q.is_file(): continue
            a,b=sha(p),sha(q)
            if p.stat().st_size!=q.stat().st_size or a!=b: continue
            size=p.stat().st_size; p.unlink(); deleted.append({"deleted":str(p),"retained":str(q),"bytes":size,"sha256":a})
        jwrite(receipt,{"format":"verified_training_root_duplicate_cleanup_v1",
            "label":label,"training_root":str(root),"retained_checkpoint":str(ckpt),"deleted_files":deleted,
            "deleted_bytes":sum(x["bytes"] for x in deleted),"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "note":"Only exact root shard duplicates were deleted; evidence, checkpoints and merged models retained."})

    def select(self,work,kind,rows,control=None):
        d=self.selection_report(kind,rows,control=control)
        jwrite(work/"selection.json",d); return {"selection_identity":identity(work/"selection.json")}

    def _selection_model(self, kind, key):
        """Return the merged checkpoint represented by a recovery candidate."""
        value = float(key)
        if kind == "qad":
            return self.art / ("merge_qad_lr_" + sn(str(value)))
        if kind == "opd":
            return self.art / ("merge_opd_025" if math.isclose(value, .25) else "merge_opd_100")
        raise OrchestrationError(f"unknown recovery selection kind: {kind}")

    def selection_report(self,kind,rows,control=None):
        if kind not in {"qad", "opd"}:
            raise OrchestrationError(f"unknown recovery selection kind: {kind}")
        strict_recovery = int(self.protocol["data"].get("version", 0)) >= 8
        field="learning_rate" if kind=="qad" else "opd_weight"
        declared=self.protocol["selection"]["qad_learning_rates" if kind=="qad" else "opd_weights"]
        if set(rows) != set(declared) or len(rows) != 2:
            raise OrchestrationError("recovery selection candidates differ from frozen protocol")
        control_name = None
        if kind == "opd":
            # OPD is only a meaningful claim when it is compared against the
            # same-budget continued-QAD control.  Requiring the control at the
            # API boundary prevents a caller from accidentally publishing an
            # OPD winner selected against OPD-only candidates.
            control_name = str(self.protocol["selection"].get("same_budget_control", "continued_qad"))
            if control is None:
                raise OrchestrationError(
                    "OPD selection requires the continued-QAD control evaluation"
                )
            control = Path(control)
            if not control.is_dir():
                raise OrchestrationError(f"OPD control evaluation does not exist: {control}")
        values={}
        for key,p in rows.items():
            model=self._selection_model(kind, key)
            eval_protocol=self.protocol
            manifest_path=Path(p)/"eval_manifest.json"
            if manifest_path.is_file():
                manifest=jread(manifest_path)
                if manifest.get("protocol_sha256") != self.protocol["sha256"]:
                    # Exploratory v5 audits completed v4 development evidence
                    # under the protocol recorded by that evidence.
                    eval_protocol=load_protocol(manifest.get("protocol_file",""))
                    if eval_protocol["partitions"] != self.protocol["partitions"]:
                        raise OrchestrationError("development evaluation partitions differ from recovery protocol")
            values[str(key)]={field:key,**eval_audit(p,eval_protocol,"development",model),
                             "evaluation_path":str(p),"selection_source":"development_only"}
            if strict_recovery:
                values[str(key)]["model_identity"] = model_id(model)
        reference_path = self.selection_path.parent / "bf16"
        if not reference_path.is_dir():
            # Category development stores the reference below its explicit
            # partition directory; retain compatibility with the older flat
            # layout without weakening the evidence audit.
            reference_path = self.selection_path.parent / "development" / "bf16"
        reference_protocol=self.protocol
        ref_manifest=reference_path/"eval_manifest.json"
        if ref_manifest.is_file() and jread(ref_manifest).get("protocol_sha256") != self.protocol["sha256"]:
            reference_protocol=load_protocol(jread(ref_manifest).get("protocol_file",""))
            if reference_protocol["partitions"] != self.protocol["partitions"]:
                raise OrchestrationError("BF16 development partitions differ from recovery protocol")
        reference=eval_audit(reference_path, reference_protocol, "development", self.base)
        # Version 8 recovery claims are conditional on a real PTQ baseline.
        # The pressure wrapper stores the selected candidate's development
        # evaluation beside selection.json; audit it here instead of allowing
        # recovery to proceed against an implicit or stale dataset.
        ptq_reference = None
        if strict_recovery:
            recipe = self.selection.get("selected_recipe")
            if not isinstance(recipe, str) or not recipe:
                raise OrchestrationError("v8 recovery selection has no selected pressure recipe")
            ptq_path = self.selection_path.parent / recipe
            if not ptq_path.is_dir():
                raise OrchestrationError(
                    f"v8 recovery cannot find selected PTQ development evidence: {ptq_path}"
                )
            ptq_model = Path(self.selection["selected_ptq_checkpoint"])
            ptq_reference = {
                "evaluation_path": str(ptq_path),
                "selection_source": "development_only",
                "model_identity": model_id(ptq_model),
                **eval_audit(ptq_path, self.protocol, "development", ptq_model),
            }
            require_pairing({"bf16": reference, "ptq": ptq_reference})
        control_value = None
        if kind == "opd":
            control_model = self.art / ("merge_" + sn(control_name))
            control_protocol = self.protocol
            control_manifest = control / "eval_manifest.json"
            if control_manifest.is_file() and jread(control_manifest).get("protocol_sha256") != self.protocol["sha256"]:
                control_protocol = load_protocol(jread(control_manifest).get("protocol_file", ""))
                if control_protocol["partitions"] != self.protocol["partitions"]:
                    raise OrchestrationError("continued-QAD control partitions differ from recovery protocol")
            control_value = {
                "evaluation_path": str(control),
                "selection_source": "development_only",
                **eval_audit(control, control_protocol, "development", control_model),
            }
            if strict_recovery:
                control_value["model_identity"] = model_id(control_model)
        require_pairing({"bf16":reference,**values,**({control_name:control_value} if control_value else {})})
        metric = self.protocol["selection"].get("recovery_selection_metric", "micro_success_rate")
        def selection_score(row):
            # eval_audit enforces the same episode count for all ten tasks,
            # so macro and micro have the same exact rational value. Avoid
            # turning floating-point summation roundoff into a strict gain.
            if strict_recovery and metric in {"macro_success_rate", "micro_success_rate"}:
                return frac(row, field)
            value = row.get(metric)
            return Fraction(str(value)) if value is not None else frac(row, field)
        winner=sorted(values.values(),key=lambda x:(-selection_score(x),x[field]))[0]
        if kind == "qad" and ptq_reference is not None:
            ptq_score = selection_score(ptq_reference)
            selected_score = selection_score(winner)
            if selected_score <= ptq_score:
                raise OrchestrationError(
                    "no QAD learning-rate candidate is strictly above the PTQ development baseline"
                )
        d={"format":"w4a4_recovery_v11_"+("qad_lr" if kind=="qad" else "opd")+"_"+("selection"),
           "protocol_file":str(self.protocol_path),"protocol_sha256":self.protocol["sha256"],
           "selection_uses_heldout":False,"criterion":f"highest development {metric}; ties choose lower "+("learning rate" if kind=="qad" else "OPD weight"),
           "environment_pairing_verified":True,"reference_bf16_evaluation":str(reference_path),
           "candidates":values,"selected_"+field:winner[field]}
        if ptq_reference is not None:
            d.update({
                "ptq_selection_identity": identity(self.selection_path),
                "ptq_reference_evaluation": ptq_reference,
                "ptq_reference_score": float(selection_score(ptq_reference)),
                "selected_qad_or_candidate_score": float(selection_score(winner)),
                "recovery_above_ptq": selection_score(winner) > selection_score(ptq_reference),
            })
            if kind == "qad":
                d["qad_selection_gate"] = {
                    "require_qad_strictly_above_ptq": True,
                    "metric": metric,
                }
        if kind == "opd":
            control_score = selection_score(control_value)
            selected_score = selection_score(winner)
            qad_reference = None
            if strict_recovery:
                # Re-read and re-audit the persisted QAD selection on every
                # verification.  An in-memory value cannot establish the
                # frozen baseline after a process restart.
                qad_selection_dir = self.art / "select_qad_lr"
                qad_selection_file = qad_selection_dir / "selection.json"
                if not qad_selection_file.is_file():
                    raise OrchestrationError(
                        "v8 OPD selection has no frozen QAD development selection"
                    )
                self.select_verify(qad_selection_dir, "qad")
                selection = jread(qad_selection_file)
                selected_lr = selection.get("selected_learning_rate")
                candidates = selection.get("candidates", {})
                qad_reference = candidates.get(str(selected_lr))
                if not isinstance(qad_reference, dict) or not qad_reference.get("evaluation_path"):
                    raise OrchestrationError(
                        "v8 OPD selection cannot resolve the selected QAD development evaluation"
                    )
                qad_reference = {
                    "evaluation_path": str(qad_reference["evaluation_path"]),
                    "selection_source": "development_only",
                    "model_identity": model_id(self._selection_model("qad", selected_lr)),
                    **eval_audit(Path(qad_reference["evaluation_path"]), self.protocol,
                                 "development", self._selection_model("qad", selected_lr)),
                }
                require_pairing({"bf16": reference, "qad": qad_reference,
                                 control_name: control_value, **values})
            if selected_score < control_score or (strict_recovery and selected_score == control_score):
                if not self.allow_opd_fallback:
                    raise OrchestrationError(
                        "best OPD candidate is below continued-QAD control or fails its strict v8 gain gate"
                    )
            if qad_reference is not None:
                qad_score = selection_score(qad_reference)
                if selected_score <= qad_score and not self.allow_opd_fallback:
                    raise OrchestrationError(
                        "best OPD candidate is not strictly above the selected QAD baseline"
                    )
            d.update({
                "control_arm": control_name,
                "control_evaluation": control_value,
                "opd_selection_gate": {
                    "control_arm": control_name,
                    "require_opd_not_below_control": True,
                    "metric": metric,
                },
                "continued_qad_control_score": float(control_score),
                "selected_opd_score": float(selected_score),
                "opd_not_below_control": True,
                "opd_beats_continued_qad": selected_score > control_score,
                "opd_gate_override": bool(self.allow_opd_fallback),
            })
            if qad_reference is not None and not self.allow_opd_fallback:
                d["opd_selection_gate"].update({
                    "require_opd_strictly_above_control": True,
                    "require_opd_strictly_above_qad": True,
                })
                d.update({
                    "qad_selection_identity": identity(qad_selection_file),
                    "selected_qad_learning_rate": selected_lr,
                    "qad_reference_evaluation": qad_reference,
                    "qad_reference_score": float(selection_score(qad_reference)),
                    "opd_beats_qad": selected_score > selection_score(qad_reference),
                })
        return d
    def select_verify(self,p,kind):
        d=jread(p/"selection.json")
        if d.get("protocol_sha256")!=self.protocol["sha256"] or d.get("selection_uses_heldout") is not False:
            raise OrchestrationError("selection is not development-only")
        if not isinstance(d.get("candidates"),dict) or len(d["candidates"])!=2: raise OrchestrationError("selection needs two candidates")
        field="learning_rate" if kind=="qad" else "opd_weight"
        rows={row[field]:Path(row["evaluation_path"]) for row in d["candidates"].values()}
        control = None
        if kind == "opd":
            expected_control = str(self.protocol["selection"].get("same_budget_control", "continued_qad"))
            expected_metric = self.protocol["selection"].get("recovery_selection_metric", "micro_success_rate")
            expected_gate = {
                "control_arm": expected_control,
                "require_opd_not_below_control": True,
                "metric": expected_metric,
            }
            override = bool(d.get("opd_gate_override", False))
            if int(self.protocol["data"].get("version", 0)) >= 8 and not override:
                expected_gate.update({
                    "require_opd_strictly_above_control": True,
                    "require_opd_strictly_above_qad": True,
                })
            if (d.get("control_arm") != expected_control or
                    d.get("opd_selection_gate") != expected_gate or
                    not d.get("opd_not_below_control") or
                    (override != self.allow_opd_fallback) or
                    (int(self.protocol["data"].get("version", 0)) >= 8 and not override and
                     not d.get("opd_beats_qad"))):
                raise OrchestrationError("OPD selection is missing the continued-QAD control gate")
            control_rec = d.get("control_evaluation")
            if not isinstance(control_rec,dict) or not control_rec.get("evaluation_path"):
                raise OrchestrationError("OPD selection has no auditable control evaluation")
            control = Path(control_rec["evaluation_path"])
        if d != self.selection_report(kind,rows,control=control):
            raise OrchestrationError("recorded selection differs from recomputed development scores/tie-break")
        return {"selection_identity":identity(p/"selection.json")}

    def run(self,until):
        if self.a.validate_only:
            return {"status":"validated_only","protocol_sha256":self.protocol["sha256"],"selection_sha256":self.selection["selection_sha256"],
                    "selected_recipe":self.selection["selected_recipe"],"selected_ptq_checkpoint":self.selection["selected_ptq_checkpoint"]}
        base=Path(self.selection["selected_ptq_checkpoint"]); lrs=[float(x) for x in self.protocol["selection"]["qad_learning_rates"]]
        trains,models,devs={},{},{}
        for i,lr in enumerate(lrs):
            tag="qad_lr_"+sn(str(lr))
            tr=self.stage("train_"+tag,lambda w,lr=lr,tag=tag:self.train(w,"train_"+tag,base,lr,self.qsteps),
                          lambda p,lr=lr:self.train_verify(p,self.qsteps,lr=lr))
            ck=tr/f"checkpoint-{self.qsteps}"; trains[lr]=ck; self.cleanup_dupes("train_"+tag,tr,ck)
            models[lr]=self.stage("merge_"+tag,lambda w,tag=tag,ck=ck:self.merge(w,"merge_"+tag,base,ck,self.qsteps),self.merge_verify)
            devs[lr]=self.stage("dev_"+tag,lambda w,model=models[lr],tag=tag,i=i:self.evaluate(w,"dev_"+tag,model,"development",i),
                                  lambda p,model=models[lr]:self.eval_verify(p,"development",model))
        if until=="qad_dev": return self.state
        qs=self.stage("select_qad_lr",lambda w:self.select(w,"qad",devs),
                      lambda p:self.select_verify(p,"qad"))
        qsd=jread(qs/"selection.json"); win_lr=float(qsd["selected_learning_rate"])
        qad_model,qad_ck=models[win_lr],trains[win_lr]
        if until=="qad_selection": return self.state
        coll=self.stage("collection_qad",lambda w:self.evaluate(w,"collection_qad",qad_model,"collection",2),
                        lambda p:self.eval_verify(p,"collection",qad_model))
        cache_stage=self.stage("teacher_cache",lambda w:self.cache(w,coll),
                               lambda p:self.cache_verify(p,coll,qad_model)); cache=cache_stage/"teacher_probes.pt"
        rec={}
        for arm,weight in (("continued_qad",0.),("opd_025",.25),("opd_100",1.)):
            tr=self.stage("train_"+arm,lambda w,arm=arm,weight=weight:self.train(w,"train_"+arm,base,win_lr,self.csteps,qad_ck,weight,cache if weight else None),
                          lambda p,weight=weight:self.train_verify(p,self.csteps,qad_ck,win_lr,weight,cache if weight else None))
            ck=tr/f"checkpoint-{self.csteps}"; self.cleanup_dupes("train_"+arm,tr,ck)
            rec[arm]=self.stage("merge_"+arm,lambda w,arm=arm,ck=ck:self.merge(w,"merge_"+arm,base,ck,self.csteps),self.merge_verify)
        devrec={}
        for i,arm in enumerate(("continued_qad","opd_025","opd_100")):
            devrec[arm]=self.stage("dev_"+arm,lambda w,arm=arm,i=i:self.evaluate(w,"dev_"+arm,rec[arm],"development",3+i),
                                    lambda p,arm=arm:self.eval_verify(p,"development",rec[arm]))
        if until=="recovery_dev": return self.state
        osel=self.stage("select_opd_weight",lambda w:self.select(
                            w,"opd",{.25:devrec["opd_025"],1.:devrec["opd_100"]},
                            control=devrec["continued_qad"]),
                         lambda p:self.select_verify(p,"opd"))
        od=jread(osel/"selection.json"); win_w=float(od["selected_opd_weight"]); win_arm="opd_025" if math.isclose(win_w,.25) else "opd_100"
        if until=="opd_selection": return self.state
        round_dir=self.art/"heldout_round"; round_dir.mkdir(parents=True,exist_ok=True); cm=coll/"eval_manifest.json"
        final=(("bf16",self.base,10),("ptq",base,11),("qad",qad_model,12),("continued_qad",rec["continued_qad"],13),("qad_opd",rec[win_arm],14))
        for arm,model,off in final:
            name="heldout_"+arm; out=round_dir/name
            self.stage_at(name,out,
                lambda p,name=name,model=model,off=off:self.evaluate(p,name,model,"heldout",off,cm),
                lambda p,model=model:{"checkpoint_identity":model_id(model),**self.eval_verify(p,"heldout",model)})
        comparison=round_dir/"paired_comparison.json"
        if not comparison.exists():
            self.run_logged("compare_heldout",[str(self.py),str(ROOT/"eval/compare_recovery.py"),"--round",str(round_dir)],ROOT,{"PROTOCOL_FILE":str(self.protocol_path)})
        if jread(comparison) != compare_round(round_dir):
            raise OrchestrationError("heldout comparison differs from recomputed paired evidence")
        result={"format":"w4a4_recovery_v11_final_manifest","protocol_file":str(self.protocol_path),"protocol_sha256":self.protocol["sha256"],
                "selection_file":str(self.selection_path),"selection_sha256":self.selection["selection_sha256"],
                "qad_selection":qsd,"opd_selection":od,"selection_uses_heldout":False,"selected_pressure_recipe":self.selection["selected_recipe"],
                "selected_pressure_checkpoint":str(base),"selected_ptq_checkpoint":str(base),"heldout_round":str(round_dir),"heldout_comparison":identity(comparison),
                "selected_qad_learning_rate":win_lr,"selected_qad_checkpoint_identity":model_id(qad_ck),
                "selected_qad_model_identity":model_id(qad_model),"selected_continued_model_identity":model_id(rec["continued_qad"]),
                "selected_opd_weight":win_w,"selected_opd_model_identity":model_id(rec[win_arm]),
                "required_arms":["bf16","ptq","qad","continued_qad","qad_opd"]}
        fm=self.run_dir/"final_manifest.json"
        if fm.exists() and jread(fm)!=result: raise OrchestrationError("final_manifest differs")
        if not fm.exists(): jwrite(fm,result)
        self.state["status"]="complete"; self.save(); return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",required=True); p.add_argument("--protocol-file",default=str(ROOT/"exp/recovery_protocol_v3_high_fp4.json"))
    p.add_argument("--ptq-selection",required=True); p.add_argument("--base"); p.add_argument("--gr00t-repo"); p.add_argument("--python"); p.add_argument("--rollout-python"); p.add_argument("--dataset")
    p.add_argument("--port-base",type=int,default=5790); p.add_argument("--cleanup-duplicates",action="store_true"); p.add_argument("--adopt-complete",action="store_true")
    p.add_argument("--capture-dataset", help="Directory of replayable normalized capture samples for QAD")
    p.add_argument("--allow-opd-nonimprovement", action="store_true",
                   help="Run held-out comparisons even when development OPD does not strictly beat both controls")
    p.add_argument("--validate-only",action="store_true"); p.add_argument("--until",choices=("qad_dev","qad_selection","recovery_dev","opd_selection","all"),default="all")
    a=p.parse_args(argv)
    try: print(json.dumps(Driver(a).run(a.until),ensure_ascii=False,indent=2)); return 0
    except (OrchestrationError,FileNotFoundError,ValueError) as e: print(f"[high-fp4-v3] ERROR: {e}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
