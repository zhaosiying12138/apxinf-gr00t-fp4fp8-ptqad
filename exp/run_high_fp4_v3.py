#!/usr/bin/env python3
"""Serial, fail-closed orchestration for the v3 high-FP4 recovery study.

The input PTQ selection must be a completed development-only selection.json.
The driver then runs two QAD learning rates, selects on development, collects
the winning QAD student rollouts, labels one frozen teacher cache, runs
continued-QAD and two OPD weights from the exact same QAD A/B checkpoint,
selects the OPD weight on development, and finally evaluates exactly five
heldout arms. Stages publish atomically and never overwrite existing outputs.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, subprocess, sys, time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
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
    base=frac(arms["bf16"],"bf16")
    dr=Fraction(str(protocol["selection"]["pressure_rule"]["min_drop_from_bf16"]))
    floor=Fraction(str(protocol["selection"]["pressure_rule"]["min_absolute_success"]))
    qualifying=[a for a in c if base-frac(arms[a],a)>=dr and frac(arms[a],a)>=floor]
    if not qualifying or d.get("selected_recipe")!=qualifying[-1]:
        raise OrchestrationError("PTQ selection is not highest-FP4 qualifying candidate")
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
                                "recovery_manifest.json","merge_manifest.json","runtime_metrics.json",
                                "model.safetensors.index.json") if (p/n).is_file()}
    return {"path":str(p),"metadata":meta,"shards":shards}

def eval_audit(path: Path, protocol: dict[str,Any], purpose: str) -> dict[str,Any]:
    part=protocol["partitions"][purpose]; man=jread(path/"eval_manifest.json")
    rows=jread(path/"task_results.json"); summ=jread(path/"summary.json")
    for k,v in {"purpose":purpose,"seed":part["seed"],"episodes":part["episodes_per_task"],
                "init_state_indices":part["init_state_indices"],"tasks":list(TASKS),
                "task_count":10,"n_envs":1,"task_seed_stride":1000,
                "episode_seed_stride":1,"settle_steps":10,"n_action_steps":8,
                "max_episode_steps":720,"protocol_sha256":protocol["sha256"]}.items():
        if man.get(k)!=v: raise OrchestrationError(f"{purpose} manifest mismatch: {k}")
    if man.get("initial_state_protocol")!="libero10_official_bank_v1" or not man.get("protocol_file"):
        raise OrchestrationError(f"{purpose} manifest lacks official-bank protocol identity")
    protocol_file=Path(man["protocol_file"])
    if not protocol_file.is_file() or sha(protocol_file)!=protocol["sha256"]:
        raise OrchestrationError(f"{purpose} manifest protocol file/SHA is not the frozen protocol")
    if set(rows)!=set(TASKS) or summ.get("tasks_complete")!=10:
        raise OrchestrationError(f"{purpose} is incomplete")
    total_successes=0
    total_episodes=0
    for ti,t in enumerate(TASKS):
        row=rows[t]
        if row.get("returncode")!=0 or len(row.get("results",[]))!=part["episodes_per_task"]:
            raise OrchestrationError(f"{purpose}/{t} has incomplete outcomes")
        resets=row.get("resets",[])
        if len(resets)<part["episodes_per_task"]: raise OrchestrationError(f"{purpose}/{t} lacks resets")
        for ei,z in enumerate(resets[:part["episodes_per_task"]]):
            if (z.get("episode_index")!=ei or z.get("seed")!=part["seed"]+1000*ti+ei or
                z.get("init_state_index")!=part["init_state_indices"][ei]):
                raise OrchestrationError(f"{purpose}/{t}/{ei} reset mismatch")
        total_successes += sum(row["results"])
        total_episodes += len(row["results"])
    if (summ.get("total_successes")!=total_successes or summ.get("total_episodes")!=total_episodes or
        summ.get("tasks_complete")!=10 or not math.isclose(float(summ.get("macro_success_rate")),
                                                            total_successes/total_episodes,abs_tol=1e-12)):
        raise OrchestrationError(f"{purpose} summary accounting disagrees with task outcomes")
    return {"evaluation_identity":{n:identity(path/n) for n in ("eval_manifest.json","task_results.json","summary.json")},
            "successes":int(summ["total_successes"]),"episodes":int(summ["total_episodes"]),
            "macro_success_rate":float(summ["macro_success_rate"])}

class Driver:
    def __init__(self,a:argparse.Namespace):
        self.a=a; self.protocol_path=need(a.protocol_file,"protocol")
        self.protocol=load_protocol(self.protocol_path); self.selection_path=need(a.ptq_selection,"PTQ selection")
        self.selection=validate_ptq(self.selection_path,self.protocol)
        self.run_dir=Path(a.run_dir).expanduser().resolve()
        self.art=self.run_dir/"artifacts"; self.work=self.run_dir/"work"; self.logs=self.run_dir/"logs"
        self.stages=self.run_dir/"stages"; self.receipts=self.run_dir/"cleanup_receipts"
        self.groot=Path(a.gr00t_repo or os.environ.get("GR00T_REPO",str(Path.home()/ "codebase/groot-fsdp2/Isaac-GR00T"))).expanduser().resolve()
        self.py=Path(a.python or os.environ.get("PTQAD_PYTHON",str(self.groot/".venv/bin/python"))).expanduser().resolve()
        self.sim=Path(a.rollout_python or os.environ.get("LIBERO_PYTHON",str(self.groot/"gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"))).expanduser().resolve()
        self.base=Path(a.base or os.environ.get("PTQAD_BASE",str(ROOT/"weights/GR00T-N1.7-LIBERO/libero_10"))).expanduser().resolve()
        self.dataset=Path(a.dataset or os.environ.get("QAD_DATASET",str(self.groot/"demo_data/libero_demo"))).expanduser().resolve()
        s=self.protocol["selection"]; self.seed=int(s["train_seed"]); self.qsteps=int(s["qad_optimizer_steps"])
        self.csteps=int(s["continuation_optimizer_steps"]); self.rank=int(s["rank"]); self.alpha=float(s["alpha"])
        self.scope=str(s["recovery_scope"]); self.batch=int(s["effective_demo_batch"]); self.every=int(s["opd_every"])
        self.port=int(a.port_base); self.adopt=bool(a.adopt_complete); self.cleanup=bool(a.cleanup_duplicates)
        self.state=self.load_state()

    def save(self): jwrite(self.run_dir/"run_manifest.json",self.state)
    def load_state(self):
        man=self.run_dir/"run_manifest.json"
        if self.run_dir.exists() and any(self.run_dir.iterdir()) and not man.exists():
            raise OrchestrationError(f"run directory lacks run_manifest.json: {self.run_dir}")
        self.run_dir.mkdir(parents=True,exist_ok=True)
        if man.exists():
            state=jread(man)
            fixed={"protocol_sha256":self.protocol["sha256"],"selection_sha256":self.selection["selection_sha256"],
                   "base":str(self.base),"train_seed":self.seed,"qad_steps":self.qsteps,
                   "continuation_steps":self.csteps,"lora_scope":self.scope,"rank":self.rank,"alpha":self.alpha}
            for k,v in fixed.items():
                if state.get(k)!=v: raise OrchestrationError(f"resume identity changed: {k}")
            for p in (self.art,self.work,self.logs,self.stages,self.receipts):
                p.mkdir(parents=True,exist_ok=True)
            return state
        state={"format":"high_fp4_v3_orchestrator","status":"initialized",
               "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
               "protocol_file":str(self.protocol_path),"protocol_sha256":self.protocol["sha256"],
               "protocol_version":self.protocol["data"].get("version"),
               "selection_file":str(self.selection_path),"selection_sha256":self.selection["selection_sha256"],
               "selected_recipe":self.selection["selected_recipe"],
               "selected_ptq_checkpoint":self.selection["selected_ptq_checkpoint"],
               "selected_ptq_recipe_sha256":self.selection["selected_ptq_recipe_sha256"],
               "base":str(self.base),"base_identity":model_id(self.base),"gr00t_repo":str(self.groot),
               "python":str(self.py),"rollout_python":str(self.sim),"dataset":str(self.dataset),
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
        self.freeze_check(); final=self.art/sn(name); marker=self.stages/(sn(name)+".json"); work=self.work/sn(name)
        if marker.exists():
            if not final.exists() or jread(marker).get("status")!="complete": raise OrchestrationError(f"stage inconsistent: {name}")
            verify(final); return final
        if final.exists(): raise OrchestrationError(f"unmarked final output: {final}")
        if work.exists():
            if not self.adopt: raise OrchestrationError(f"incomplete stage {work}; inspect then --adopt-complete")
            audited=verify(work); work.rename(final); self.mark(name,{"output":str(final),"adopted":True,**audited}); return final
        # Leave the stage output path absent: run_recovery_eval.py deliberately
        # refuses an existing --out directory. Each action creates its own
        # output atomically or through its own temporary exporter.
        work.parent.mkdir(parents=True,exist_ok=True)
        produced=action(work); audited=verify(work); work.rename(final)
        self.mark(name,{"output":str(final),"adopted":False,**produced,**audited}); return final

    def run_logged(self,name,c, cwd, extra):
        log=self.logs/(sn(name)+".log")
        if log.exists(): raise OrchestrationError(f"refusing existing log: {log}")
        env=os.environ.copy(); env.update(extra); env.setdefault("HF_HUB_OFFLINE","1")
        env.setdefault("TRANSFORMERS_OFFLINE","1"); env.setdefault("NO_ALBUMENTATIONS_UPDATE","1")
        with log.open("x") as f:
            f.write(f"UTC {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}\nCWD {cwd}\nCOMMAND {cmdtext(c)}\n")
            r=subprocess.run(c,cwd=str(cwd),env=env,stdout=f,stderr=subprocess.STDOUT)
            f.write(f"RETURN_CODE {r.returncode}\n")
            if r.returncode==0: f.write(f"COMPLETED_UTC {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}\n")
        if r.returncode: raise OrchestrationError(f"{name} failed ({r.returncode}); inspect {log}")

    def train(self,work,name,base,lr,steps,initial=None,weight=0.,cache=None):
        e={"GR00T_BASE_CKPT":str(base),"QAD_DATASET":str(self.dataset),"QAD_OUT":str(work),
           "QAD_STEPS":str(steps),"QAD_SAVE_STEPS":str(steps),"QAD_GLOBAL_BATCH":str(self.batch),
           "QAD_MICRO_BATCH":"1","QAD_LORA_R":str(self.rank),"QAD_LORA_ALPHA":str(self.alpha),
           "QAD_LORA_SCOPE":self.scope,"QAD_LR":str(lr),"QAD_OPD_MSE_W":str(weight),
           "OPD_EVERY":str(self.every),"TRAIN_SEED":str(self.seed),"PROTOCOL_FILE":str(self.protocol_path),
           "QAD_ACTIVATION_CHECKPOINTING":"1"}
        if initial: e["QAD_INIT_ADAPTER"]=str(initial)
        if cache: e["OPD_CACHE_PATH"]=str(cache)
        self.run_logged(name,[str(self.py),str(ROOT/"rl/lora_qad.py")],self.groot,e)
        return {"base":str(base),"learning_rate":lr,"optimizer_steps":steps,"opd_weight":weight,
                "initial_adapter":str(initial) if initial else None,"teacher_cache":str(cache) if cache else None}

    def train_verify(self,p,steps,initial=None):
        m=jread(p/"runtime_metrics.json"); rec=jread(p/"recovery_manifest.json")
        if m.get("status")!="completed" or m.get("global_steps")!=steps or m.get("requested_optimizer_steps")!=steps:
            raise OrchestrationError(f"training does not prove {steps} steps: {p}")
        if initial and (Path(rec.get("initial_adapter","")).resolve()!=initial.resolve() or rec.get("optimizer_resumed") is not False):
            raise OrchestrationError("continuation is not exact-A/B with a fresh optimizer")
        checkpoint=p/f"checkpoint-{steps}"
        return {"checkpoint_identity":model_id(checkpoint),"recovery_manifest_sha256":sha(p/"recovery_manifest.json")}

    def merge(self,work,name,base,ckpt,steps):
        self.run_logged(name,[str(self.py),str(ROOT/"rl/lora_merge_bake.py"),"--base",str(base),"--ckpt",str(ckpt),
                             "--out",str(work),"--rank",str(self.rank),"--alpha",str(self.alpha)],ROOT,{})
        return {"base":str(base),"training_checkpoint":str(ckpt),"optimizer_steps":steps}
    def merge_verify(self,p):
        m=jread(p/"merge_manifest.json")
        if m.get("status")!="complete" or not (p/"ptq_recipe.json").is_file(): raise OrchestrationError(f"bad merged model: {p}")
        return {"model_identity":model_id(p),"merge_manifest_sha256":sha(p/"merge_manifest.json")}

    def evaluate(self,work,name,ckpt,purpose,offset=0,collection=None):
        part=self.protocol["partitions"][purpose]
        c=[str(self.py),str(ROOT/"eval/run_recovery_eval.py"),"--checkpoint",str(ckpt),"--out",str(work),
           "--purpose",purpose,"--seed",str(part["seed"]),"--episodes",str(part["episodes_per_task"]),
           "--gr00t",str(self.groot),"--server-python",str(self.py),"--rollout-python",str(self.sim),
           "--port",str(self.port+offset),"--protocol-file",str(self.protocol_path)]
        if collection: c += ["--collection-manifest",str(collection)]
        self.run_logged(name,c,self.groot,{"PROTOCOL_FILE":str(self.protocol_path),"QAD_DATASET":str(self.dataset)})
        return {"checkpoint":str(ckpt),"purpose":purpose}
    def eval_verify(self,p,purpose): return eval_audit(p,self.protocol,purpose)

    def cache(self,work,collection):
        count=10*self.protocol["partitions"]["collection"]["episodes_per_task"]*4
        self.run_logged("teacher-cache",[str(self.py),str(ROOT/"rl/opd_probe_cache.py"),"--teacher",str(self.base),
            "--input-dir",str(collection/"observations"),"--count",str(count),"--out",str(work/"teacher_probes.pt"),
            "--dataset",str(self.dataset),"--seed",str(self.seed)],self.groot,{})
        return {"collection_observations":str(collection/"observations"),"requested_count":count}
    def cache_verify(self,p):
        c,m=p/"teacher_probes.pt",p/"teacher_probes.json"
        if not c.is_file() or not m.is_file(): raise OrchestrationError("teacher cache incomplete")
        d=jread(m)
        if d.get("source_kind")!="student_rollout" or d.get("teacher")!=str(self.base) or int(d.get("count",0))<1:
            raise OrchestrationError("teacher cache provenance invalid")
        return {"cache_identity":identity(c),"metadata_identity":identity(m),"count":int(d["count"])}

    def cleanup_dupes(self,label,root,ckpt):
        if not self.cleanup: return
        try:
            running=subprocess.check_output(["ps","-eo","pid=,args="],text=True).splitlines()
        except (OSError,subprocess.CalledProcessError):
            running=[]
        active=[line.strip() for line in running if any(x in line for x in
                ("run_recovery_eval.py","serve_recovery.py","lora_qad.py","lora_merge_bake.py"))
                and "run_high_fp4_v3.py" not in line]
        if active:
            raise OrchestrationError(f"active training/evaluation process prevents cleanup: {active}")
        deleted=[]
        for p in sorted(root.glob("*.safetensors")):
            q=ckpt/p.name
            if p.is_symlink() or not p.is_file() or not q.is_file(): continue
            a,b=sha(p),sha(q)
            if p.stat().st_size!=q.stat().st_size or a!=b: continue
            size=p.stat().st_size; p.unlink(); deleted.append({"deleted":str(p),"retained":str(q),"bytes":size,"sha256":a})
        jwrite(self.receipts/(sn(label)+".json"),{"format":"verified_training_root_duplicate_cleanup_v1",
            "label":label,"training_root":str(root),"retained_checkpoint":str(ckpt),"deleted_files":deleted,
            "deleted_bytes":sum(x["bytes"] for x in deleted),"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "note":"Only exact root shard duplicates were deleted; evidence, checkpoints and merged models retained."})

    def select(self,work,kind,rows):
        values={}
        for key,p in rows.items():
            values[str(key)]={("learning_rate" if kind=="qad" else "opd_weight"):key,
                              **self.eval_verify(p,"development"),"evaluation_path":str(p),"selection_source":"development_only"}
        winner=sorted(values.values(),key=lambda x:(-x["macro_success_rate"],x["learning_rate"] if kind=="qad" else x["opd_weight"]))[0]
        d={"format":"high_fp4_v3_"+("qad_lr" if kind=="qad" else "opd")+"_"+("selection"),
           "protocol_file":str(self.protocol_path),"protocol_sha256":self.protocol["sha256"],
           "selection_uses_heldout":False,"criterion":"highest development macro success; ties choose lower "+("learning rate" if kind=="qad" else "OPD weight"),
           "candidates":values,"selected_"+("learning_rate" if kind=="qad" else "opd_weight"):winner["learning_rate" if kind=="qad" else "opd_weight"]}
        jwrite(work/"selection.json",d); return {"selection_identity":identity(work/"selection.json")}
    def select_verify(self,p,kind):
        d=jread(p/"selection.json")
        if d.get("protocol_sha256")!=self.protocol["sha256"] or d.get("selection_uses_heldout") is not False:
            raise OrchestrationError("selection is not development-only")
        if not isinstance(d.get("candidates"),dict) or len(d["candidates"])!=2: raise OrchestrationError("selection needs two candidates")
        for row in d["candidates"].values(): self.eval_verify(Path(row["evaluation_path"]),"development")
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
                          lambda p:self.train_verify(p,self.qsteps))
            ck=tr/f"checkpoint-{self.qsteps}"; trains[lr]=ck; self.cleanup_dupes("train_"+tag,tr,ck)
            models[lr]=self.stage("merge_"+tag,lambda w,tag=tag,ck=ck:self.merge(w,"merge_"+tag,base,ck,self.qsteps),self.merge_verify)
            devs[lr]=self.stage("dev_"+tag,lambda w,model=models[lr],tag=tag,i=i:self.evaluate(w,"dev_"+tag,model,"development",i),
                                  lambda p:self.eval_verify(p,"development"))
        if until=="qad_dev": return self.state
        qs=self.stage("select_qad_lr",lambda w:self.select(w,"qad",devs),
                      lambda p:self.select_verify(p,"qad"))
        qsd=jread(qs/"selection.json"); win_lr=float(qsd["selected_learning_rate"])
        qad_model,qad_ck=models[win_lr],trains[win_lr]
        if until=="qad_selection": return self.state
        coll=self.stage("collection_qad",lambda w:self.evaluate(w,"collection_qad",qad_model,"collection",2),
                        lambda p:self.eval_verify(p,"collection"))
        cache_stage=self.stage("teacher_cache",lambda w:self.cache(w,coll),self.cache_verify); cache=cache_stage/"teacher_probes.pt"
        rec={}
        for arm,weight in (("continued_qad",0.),("opd_025",.25),("opd_100",1.)):
            tr=self.stage("train_"+arm,lambda w,arm=arm,weight=weight:self.train(w,"train_"+arm,base,win_lr,self.csteps,qad_ck,weight,cache if weight else None),
                          lambda p:self.train_verify(p,self.csteps,qad_ck))
            ck=tr/f"checkpoint-{self.csteps}"; self.cleanup_dupes("train_"+arm,tr,ck)
            rec[arm]=self.stage("merge_"+arm,lambda w,arm=arm,ck=ck:self.merge(w,"merge_"+arm,base,ck,self.csteps),self.merge_verify)
        devrec={}
        for i,arm in enumerate(("continued_qad","opd_025","opd_100")):
            devrec[arm]=self.stage("dev_"+arm,lambda w,arm=arm,i=i:self.evaluate(w,"dev_"+arm,rec[arm],"development",3+i),
                                    lambda p:self.eval_verify(p,"development"))
        if until=="recovery_dev": return self.state
        osel=self.stage("select_opd_weight",lambda w:self.select(w,"opd",{.25:devrec["opd_025"],1.:devrec["opd_100"]}),
                         lambda p:self.select_verify(p,"opd"))
        od=jread(osel/"selection.json"); win_w=float(od["selected_opd_weight"]); win_arm="opd_025" if math.isclose(win_w,.25) else "opd_100"
        if until=="opd_selection": return self.state
        round_dir=self.art/"heldout_round"; round_dir.mkdir(parents=True,exist_ok=True); cm=coll/"eval_manifest.json"
        final=(("bf16",self.base,10),("ptq",base,11),("qad",qad_model,12),("continued_qad",rec["continued_qad"],13),("qad_opd",rec[win_arm],14))
        for arm,model,off in final:
            name="heldout_"+arm; out=round_dir/name
            if out.exists():
                audited=self.eval_verify(out,"heldout")
                marker=self.stages/(sn(name)+".json")
                if not marker.exists():
                    self.mark(name,{"output":str(out),"adopted":True,
                                    "checkpoint_identity":model_id(model),**audited})
                continue
            w=self.work/name
            if w.exists():
                if not self.adopt: raise OrchestrationError(f"incomplete heldout output: {w}")
                self.eval_verify(w,"heldout")
            else:
                self.evaluate(w,name,model,"heldout",off,cm); self.eval_verify(w,"heldout")
            w.rename(out); self.mark(name,{"output":str(out),"checkpoint_identity":model_id(model),"evaluation_identity":self.eval_verify(out,"heldout")})
        comparison=round_dir/"paired_comparison.json"
        if not comparison.exists():
            self.run_logged("compare_heldout",[str(self.py),str(ROOT/"eval/compare_recovery.py"),"--round",str(round_dir)],ROOT,{"PROTOCOL_FILE":str(self.protocol_path)})
        result={"format":"high_fp4_v3_final_manifest","protocol_file":str(self.protocol_path),"protocol_sha256":self.protocol["sha256"],
                "selection_file":str(self.selection_path),"selection_sha256":self.selection["selection_sha256"],
                "qad_selection":qsd,"opd_selection":od,"selection_uses_heldout":False,"selected_pressure_recipe":self.selection["selected_recipe"],
                "selected_pressure_checkpoint":str(base),"heldout_round":str(round_dir),"heldout_comparison":identity(comparison),
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
    p.add_argument("--validate-only",action="store_true"); p.add_argument("--until",choices=("qad_dev","qad_selection","recovery_dev","opd_selection","all"),default="all")
    a=p.parse_args(argv)
    try: print(json.dumps(Driver(a).run(a.until),ensure_ascii=False,indent=2)); return 0
    except (OrchestrationError,FileNotFoundError,ValueError) as e: print(f"[high-fp4-v3] ERROR: {e}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
