"""CPU-only contract fixtures for ``run_high_fp4_v3.py``.

These tests validate protocol/selection provenance and the no-overwrite resume
guard. They deliberately create tiny fake checkpoints and never import torch,
launch a simulator, or start a GPU worker.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import shutil
import tempfile
import unittest

from exp import run_high_fp4_v3 as driver


class HighFp4V3CpuFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.protocol = self.root / "exp/recovery_protocol_v3_high_fp4.json"
        self.tmp = Path(tempfile.mkdtemp(prefix="fp4vla-v3-cpu-"))
        self.base = self.tmp / "checkpoints/bf16"
        self.ptq = self.tmp / "checkpoints/ptq"
        for path in (self.base, self.ptq):
            path.mkdir(parents=True)
            (path / "model-00001.safetensors").write_bytes(b"fixture-weight")
            (path / "config.json").write_text("{}\n")
            (path / "statistics.json").write_text("{}\n")
        (self.ptq / "ptq_recipe.json").write_text("{}\n")
        protocol_sha = driver.sha(self.protocol)
        self.selection = self.tmp / "development/selection.json"
        self.selection.parent.mkdir()
        arms = {}
        sources = {}
        for name, successes, checkpoint in (
            ("bf16", 46, self.base),
            ("head_lang_vision", 25, self.ptq),
            ("calib", 15, self.ptq),
        ):
            arms[name] = {
                "successes": successes,
                "episodes": 50,
                "macro_success_rate": successes / 50,
                "checkpoint": str(checkpoint),
                "environment_pairing_verified": True,
            }
            folder=self.selection.parent/name
            self.write_evaluation(folder,checkpoint,successes)
            for filename in ("eval_manifest.json","task_results.json","summary.json"):
                sources[f"{name}/{filename}"]=driver.sha(folder/filename)
        self.selection.write_text(json.dumps({
            "protocol_sha256": protocol_sha,
            "selection_uses_heldout": False,
            "selected_recipe": "calib",
            "pressure_candidates": ["head_lang_vision", "calib"],
            "arms": arms,
            "source_sha256": sources,
        }))

    def write_evaluation(self, folder, checkpoint, successes, purpose="development"):
        folder.mkdir(parents=True)
        protocol=driver.load_protocol(self.protocol)
        part=protocol["partitions"][purpose]
        episodes=part["episodes_per_task"]
        manifest={
            "out":str(folder),"video_root":str(folder/"videos"),"checkpoint":str(checkpoint),
            "protocol_file":str(self.protocol),"protocol_sha256":protocol["sha256"],
            "purpose":purpose,"seed":part["seed"],"episodes":episodes,
            "init_state_indices":part["init_state_indices"],"tasks":list(driver.TASKS),
            "task_count":10,"n_envs":1,"task_seed_stride":1000,"episode_seed_stride":1,
            "server_seed_offset":10000000,"settle_steps":10,"n_action_steps":8,
            "max_episode_steps":720,"initial_state_protocol":"libero10_official_bank_v1",
        }
        driver.jwrite(folder/"eval_manifest.json",manifest)
        rows={}
        remaining=successes
        for ti,task in enumerate(driver.TASKS):
            task_successes=min(remaining,episodes)
            remaining-=task_successes
            outcomes=[i<task_successes for i in range(episodes)]
            resets=[{"episode_index":i,"seed":part["seed"]+1000*ti+i,
                     "init_state_index":part["init_state_indices"][i],"settle_steps":10,
                     "initial_state_sha256":"a"*64,"restored_state_sha256":"b"*64,
                     "init_state_bank_sha256":"c"*64} for i in range(episodes)]
            log=folder/(task+".log")
            log.write_text("".join("FP4VLA_EPISODE_RESET "+json.dumps(row)+"\n" for row in resets)+
                           f"results: ('{task}', {outcomes})\n")
            rows[task]={**driver.parse_log(log),"returncode":0,"seed":part["seed"]+1000*ti}
        driver.jwrite(folder/"task_results.json",rows)
        driver.jwrite(folder/"summary.json",{
            "tasks_complete":10,"total_successes":successes,"total_episodes":episodes*10,
            "macro_success_rate":successes/(episodes*10),"purpose":purpose,"seed":part["seed"],
        })
        return folder

    def make_driver(self, adopt=False):
        return driver.Driver(argparse.Namespace(
            protocol_file=str(self.protocol),ptq_selection=str(self.selection),
            run_dir=str(self.tmp/"run"),base=str(self.base),gr00t_repo=None,python=None,
            rollout_python=None,dataset=None,port_base=5790,cleanup_duplicates=False,
            adopt_complete=adopt,validate_only=True))

    def rewrite_raw_row(self, folder, task, row):
        log=folder/(task+".log")
        log.write_text("".join("FP4VLA_EPISODE_RESET "+json.dumps(r)+"\n" for r in row["resets"])+
                       f"results: ('{task}', {row['results']})\n")
        row["log_sha256"]=driver.sha(log)
        rows=driver.jread(folder/"task_results.json")
        rows[task]=row
        driver.jwrite(folder/"task_results.json",rows)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_validate_only_records_protocol_and_selection(self):
        run_dir = self.tmp / "run"
        result = driver.main([
            "--run-dir", str(run_dir), "--protocol-file", str(self.protocol),
            "--ptq-selection", str(self.selection), "--base", str(self.base),
            "--validate-only",
        ])
        self.assertEqual(result, 0)
        state = json.loads((run_dir / "run_manifest.json").read_text())
        self.assertEqual(state["protocol_sha256"], driver.sha(self.protocol))
        self.assertFalse(state["selection_uses_heldout"])

    def test_selection_change_is_rejected_on_resume(self):
        run_dir = self.tmp / "run"
        args = ["--run-dir", str(run_dir), "--protocol-file", str(self.protocol),
                "--ptq-selection", str(self.selection), "--base", str(self.base),
                "--validate-only"]
        self.assertEqual(driver.main(args), 0)
        changed = json.loads(self.selection.read_text())
        changed["selected_recipe"] = "head_lang_vision"
        self.selection.write_text(json.dumps(changed))
        self.assertEqual(driver.main(args), 2)

    def test_rejects_numeric_boolean_outcome(self):
        folder=self.selection.parent/"bf16"
        rows=driver.jread(folder/"task_results.json")
        rows[driver.TASKS[0]]["results"][0]=1
        driver.jwrite(folder/"task_results.json",rows)
        with self.assertRaises(driver.OrchestrationError):
            driver.eval_audit(folder,driver.load_protocol(self.protocol),"development")

    def test_rejects_raw_log_changed(self):
        folder=self.selection.parent/"bf16"
        with (folder/(driver.TASKS[0]+".log")).open("a") as stream:
            stream.write("changed\n")
        with self.assertRaisesRegex(driver.OrchestrationError,"raw log SHA"):
            driver.eval_audit(folder,driver.load_protocol(self.protocol),"development")

    def test_rejects_task_accounting_changed(self):
        folder=self.selection.parent/"bf16"
        rows=driver.jread(folder/"task_results.json")
        rows[driver.TASKS[0]]["episodes"]=4
        driver.jwrite(folder/"task_results.json",rows)
        with self.assertRaisesRegex(ValueError,"raw-result episodes"):
            driver.eval_audit(folder,driver.load_protocol(self.protocol),"development")

    def test_rejects_raw_result_disagreement_even_with_updated_hash(self):
        folder=self.selection.parent/"bf16"
        rows=driver.jread(folder/"task_results.json")
        task=driver.TASKS[0]
        log=folder/(task+".log")
        log.write_text(log.read_text().replace("[True, True, True, True, True]",
                                               "[False, True, True, True, True]"))
        rows[task]["log_sha256"]=driver.sha(log)
        driver.jwrite(folder/"task_results.json",rows)
        with self.assertRaisesRegex(driver.OrchestrationError,"raw log differs"):
            driver.eval_audit(folder,driver.load_protocol(self.protocol),"development")

    def test_rejects_missing_reset_hash(self):
        folder=self.selection.parent/"bf16"
        task=driver.TASKS[0]
        row=driver.jread(folder/"task_results.json")[task]
        row["resets"][0].pop("restored_state_sha256")
        self.rewrite_raw_row(folder,task,row)
        with self.assertRaisesRegex(ValueError,"restored_state_sha256"):
            driver.eval_audit(folder,driver.load_protocol(self.protocol),"development")

    def test_rejects_summary_string_number(self):
        folder=self.selection.parent/"bf16"
        summary=driver.jread(folder/"summary.json")
        summary["total_successes"]="46"
        driver.jwrite(folder/"summary.json",summary)
        with self.assertRaisesRegex(driver.OrchestrationError,"summary accounting"):
            driver.eval_audit(folder,driver.load_protocol(self.protocol),"development")

    def test_rejects_moved_evaluation_manifest(self):
        folder=self.selection.parent/"bf16"
        moved=folder.with_name("moved")
        folder.rename(moved)
        with self.assertRaisesRegex(driver.OrchestrationError,"output path moved"):
            driver.eval_audit(moved,driver.load_protocol(self.protocol),"development")

    def make_recovery_devs(self, drv, kind):
        candidates=drv.protocol["selection"]["qad_learning_rates" if kind=="qad" else "opd_weights"]
        rows={}
        for value in candidates:
            model=drv.art/("merge_qad_lr_"+driver.sn(str(float(value))) if kind=="qad" else
                            "merge_opd_025" if value==.25 else "merge_opd_100")
            model.mkdir()
            folder=drv.art/("dev_"+kind+"_"+str(value))
            self.write_evaluation(folder,model,44)
            rows[value]=folder
        return rows

    def test_qad_tie_chooses_lower_lr_and_tampered_winner_rejected(self):
        drv=self.make_driver()
        rows=self.make_recovery_devs(drv,"qad")
        out=drv.art/"select_qad_lr"
        drv.select(out,"qad",rows)
        drv.select_verify(out,"qad")
        data=driver.jread(out/"selection.json")
        self.assertEqual(data["selected_learning_rate"],.00005)
        data["selected_learning_rate"]=.0001
        driver.jwrite(out/"selection.json",data)
        with self.assertRaisesRegex(driver.OrchestrationError,"recomputed"):
            drv.select_verify(out,"qad")

    def test_opd_tie_chooses_lower_weight(self):
        drv=self.make_driver()
        rows=self.make_recovery_devs(drv,"opd")
        report=drv.selection_report("opd",rows)
        self.assertEqual(report["selected_opd_weight"],.25)

    def test_selection_rejects_pairing_mismatch(self):
        drv=self.make_driver()
        rows=self.make_recovery_devs(drv,"qad")
        folder=rows[.0001]
        task=driver.TASKS[0]
        row=driver.jread(folder/"task_results.json")[task]
        row["resets"][0]["restored_state_sha256"]="d"*64
        self.rewrite_raw_row(folder,task,row)
        with self.assertRaisesRegex(driver.OrchestrationError,"different environment"):
            drv.selection_report("qad",rows)

    def test_stage_uses_stable_paths_and_skips_complete_action(self):
        drv=self.make_driver()
        actions=[]
        def action(path):
            actions.append(path)
            path.mkdir()
            driver.jwrite(path/"manifest.json",{"out":str(path)})
            return {}
        def verify(path):
            self.assertEqual(driver.jread(path/"manifest.json")["out"],str(path))
            return {"identity":driver.identity(path/"manifest.json")}
        final=drv.stage("fixture",action,verify)
        self.assertEqual(final,drv.art/"fixture")
        self.assertEqual(actions,[final])
        drv.stage("fixture",action,verify)
        self.assertEqual(actions,[final])
        self.assertFalse((drv.work/"fixture").exists())

    def test_stage_rejects_partial_and_changed_complete(self):
        drv=self.make_driver()
        path=drv.art/"fixture"
        path.mkdir()
        with self.assertRaisesRegex(driver.OrchestrationError,"incomplete stage"):
            drv.stage("fixture",lambda p:{},lambda p:{})
        drv.adopt=True
        driver.jwrite(path/"manifest.json",{"out":str(path)})
        verify=lambda p:{"identity":driver.identity(p/"manifest.json")}
        drv.stage("fixture",lambda p:self.fail("must not rerun"),verify)
        driver.jwrite(path/"manifest.json",{"out":"changed"})
        with self.assertRaisesRegex(driver.OrchestrationError,"evidence changed"):
            drv.stage("fixture",lambda p:{},verify)

    def test_cache_short_count_rejected_before_payload_load(self):
        drv=self.make_driver()
        folder=drv.art/"teacher_cache"
        folder.mkdir()
        (folder/"teacher_probes.pt").write_bytes(b"fixture")
        driver.jwrite(folder/"teacher_probes.json",{
            "source_kind":"student_rollout","teacher":str(drv.base),"seed":drv.seed,
            "requested_count":160,"count":159,
        })
        with self.assertRaisesRegex(driver.OrchestrationError,"provenance invalid"):
            drv.cache_verify(folder,drv.art/"collection_qad",drv.art/"winning_model")

    def test_training_request_paths_and_continuation_identity_are_checked(self):
        drv=self.make_driver()
        initial=drv.art/"initial/checkpoint-500"
        initial.mkdir(parents=True)
        (initial/"model.safetensors").write_bytes(b"qad-adapter-fixture")
        cache=drv.art/"teacher_probes.pt"
        cache.write_bytes(b"cache-fixture")
        out=drv.art/"train_opd_025"
        def fake_trainer(name,command,cwd,env):
            self.assertEqual(env["QAD_INIT_ADAPTER"],str(initial))
            self.assertEqual(env["OPD_CACHE_PATH"],str(cache))
            self.assertEqual(env["QAD_LR"],"5e-05")
            self.assertEqual(env["QAD_OPD_MSE_W"],"0.25")
            checkpoint=out/"checkpoint-100"
            checkpoint.mkdir()
            (checkpoint/"model.safetensors").write_bytes(b"trained-adapter-fixture")
            rec={
                "base":str(self.ptq),
                "rank":drv.rank,"alpha":drv.alpha,"scope":drv.scope,
                "train_seed":drv.seed,"protocol_sha256":drv.protocol["sha256"],
                "optimizer_resumed":False,"probe_weight":.25,"probe_every":drv.every,
                "micro_batch":1,"effective_global_batch":drv.batch,"gradient_accumulation_steps":drv.batch,
                "initial_adapter":str(initial),"probe_cache":str(cache),"probe_cache_sha256":driver.sha(cache),
                "base_config_sha256":driver.sha(self.ptq/"config.json"),
                "base_statistics_sha256":driver.sha(self.ptq/"statistics.json"),
                "base_recipe_sha256":driver.sha(self.ptq/"ptq_recipe.json"),
            }
            driver.jwrite(out/"recovery_manifest.json",rec)
            driver.jwrite(checkpoint/"recovery_manifest.json",rec)
            driver.jwrite(out/"runtime_metrics.json",{
                "status":"completed","global_steps":100,"requested_optimizer_steps":100,
            })
        drv.run_logged=fake_trainer
        drv.train(out,"train_opd_025",self.ptq,.00005,100,initial,.25,cache)
        audited=drv.train_verify(out,100,initial,.00005,.25,cache)
        self.assertEqual(audited["checkpoint_identity"]["path"],str(out/"checkpoint-100"))
        (initial/"model.safetensors").write_bytes(b"changed-adapter")
        with self.assertRaisesRegex(driver.OrchestrationError,"invocation/path/source"):
            drv.train_verify(out,100,initial,.00005,.25,cache)

    def test_merge_rejects_stale_adapter_source_path(self):
        drv=self.make_driver()
        out=drv.art/"merge_fixture"
        out.mkdir()
        (out/"ptq_recipe.json").write_text("{}")
        driver.jwrite(out/"merge_manifest.json",{
            "status":"complete","training_checkpoint":str(drv.work/"train_qad/checkpoint-500"),
        })
        with self.assertRaisesRegex(driver.OrchestrationError,"merge training checkpoint"):
            drv.merge_verify(out)


if __name__ == "__main__":
    unittest.main()
