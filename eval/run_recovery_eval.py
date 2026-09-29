"""Serial LIBERO-10 collection/held-out evaluation with explicit seed manifests.

Run with the standard-library Python; model and simulation use their own venvs.
One server/task and one environment guarantee independent per-episode reset
seeds. Do not run concurrently with GPU training, baking or another evaluation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import subprocess
import time

TASKS = [
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
]


INIT_INDICES = {"development": [0, 1], "collection": [2, 3],
                "heldout": list(range(10, 20)), "smoke": [0]}


def protocol_entry(path, purpose):
    """Read a versioned bank partition from a protocol file.

    The original protocol is kept as the default for backwards compatibility.
    New studies pass an explicit file so that a new partition cannot be
    silently evaluated under the old hard-coded indices.  Both the compact
    ``{"partitions": {purpose: {...}}}`` form and the original top-level
    purpose keys are accepted; all other fields remain provenance in the
    manifest and are not interpreted here.
    """
    protocol = json.loads(Path(path).read_text())
    partitions = protocol.get("partitions", protocol)
    entry = partitions.get(purpose)
    if not isinstance(entry, dict) or "init_state_indices" not in entry:
        raise ValueError(f"Protocol {path} has no {purpose}.init_state_indices partition")
    indices = entry["init_state_indices"]
    if (not isinstance(indices, list) or not indices or
            any(type(x) is not int for x in indices) or
            len(set(indices)) != len(indices) or
            any(x < 0 or x >= 50 for x in indices)):
        raise ValueError(f"Protocol {path} has invalid {purpose}.init_state_indices")
    return entry


def protocol_partition(path, purpose):
    """Return the validated initial-state indices for ``purpose``."""
    return protocol_entry(path, purpose)["init_state_indices"]


def validate_resets(result, seed, indices):
    """Fail closed: result-only logs cannot certify paired evaluation."""
    resets = result["resets"]
    if len(resets) < len(indices):
        raise ValueError("Missing episode initial-state records")
    for episode, bank_index in enumerate(indices):
        record = resets[episode]
        if (record.get("episode_index") != episode or record.get("seed") != seed + episode or
            record.get("init_state_index") != bank_index or record.get("settle_steps") != 10):
            raise ValueError(f"Episode {episode} reset differs from declared bank/seed protocol")
        for key in ("initial_state_sha256", "restored_state_sha256", "init_state_bank_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", record.get(key, "")):
                raise ValueError(f"Episode {episode} missing valid {key}")
    return resets[:len(indices)]


def parse_log(path):
    text = Path(path).read_text(errors="replace")
    records = re.findall(r"results:\s*\('[^']+',\s*\[([^\]]*)\]", text)
    values = [x == "True" for x in re.findall(r"\b(?:True|False)\b", records[-1])] if records else []
    resets = [json.loads(x) for x in re.findall(r"FP4VLA_EPISODE_RESET (\{[^\n]+\})", text)]
    return {"results": values, "successes": sum(values), "episodes": len(values),
            "success_rate": sum(values) / len(values) if values else None,
            "resets": resets, "log_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--purpose", required=True, choices=("development", "collection", "heldout", "smoke"))
    ap.add_argument("--protocol-file", help="Versioned protocol JSON supplying the bank partition; defaults to exp/recovery_protocol.json")
    ap.add_argument("--collection-manifest", help="Required for heldout: prove disjoint reset seeds")
    ap.add_argument("--episodes", type=int, help="Default 2 development/collection, 10 heldout, 1 smoke")
    ap.add_argument("--init-state-indices", help="Explicit bank indices; formal purposes must match fixed partition")
    ap.add_argument("--port", type=int, default=5595)
    ap.add_argument("--gr00t", default=os.environ.get("GR00T_REPO", str(Path.home() / "codebase/groot-fsdp2/Isaac-GR00T")))
    ap.add_argument("--server-python", default=os.environ.get("PTQAD_PYTHON"))
    ap.add_argument("--rollout-python", default=os.environ.get("LIBERO_PYTHON"))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--task-count", type=int, default=10, help="Only reduce for smoke, never a ten-task score")
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[1]
    protocol_path = Path(args.protocol_file).resolve() if args.protocol_file else project / "exp/recovery_protocol.json"
    if not protocol_path.is_file():
        ap.error(f"Protocol file does not exist: {protocol_path}")
    protocol_data = protocol_entry(protocol_path, args.purpose) if args.protocol_file else None
    protocol_indices = protocol_data["init_state_indices"] if protocol_data else INIT_INDICES[args.purpose]
    indices = [int(x) for x in args.init_state_indices.split(",")] if args.init_state_indices else protocol_indices
    if args.protocol_file and indices != protocol_indices:
        ap.error("Explicit bank indices must exactly match the selected versioned protocol partition")
    if protocol_data and protocol_data.get("seed") is not None and args.seed != int(protocol_data["seed"]):
        ap.error(f"Seed {args.seed} does not match the selected protocol partition ({protocol_data['seed']})")
    if (protocol_data and args.episodes is not None and
            protocol_data.get("episodes_per_task") is not None and
            args.episodes != int(protocol_data["episodes_per_task"])):
        ap.error("episodes does not match the selected versioned protocol partition")
    if not args.protocol_file and args.purpose != "smoke" and indices != INIT_INDICES[args.purpose]:
        ap.error("Formal evaluation must use the fixed official-bank partition")
    declared_episodes = protocol_data.get("episodes_per_task") if protocol_data else None
    args.episodes = args.episodes if args.episodes is not None else (declared_episodes or len(indices))
    if args.episodes != len(indices) or len(set(indices)) != len(indices):
        ap.error("episodes must equal the number of unique official bank indices")
    if not 1 <= args.task_count <= 10 or not 1 <= args.episodes < 1000:
        ap.error("task-count must be 1..10 and episodes 1..999")
    if args.task_count != 10 and args.purpose != "smoke":
        ap.error("Development, collection and heldout protocols require all 10 tasks")
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError(f"New evidence requires a new output directory: {output}")
    if args.purpose == "heldout":
        if not args.collection_manifest:
            ap.error("heldout requires --collection-manifest")
        collection = json.loads(Path(args.collection_manifest).read_text())
        if collection["purpose"] != "collection":
            raise ValueError("Expected a collection manifest")
        if collection.get("initial_state_protocol") != "libero10_official_bank_v1":
            raise ValueError("Collection manifest does not certify official bank initialization")
        if set(collection["init_state_indices"]) & set(indices):
            raise ValueError("Collection and heldout official bank indices overlap")
        for i in range(10):
            train = set(range(collection["seed"] + 1000*i, collection["seed"] + 1000*i + collection["episodes"] + 1))
            test = set(range(args.seed + 1000*i, args.seed + 1000*i + args.episodes + 1))
            if train & test:
                raise ValueError("Collection and heldout reset seeds overlap")
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))  # refuse to attach to another server
    output.mkdir(parents=True)
    gr00t = Path(args.gr00t).resolve()
    python = args.server_python or str(gr00t / ".venv/bin/python")
    rollout_python = args.rollout_python or str(gr00t / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python")
    media = os.environ.get("PTQAD_MEDIA_LIB")
    # PTQAD_MEDIA_LIB points to <ffmpeg environment>/lib, so its CLI is a
    # sibling bin directory. PTQAD_MEDIA_BIN explicitly overrides that default.
    media_bin = os.environ.get("PTQAD_MEDIA_BIN") or (str(Path(media).parent / "bin") if media else None)
    video_path = (media_bin + os.pathsep if media_bin else "") + os.environ.get("PATH", "")
    ffmpeg = shutil.which("ffmpeg", path=video_path)
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg executable missing: set PTQAD_MEDIA_LIB=<environment>/lib or PTQAD_MEDIA_BIN=<environment>/bin")
    manifest = vars(args) | {"checkpoint": str(Path(args.checkpoint).resolve()),
                              "gr00t": str(gr00t), "n_envs": 1,
                              "task_seed_stride": 1000, "episode_seed_stride": 1,
                              "server_seed_offset": 10000000, "tasks": TASKS[:args.task_count],
                              "n_action_steps": 8, "max_episode_steps": 720}
    manifest.update({"initial_state_protocol": "libero10_official_bank_v1",
                     "init_state_indices": indices, "settle_steps": 10,
                     "ffmpeg_executable": ffmpeg, "media_library_dir": media,
                     "video_root": str(output / "videos"),
                     "paired_scope": "environment initial states; policy noise is seeded per task"})
    manifest["protocol_file"] = str(protocol_path)
    manifest["protocol_sha256"] = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    (output / "eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    results = {}
    started = time.time()
    for index, task in enumerate(TASKS[:args.task_count]):
        seed = args.seed + 1000 * index
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "FP4VLA_QUANT": "0", "PYTHONPATH": str(gr00t),
                    "GR00T_EVAL_SEED": str(seed + 10000000),
                    "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
        media = env.get("PTQAD_MEDIA_LIB")
        if media:
            env["LD_LIBRARY_PATH"] = media + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env["PATH"] = video_path
        env.pop("OPD_CAPTURE_DIR", None)
        if args.purpose == "collection":
            env["OPD_CAPTURE_DIR"] = str(output / "observations" / task)
        server_cmd = [python, str(project / "eval/serve_recovery.py"), "--model-path", manifest["checkpoint"],
                      "--embodiment-tag", "LIBERO_PANDA", "--use-sim-policy-wrapper", "--port", str(args.port)]
        with open(output / f"{task}.server.log", "w") as log:
            server = subprocess.Popen(server_cmd, cwd=gr00t, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                ready = False
                for _ in range(180):
                    if server.poll() is not None:
                        raise RuntimeError(f"Server exited; see {log.name}")
                    try:
                        with socket.create_connection(("127.0.0.1", args.port), timeout=1):
                            ready = True
                            break
                    except OSError:
                        time.sleep(1)
                if not ready:
                    raise TimeoutError("Server did not become ready")
                command = [rollout_python, str(project / "eval/rollout_seeded.py"),
                           "--env-name", f"libero_sim/{task}", "--n-episodes", str(args.episodes),
                           "--n-envs", "1", "--seed", str(seed), "--max-episode-steps", "720",
                           "--init-state-indices", ",".join(map(str, indices)),
                           "--video-dir", str(output / "videos" / task),
                           "--n-action-steps", "8", "--policy-client-host", "127.0.0.1",
                           "--policy-client-port", str(args.port)]
                with open(output / f"{task}.log", "w") as rollout_log:
                    completed = subprocess.run(command, cwd=gr00t, env=env, stdout=rollout_log,
                                               stderr=subprocess.STDOUT, timeout=args.timeout)
                result = parse_log(output / f"{task}.log")
                result.update({"returncode": completed.returncode, "seed": seed})
                results[task] = result
                (output / "task_results.json").write_text(json.dumps(results, indent=2) + "\n")
                if completed.returncode or result["episodes"] != args.episodes:
                    raise RuntimeError(f"Incomplete task {task}: rc={completed.returncode}, episodes={result['episodes']}")
                validate_resets(result, seed, indices)
                print(f"[eval] {task}: {result['successes']}/{result['episodes']}", flush=True)
            finally:
                stop(server)
    summary = {"tasks_complete": len(results), "total_successes": sum(x["successes"] for x in results.values()),
               "total_episodes": sum(x["episodes"] for x in results.values()),
               "macro_success_rate": sum(x["success_rate"] for x in results.values()) / 10 if len(results) == 10 else None,
               "purpose": args.purpose, "seed": args.seed,
               "wall_seconds_including_server_loads": time.time() - started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
