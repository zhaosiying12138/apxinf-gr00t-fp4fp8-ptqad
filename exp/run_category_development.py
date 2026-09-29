#!/usr/bin/env python3
"""Run the isolated v4 category-FP4 development comparison.

The driver creates a new BF16 evaluation and two category-aware descendants.
Each descendant is made from an already completed pure-PTQ parent; the parent
chain is checked before category calibration or baking.  No v3 result is
copied into the v4 evidence directory and no held-out result is read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "exp"))
from run_high_fp4_v3 import TASKS, eval_audit

CANDIDATES = ("head_lang_vision_category", "calib_category")
PARENT_RECIPES = {"head_lang_vision_category": "head_lang_vision", "calib_category": "calib"}


class CategoryDevelopmentError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise CategoryDevelopmentError(message)


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CategoryDevelopmentError(f"invalid JSON {path}: {exc}") from exc


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def write_new(path: Path, value) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_protocol(path: Path) -> dict:
    path = path.resolve(strict=True)
    data = read_json(path)
    require(data.get("version") == 4, "category development requires protocol version 4")
    parts = data.get("partitions")
    require(isinstance(parts, dict), "v4 protocol has no partitions")
    partitions = {}
    expected = {"development": 5, "collection": 4, "heldout": 10}
    for name, episodes in expected.items():
        row = parts.get(name)
        require(isinstance(row, dict), f"protocol has no {name} partition")
        indices = row.get("init_state_indices")
        require(type(row.get("seed")) is int and row.get("episodes_per_task") == episodes,
                f"invalid {name} seed/episodes")
        require(isinstance(indices, list) and len(indices) == episodes and
                len(set(indices)) == episodes and all(type(x) is int and 0 <= x < 50 for x in indices),
                f"invalid {name} initial-state indices")
        partitions[name] = {"seed": row["seed"], "episodes_per_task": episodes,
                            "init_state_indices": indices}
    selection = data.get("selection", {})
    require(selection.get("pressure_candidates") == list(CANDIDATES),
            "v4 pressure candidates differ from the declared category ladder")
    pressure = selection.get("pressure_rule", {})
    require(pressure.get("choose") == "highest_fp4" and
            float(pressure.get("min_drop_from_bf16", -1)) == 0.20 and
            float(pressure.get("min_absolute_success", -1)) == 0.30,
            "v4 pressure rule differs from the preregistered rule")
    category = data.get("category_quantization", {})
    require(category.get("parent_recipes") == ["head_lang_vision", "calib"],
            "v4 parent recipe list differs")
    require(category.get("active_category") == 2 and category.get("categories") == 32,
            "v4 category bank contract differs")
    allowlist = category.get("allowlist")
    require(isinstance(allowlist, list) and len(allowlist) == 7 and len(set(allowlist)) == 7,
            "v4 category allowlist is incomplete")
    return {"data": data, "partitions": partitions, "selection": selection,
            "category": category, "path": str(path), "sha256": sha(path)}


def checkpoint_shards(path: Path) -> dict[str, dict]:
    path = path.resolve(strict=True)
    index = read_json(path / "model.safetensors.index.json")
    names = sorted(set(index.get("weight_map", {}).values()))
    require(names and all(Path(name).name == name for name in names),
            f"invalid checkpoint index: {path}")
    return {name: identity(path / name) for name in names}


def validate_category_parent(parent: Path, base: Path) -> dict:
    """Prove the category bake starts from a pure PTQ parent of this BF16 root."""
    sys.path.insert(0, str(ROOT / "quant" / "ptq"))
    from category_fp4 import validate_parent

    source, entries, recipe = validate_parent(parent)
    require(Path(source["root_bf16"]["path"]).resolve() == base.resolve(),
            f"category parent root is not the requested BF16 base: {parent}")
    require(recipe.get("recipe") in ("head_lang_vision", "calib"),
            f"unsupported category parent recipe: {parent}")
    return {"source": source, "entries": entries, "recipe": recipe,
            "checkpoint_shards": checkpoint_shards(parent),
            "ptq_recipe": identity(parent / "ptq_recipe.json"),
            "bake_manifest": identity(parent / "bake_manifest.json")}


def validate_category_output(output: Path, parent: Path, base: Path, candidate: str) -> dict:
    """Verify category_ptq_recipe/category_bake_manifest and all parent links."""
    output = output.resolve(strict=True)
    parent_info = validate_category_parent(parent, base)
    manifest = read_json(output / "category_bake_manifest.json")
    recipe = read_json(output / "category_ptq_recipe.json")
    require(manifest.get("status") == "complete" and
            manifest.get("version") == "category-k-axis-nvfp4-v1",
            f"{candidate}: incomplete category bake")
    require(recipe.get("recipe") == "category_nvfp4_extension" and
            recipe.get("version") == "category-k-axis-nvfp4-v1",
            f"{candidate}: invalid category recipe")
    manifest_parent = manifest.get("parent", {}).get("parent", {})
    manifest_root = manifest.get("root_bf16", {})
    recipe_root = recipe.get("source_base", {})
    require(Path(manifest_parent.get("path", "")).resolve() == parent,
            f"{candidate}: manifest parent path differs")
    require(Path(recipe.get("parent", "")).resolve() == parent,
            f"{candidate}: recipe parent path differs")
    require(Path(manifest_root.get("path", "")).resolve() == base,
            f"{candidate}: category manifest root differs")
    require(Path(recipe_root.get("path", "")).resolve() == base,
            f"{candidate}: category recipe root differs")
    require(manifest["parent_recipe"]["sha256"] == parent_info["ptq_recipe"]["sha256"] and
            manifest["parent_bake_manifest"]["sha256"] == parent_info["bake_manifest"]["sha256"] and
            recipe["parent_recipe_sha256"] == parent_info["ptq_recipe"]["sha256"] and
            recipe["parent_bake_manifest_sha256"] == parent_info["bake_manifest"]["sha256"],
            f"{candidate}: parent manifest hashes differ")
    category_source_keys = set(parent_info["source"]["category_source_sha256"])
    require(set(recipe.get("source_category_sha256", {})) == category_source_keys and
            set(manifest.get("source_category_sha256", {})) == category_source_keys and
            set(manifest.get("output_category_sha256", {})) == category_source_keys,
            f"{candidate}: category key set differs")
    shards = checkpoint_shards(output)
    require(set(shards) == set(parent_info["checkpoint_shards"]),
            f"{candidate}: output shard set differs from parent")
    category_shards = {
        parent_info["entries"][key]["shard"]
        for key in parent_info["source"]["category_source_sha256"]
    }
    require(manifest.get("output_weights") and
            set(manifest["output_weights"]) == category_shards,
            f"{candidate}: output shard identities are missing")
    for name, record in manifest["output_weights"].items():
        require(name in shards, f"{candidate}: manifest names unknown output shard {name}")
        require(record.get("bytes") == shards[name]["bytes"] and
                record.get("sha256") == shards[name]["sha256"],
                f"{candidate}: output shard changed after manifest: {name}")
    return {"candidate": candidate, "parent": str(parent), "parent_identity": parent_info,
            "output_identity": {"shards": shards,
                                "category_recipe": identity(output / "category_ptq_recipe.json"),
                                "category_bake_manifest": identity(output / "category_bake_manifest.json")},
            "memory": recipe.get("memory"),
            "category_recipe_sha256": sha(output / "category_ptq_recipe.json")}


def run_logged(name: str, command: list[str], cwd: Path, log: Path, env: dict[str, str]) -> None:
    require(not log.exists(), f"refusing existing stage log: {log}")
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    for key in list(merged):
        if key.startswith(("QAD_", "OPD_")) or key in ("GR00T_BASE_CKPT", "TRAIN_SEED",
                                                        "PROTOCOL_FILE", "PTQAD_PROTOCOL_FILE"):
            merged.pop(key)
    merged.update(env)
    merged.setdefault("HF_HUB_OFFLINE", "1")
    merged.setdefault("TRANSFORMERS_OFFLINE", "1")
    with log.open("x") as stream:
        stream.write(json.dumps({"command": command, "cwd": str(cwd)}) + "\n")
        completed = subprocess.run(command, cwd=str(cwd), env=merged,
                                   stdout=stream, stderr=subprocess.STDOUT)
        stream.write(f"RETURN_CODE {completed.returncode}\n")
    require(completed.returncode == 0, f"{name} failed; inspect {log}")


def score(audit: dict, label: str) -> Fraction:
    require(type(audit.get("successes")) is int and type(audit.get("episodes")) is int and
            audit["episodes"] > 0, f"{label}: invalid score")
    return Fraction(audit["successes"], audit["episodes"])


def choose_pressure(audits: dict, candidates: tuple[str, ...], pressure: dict):
    """Select only from completed development scores, in declared FP4 order."""
    require("bf16" in audits and all(candidate in audits for candidate in candidates),
            "all v4 development arms must be complete before selection")
    baseline = score(audits["bf16"], "bf16")
    drop = Fraction(str(pressure["min_drop_from_bf16"]))
    floor = Fraction(str(pressure["min_absolute_success"]))
    qualifying = [candidate for candidate in candidates
                  if baseline - score(audits[candidate], candidate) >= drop and
                  score(audits[candidate], candidate) >= floor]
    return (qualifying[-1] if qualifying else None), qualifying


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--protocol-file", default=os.environ.get("CATEGORY_PROTOCOL_FILE"))
    parser.add_argument("--base", default=os.environ.get("PTQAD_BASE"))
    parser.add_argument("--parent-head-lang-vision", required=True)
    parser.add_argument("--parent-calib", required=True)
    parser.add_argument("--gr00t-repo", default=os.environ.get("GR00T_REPO"))
    # Resolve these from the GR00T checkout by default.  In particular, do
    # not inherit a shell's PTQAD_PYTHON/LIBERO_PYTHON value: a host Python
    # can exist while lacking GR00T's numpy/torch runtime, yielding a partial
    # evaluation directory before the failure is noticed.
    parser.add_argument("--python")
    parser.add_argument("--rollout-python")
    parser.add_argument("--dataset", default=os.environ.get("QAD_DATASET"))
    parser.add_argument("--media-lib", default=os.environ.get("PTQAD_MEDIA_LIB",
                                                                 "/home/zhaosiying/miniforge3/envs/media7/lib"))
    parser.add_argument("--port", type=int, default=5630)
    parser.add_argument("--calibration-windows", type=int, default=128)
    parser.add_argument("--calibration-batch", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require(args.protocol_file, "--protocol-file is required")
    protocol_path = Path(args.protocol_file).expanduser().resolve()
    protocol = load_protocol(protocol_path)
    base = Path(args.base or ROOT / "weights/GR00T-N1.7-LIBERO/libero_10").resolve(strict=True)
    run = Path(args.run_dir).expanduser().resolve()
    require(not run.exists(), f"new v4 run directory required: {run}")
    parents = {
        "head_lang_vision_category": Path(args.parent_head_lang_vision).resolve(strict=True),
        "calib_category": Path(args.parent_calib).resolve(strict=True),
    }
    parent_info = {}
    for candidate, parent in parents.items():
        expected = PARENT_RECIPES[candidate]
        info = validate_category_parent(parent, base)
        require(info["recipe"].get("recipe") == expected,
                f"{candidate}: expected {expected} parent, got {info['recipe'].get('recipe')}")
        parent_info[candidate] = info
    groot = Path(args.gr00t_repo or Path.home() / "codebase/groot-fsdp2/Isaac-GR00T").resolve(strict=True)
    python = Path(args.python or groot / ".venv/bin/python").resolve(strict=True)
    rollout = Path(args.rollout_python or groot / "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python").resolve(strict=True)
    dataset = Path(args.dataset or groot / "demo_data/libero_demo").resolve(strict=True)
    media_lib = Path(args.media_lib).expanduser().resolve(strict=True)
    require((media_lib.parent / "bin" / "ffmpeg").is_file(),
            f"ffmpeg is missing beside media library: {media_lib}; pass --media-lib explicitly")
    media_env = {"PTQAD_MEDIA_LIB": str(media_lib)}
    # Make the media runtime part of every child evaluation environment.  This
    # avoids relying on a shell profile when the driver is launched from WSL.
    os.environ["PTQAD_MEDIA_LIB"] = str(media_lib)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "run_dir": str(run), "protocol_sha256": protocol["sha256"],
                          "order": ["bf16", *CANDIDATES], "parent_recipes": PARENT_RECIPES}, indent=2))
        return

    run.mkdir(parents=True)
    development = run / "development"
    logs = run / "logs"
    write_new(run / "run_manifest.json", {
        "version": 1, "kind": "category_fp4_development_v4", "status": "running",
        "protocol_file": str(protocol_path), "protocol_sha256": protocol["sha256"],
        "base": str(base), "base_identity": {"config": sha(base / "config.json"),
        "statistics": sha(base / "statistics.json")}, "parent_recipes": {
            candidate: {"path": str(parents[candidate]), "recipe": PARENT_RECIPES[candidate],
                        "ptq_recipe_sha256": parent_info[candidate]["ptq_recipe"]["sha256"],
                        "bake_manifest_sha256": parent_info[candidate]["bake_manifest"]["sha256"]}
            for candidate in CANDIDATES},
        "selection_uses_heldout": False, "device": args.device,
        "calibration_windows": args.calibration_windows, "calibration_batch": args.calibration_batch,
        "python": str(python), "rollout_python": str(rollout), "media_lib": str(media_lib),
    })

    # BF16 is always evaluated into this new v4 directory first.
    part = protocol["partitions"]["development"]
    bf16_eval = development / "bf16"
    run_logged("eval-bf16", [str(python), str(ROOT / "eval/run_recovery_eval.py"),
        "--checkpoint", str(base), "--out", str(bf16_eval), "--purpose", "development",
        "--seed", str(part["seed"]), "--episodes", str(part["episodes_per_task"]),
        "--gr00t", str(groot), "--server-python", str(python), "--rollout-python", str(rollout),
        "--port", str(args.port), "--protocol-file", str(protocol_path)], groot,
        logs / "eval-bf16.log", {"PROTOCOL_FILE": str(protocol_path), **media_env})
    audits = {"bf16": eval_audit(bf16_eval, protocol, "development", base)}
    pairing = audits["bf16"]["pairing_sha256"]

    for offset, candidate in enumerate(CANDIDATES, start=1):
        parent = parents[candidate]
        cache = run / "category_calibration" / candidate
        output = run / candidate
        run_logged(f"category-calibrate-{candidate}", [str(python), str(ROOT / "quant/ptq/collector_category.py"),
            "--parent", str(parent), "--out", str(cache), "--dataset", str(dataset),
            "--windows", str(args.calibration_windows), "--batch", str(args.calibration_batch),
            "--seed", str(protocol["selection"]["train_seed"]), "--device", args.device], groot,
            logs / f"category-calibrate-{candidate}.log", {"PROTOCOL_FILE": str(protocol_path)})
        run_logged(f"category-bake-{candidate}", [str(python), str(ROOT / "quant/ptq/bake_category.py"),
            "--parent", str(parent), "--calib", str(cache), "--out", str(output),
            "--expected-windows", str(args.calibration_windows), "--gptq-damp", "0.01"], ROOT,
            logs / f"category-bake-{candidate}.log", {"PROTOCOL_FILE": str(protocol_path)})
        category_identity = validate_category_output(output, parent, base, candidate)
        evaluation = development / candidate
        run_logged(f"eval-{candidate}", [str(python), str(ROOT / "eval/run_recovery_eval.py"),
            "--checkpoint", str(output), "--out", str(evaluation), "--purpose", "development",
            "--seed", str(part["seed"]), "--episodes", str(part["episodes_per_task"]),
            "--gr00t", str(groot), "--server-python", str(python), "--rollout-python", str(rollout),
            "--port", str(args.port + offset), "--protocol-file", str(protocol_path)], groot,
            logs / f"eval-{candidate}.log", {"PROTOCOL_FILE": str(protocol_path), **media_env})
        audits[candidate] = eval_audit(evaluation, protocol, "development", output)
        require(audits[candidate]["pairing_sha256"] == pairing,
                f"{candidate}: initial-state pairing differs from BF16")
        write_new(run / f"{candidate}.identity.json", category_identity)

    selected, qualifying = choose_pressure(audits, CANDIDATES,
                                           protocol["selection"]["pressure_rule"])
    result = {"status": "complete", "protocol_file": str(protocol_path),
              "protocol_sha256": protocol["sha256"], "selection_uses_heldout": False,
              "selection_rule": "highest-FP4 category candidate with >=0.20 drop and >=0.30 absolute; no fallback",
              "selected_recipe": selected, "qualifying_candidates": qualifying,
              "pairing_sha256": pairing, "arms": {
                  name: {"checkpoint": str(base if name == "bf16" else run / name),
                         "successes": audits[name]["successes"], "episodes": audits[name]["episodes"],
                         "macro_success_rate": audits[name]["macro_success_rate"],
                         "pairing_sha256": audits[name]["pairing_sha256"]}
                  for name in ("bf16", *CANDIDATES)},
              "parent_recipes": PARENT_RECIPES,
              "category_parent_identities": {name: parent_info[name]["ptq_recipe"] for name in CANDIDATES}}
    write_new(run / "selection.json", result)
    run_manifest_path = run / "run_manifest.json"
    run_manifest = read_json(run_manifest_path)
    run_manifest.update({"status": "complete", "selection": str(run / "selection.json"),
                         "selected_recipe": selected})
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
