"""Label independently collated observations with the complete frozen teacher.

Run from GR00T. --input-dir consumes capture_onpolicy.py outputs; without it
this is a demo/offline probe, not on-policy data. FP32 parameters and BF16
autocast default to the training model's actual compute/noise convention.
"""
import argparse
import json
from itertools import zip_longest
import os
from pathlib import Path
import sys
import time

import torch
from probe_distill import CACHE_VERSION, file_sha256, flow_signature, prediction, replay_context, require_full_model, tensor_tree
from gr00t_runtime import configure_libero_data, verify_libero_statistics, libero_action_spec, action_mask
from runtime_metrics import cuda_memory_peaks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default=os.environ.get("OPD_TEACHER", str(Path(__file__).resolve().parents[1] / "weights/GR00T-N1.7-LIBERO/libero_10")))
    ap.add_argument("--out", default=os.environ.get("OPD_CACHE_PATH", str(Path(os.environ.get("OPD_CACHE_OUT", "/mnt/c/fq_opd_probes_v2")) / "teacher_probes.pt")))
    ap.add_argument("--input-dir", help="Captured student observations, one sample_*.pt per observation")
    ap.add_argument("--dataset", default=os.environ.get("QAD_DATASET", "./demo_data/libero_demo"))
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--model-dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--autocast-dtype", choices=("bfloat16", "none"), default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.count < 1:
        ap.error("--count must be positive")
    outpath = Path(args.out)
    if outpath.exists():
        raise FileExistsError(f"Refusing to overwrite teacher evidence: {outpath}")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, os.getcwd())
    from gr00t.configs.base_config import get_default_config
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model import MODEL_REGISTRY
    from transformers import AutoModel

    start = time.time()
    teacher = Path(args.teacher).resolve()
    teacher_weights = {p.name: {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                       for p in sorted(teacher.glob("*.safetensors"))}
    if not teacher_weights:
        raise ValueError("Teacher checkpoint has no safetensors weights")
    source_files = []
    # model_path is NOT a pipeline load argument; start_from_checkpoint is.
    config = get_default_config().load_dict({"data": {
        "download_cache": False, "override_pretraining_statistics": False,
        "datasets": [{"dataset_paths": [args.dataset], "mix_ratio": 1.0,
                      "embodiment_tag": "libero_sim"}]}})
    config.model = Gr00tN1d7Config.from_pretrained(teacher, local_files_only=True)
    config.model.model_name = os.environ.get("GR00T_BACKBONE_MODEL", config.model.model_name)
    config.model.use_relative_action = True
    config.training.start_from_checkpoint = str(teacher)
    config.training.use_fsdp2 = False
    config.training.use_wandb = False
    config.training.transformers_local_files_only = True
    config.load_config_path = None
    configure_libero_data(config, teacher)
    action_spec = libero_action_spec(teacher)

    if args.input_dir:
        model, info = AutoModel.from_pretrained(
            teacher, config=config.model, local_files_only=True,
            transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
            output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Teacher weight loading was not exact: {info}")
        groups = {}
        for path in sorted(Path(args.input_dir).rglob("sample_*.pt")):
            groups.setdefault(str(path.parent), []).append(path)
        # Round-robin task folders before truncation so a small smoke cache is
        # not accidentally drawn entirely from the alphabetically first task.
        paths = [p for row in zip_longest(*groups.values()) for p in row if p is not None][:args.count]
        if not paths:
            raise ValueError("No captured observations found")
        source_files = [{"path": str(p.resolve()), "bytes": p.stat().st_size,
                         "sha256": file_sha256(p)} for p in paths]
        raw_samples = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
        if any(s.get("source_kind") != "student_rollout" for s in raw_samples):
            raise ValueError("Input is not a captured student rollout")
        stats_hash = file_sha256(teacher / "statistics.json")
        if any(s["student_statistics_sha256"] != stats_hash for s in raw_samples):
            raise ValueError("Student/teacher normalization differs; cannot reuse normalized rollout inputs")
        source = "student_rollout"
    else:
        pipeline = MODEL_REGISTRY.get(type(config.model))(config, outpath.parent)
        pipeline.setup()
        verify_libero_statistics(pipeline.processor, teacher)
        model = pipeline.return_model()
        dataset, _ = pipeline.return_dataset()
        collator = pipeline.return_collator()
        iterator = iter(dataset)
        raw_samples = [{"inputs": tensor_tree(collator([next(iterator)])["inputs"]),
                        "source_kind": "demo", "sample_index": i}
                       for i in range(args.count)]
        source = "demo"

    arch = require_full_model(model)
    model = model.to(device=args.device, dtype=getattr(torch, args.model_dtype)).eval()
    model.requires_grad_(False)
    samples = []
    for index, raw in enumerate(raw_samples):
        seed = args.seed + index
        inputs = tensor_tree(raw["inputs"])
        valid = action_mask(inputs["action"], action_spec)
        if "action_mask" in inputs:
            valid *= inputs["action_mask"].float()
        if valid.sum() <= 0:
            raise ValueError("Probe has no valid action elements")
        inputs["action_mask"] = valid
        if inputs["action"].shape[0] != 1:
            raise ValueError("Each input must be independently collated as a microbatch of one")
        with replay_context(model, seed, args.autocast_dtype), torch.no_grad():
            pred = prediction(model(tensor_tree(inputs, args.device))).float().cpu()
        if not torch.isfinite(pred).all():
            raise ValueError(f"Nonfinite teacher velocity at probe {index}; refusing cache")
        samples.append({"inputs": inputs, "pred": pred, "seed": seed,
                        "provenance": {k: v for k, v in raw.items() if k != "inputs"}})
        print(f"[probe-cache] {index + 1}/{len(raw_samples)} pred={tuple(pred.shape)}", flush=True)
    metadata = {"teacher": str(teacher), "teacher_config_sha256": file_sha256(teacher / "config.json"),
                "teacher_statistics_sha256": file_sha256(teacher / "statistics.json"),
                "teacher_weights": teacher_weights, "source_observation_files": source_files,
                "labeling_implementation_sha256": file_sha256(__file__),
                "replay_implementation_sha256": file_sha256(Path(__file__).with_name("probe_distill.py")),
                "architecture": arch, "model_dtype": args.model_dtype,
                "flow_config": flow_signature(model),
                "autocast_dtype": args.autocast_dtype, "eval_mode": True,
                "source_kind": source, "seed": args.seed, "count": len(samples),
                "requested_count": args.count,
                "objective": "masked velocity MSE at shared full interpolated action/noise/time",
                "action_mask": action_spec,
                "cuda_peak_memory": cuda_memory_peaks(),
                "timing_scope": "checkpoint loading and teacher labeling, before cache serialization",
                "cuda_peak_scope": "PyTorch allocator peaks for this labeling process",
                "elapsed_seconds": time.time() - start}
    torch.save({"version": CACHE_VERSION, "metadata": metadata, "samples": samples}, outpath)
    outpath.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[probe-cache] saved {outpath} source={source} architecture={arch}", flush=True)


if __name__ == "__main__":
    main()
