"""Independent CPU check of the installed HF Trainer's probe loss scaling.

This uses the real Trainer.training_step and Accelerator.backward on a tiny
model. It does not instantiate GR00T or prove full-model numerical agreement.
Run with the GR00T training environment's Python from any directory.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import copy
import datetime
import hashlib
import importlib.metadata
import inspect
import platform
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))
from probe_distill import CACHE_VERSION, architecture, flow_signature, install_sequential_probe, replay_context


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.eye(4), requires_grad=False)
        self.lora_A = nn.Parameter(torch.arange(8.).reshape(2, 4) / 20)
        self.lora_B = nn.Parameter(torch.arange(8.).reshape(4, 2) / 30)
        # Only the structure validator is stubbed; numerical Trainer and
        # Accelerator operations below are the installed implementation.
        self.config = SimpleNamespace()
        self.backbone = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=range(16))))
        self.action_head = SimpleNamespace(model=SimpleNamespace(transformer_blocks=range(32)),
                                          vl_self_attention=SimpleNamespace(transformer_blocks=range(4)))

    def forward(self, inputs):
        x = inputs["action"] + torch.randn_like(inputs["action"]) * .05
        prediction = x @ self.base.T + (x @ self.lora_A.T) @ self.lora_B.T
        return {"loss": prediction.square().mean(), "pred_actions": prediction}


def source_record(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    sources = {"checker": source_record(__file__),
               "probe_distill": source_record(ROOT / "rl/probe_distill.py"),
               "installed_trainer": source_record(inspect.getsourcefile(Trainer)),
               "installed_accelerator": source_record(inspect.getsourcefile(Accelerator))}
    torch.set_num_threads(2)
    results = []
    for accumulation in (1, 2, 16):
        with tempfile.TemporaryDirectory() as directory:
            model = TinyStudent()
            reference = copy.deepcopy(model)
            action = torch.arange(12.).reshape(1, 3, 4) / 8
            mask = torch.zeros_like(action)
            mask[:, :2, :2] = 1
            target = torch.full_like(action, 10000.)
            target[:, :2, :2] = -.2
            sample = {"inputs": {"action": action, "action_mask": mask}, "seed": 71, "pred": target}
            cache = Path(directory) / "cache.pt"
            torch.save({"version": CACHE_VERSION, "metadata": {
                "architecture": architecture(model), "model_dtype": "float32",
                "flow_config": flow_signature(model), "autocast_dtype": "none",
                "action_mask": {"horizon": 2, "dimensions": 2}}, "samples": [sample]}, cache)

            class ActualTrainer(Trainer):
                pass

            install_sequential_probe(ActualTrainer, cache, weight=.3, every=4)
            trainer = ActualTrainer(model=model, args=TrainingArguments(
                output_dir=directory, use_cpu=True, report_to=[], gradient_accumulation_steps=accumulation))
            trainer.state.global_step = 3
            trainer.current_gradient_accumulation_steps = accumulation
            torch.manual_seed(900)
            for micro in range(accumulation):
                trainer.training_step(model, {"inputs": {"action": action + micro / 10}})
            torch.manual_seed(900)
            for micro in range(accumulation):
                (reference({"action": action + micro / 10})["loss"] / accumulation).backward()
                with replay_context(reference, 71, "none"):
                    pred = reference(sample["inputs"])["pred_actions"]
                    # Independent reference slices only valid elements. Huge
                    # padded targets must contribute neither loss nor gradient.
                    loss = (pred[:, :2, :2] - target[:, :2, :2]).square().mean()
                    (.3 * loss / accumulation).backward()
            differences = {name: float((parameter.grad - dict(reference.named_parameters())[name].grad).abs().max())
                           for name, parameter in model.named_parameters() if parameter.requires_grad}
            if not all(torch.isfinite(torch.tensor(value)) and value < 1e-6 for value in differences.values()):
                raise RuntimeError(f"Gradient reference mismatch: {differences}")
            if model.base.grad is not None:
                raise RuntimeError("Frozen base unexpectedly received a gradient")
            results.append({"accumulation": accumulation,
                            "accelerator_accumulation": trainer.accelerator.gradient_accumulation_steps,
                            "parameter_gradient_max_abs_error": differences,
                            "frozen_base_has_no_gradient": True})
    report = {"status": "passed", "cache_version": CACHE_VERSION, "device": "cpu",
              "cuda_initialized": torch.cuda.is_initialized(), "cases": results,
              "scope": "Real installed Trainer/Accelerator, tiny LoRA model, masked probe; not a full GR00T forward."}
    if report["cuda_initialized"]:
        raise RuntimeError("CPU checker unexpectedly initialized CUDA")
    for value in sources.values():
        if source_record(value["path"]) != value:
            raise RuntimeError("Checker/producer/installed dependency changed during validation")
    report.update(started_utc=started,
                  completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  checker_and_source_identities=sources,
                  environment={"python": platform.python_version(), "executable": sys.executable,
                               "packages": {name: importlib.metadata.version(name)
                                            for name in ("torch", "transformers", "accelerate")},
                               "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                               "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                               "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"]},
                  synthetic_input_definition="Exactly the tiny tensors, RNG seeds, mask and target specified in the hashed checker; no private dataset/cache input.",
                  reproduction="CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 <GR00T-training-python> -B -O paper/validation/verify_trainer_scaling.py")
    output = Path(__file__).with_suffix(".json")
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
