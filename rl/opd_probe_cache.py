"""Pre-cache teacher predictions on a fixed probe batch (offline, safe).

Inference-only: same config route as launch_finetune (get_default_config
.load_dict), model+dataset via MODEL_REGISTRY pipeline, BF16 weights, NO
fake-quant, eval forever — the exact inference path proven stable in all our
closed-loop runs. Saves (probe_inputs, teacher pred_actions) to
/mnt/c/fq_opd_probes/teacher_probes.pt for the OPD training anchor.
"""
import os, sys, time

BASE = "/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10"
OUT = "/mnt/c/fq_opd_probes"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
sys.path.insert(0, os.getcwd())

import torch
from gr00t.configs.base_config import get_default_config
from gr00t.model import MODEL_REGISTRY

config = get_default_config().load_dict({
    "data": {"download_cache": False, "datasets": [{
        "dataset_paths": ["./demo_data/libero_demo"],
        "mix_ratio": 1.0,
        "embodiment_tag": "libero_sim"}]},
})
config.load_config_path = None
config.model.model_path = BASE
config.model.action_horizon = 64
config.training.use_fsdp2 = False
config.training.use_wandl = False

from pathlib import Path
pipeline = MODEL_REGISTRY.get(type(config.model))(config, Path(OUT))
pipeline.setup()
model = pipeline.return_model().to("cuda").to(torch.bfloat16).eval()
train_dataset, _ = pipeline.return_dataset()
collator = pipeline.return_collator()

it = iter(train_dataset)
raw = [next(it) for _ in range(8)]
batch = collator(raw)
tin = batch["inputs"] if "inputs" in batch else batch
tin = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in tin.items()}

torch.manual_seed(20260927)  # pins the head's (noise, t) sampling
with torch.no_grad():
    out = model(tin)
pred = out["pred_actions"].float().cpu()
probe_inputs = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in tin.items()}
torch.save({"inputs": probe_inputs, "pred": pred}, f"{OUT}/teacher_probes.pt")
print(f"[probe-cache] batch={tuple(pred.shape)} in {time.time()-T0:.0f}s "
      f"mean={float(pred.mean()):.4f} std={float(pred.std()):.4f}", flush=True)
