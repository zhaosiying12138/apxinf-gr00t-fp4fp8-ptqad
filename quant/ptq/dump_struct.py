"""Dump GR00T N1.7 module structure (Linears + norms) for PTQ fold-map."""
import os, sys, json, time
BASE = os.environ.get("PTQ_BASE", "/home/zhaosiying/codebase/fp4vla/weights/GR00T-N1.7-LIBERO/libero_10")
OUT = "/mnt/c/fq_ptq_calib"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.getcwd())
T0 = time.time()
import torch
from gr00t.configs.base_config import get_default_config
from gr00t.model import MODEL_REGISTRY
config = get_default_config().load_dict({
    "data": {"download_cache": False, "datasets": [{
        "dataset_paths": ["./demo_data/libero_demo"], "mix_ratio": 1.0,
        "embodiment_tag": "libero_sim"}]},
})
config.load_config_path = None
config.model.model_path = BASE
config.model.action_horizon = 64
config.training.use_fsdp2 = False
from pathlib import Path
pipeline = MODEL_REGISTRY.get(type(config.model))(config, Path(OUT))
pipeline.setup()
model = pipeline.return_model().to("cuda").to(torch.bfloat16).eval()
rows = []
for name, mod in model.named_modules():
    cls = type(mod).__name__
    if isinstance(mod, torch.nn.Linear):
        rows.append({"name": name, "cls": cls, "shape": list(mod.weight.shape)})
    elif "Norm" in cls or "norm" in cls.lower():
        w = getattr(mod, "weight", None)
        rows.append({"name": name, "cls": cls,
                     "shape": list(w.shape) if w is not None else None})
json.dump(rows, open(f"{OUT}/struct.json", "w"), indent=1)
n_lin = sum(1 for r in rows if r["cls"] == "Linear")
print(f"[struct] {n_lin} Linears, {len(rows)-n_lin} norms, {time.time()-T0:.0f}s")
# quick console digest: layer families
from collections import Counter
fam = Counter()
for r in rows:
    if r["cls"] == "Linear":
        n = r["name"]
        key = n
        for pat in ["language_model.layers", "vision_tower", "vision_model.encoder.layers",
                    "action_head", "multi_modal_projector", "embed_out", "lm_head"]:
            if pat in n:
                key = pat
                break
        fam[key] += 1
for k, v in fam.most_common(30):
    print(f"  {v:4d}  {k}")
