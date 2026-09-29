"""Count candidate GR00T LoRA parameters from safetensors headers, CPU only."""
import argparse
import json
from pathlib import Path
import struct

from lora_scope import SCOPES, in_scope


def inventory(base, rank=32):
    layers = {}
    for path in sorted(Path(base).glob("*.safetensors")):
        with path.open("rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            for name, entry in json.loads(file.read(length)).items():
                if name == "__metadata__" or not name.endswith(".weight"):
                    continue
                shape = entry["shape"]
                if len(shape) != 2 or shape[1] % 16 or name.endswith(".position_embedding.weight"):
                    continue
                layers[name[:-7]] = rank * sum(shape)
    result = {}
    for scope in SCOPES:
        selected = {name: count for name, count in layers.items() if in_scope(name, scope)}
        params = sum(selected.values())
        result[scope] = {"linear_modules": len(selected), "trainable_parameters": params,
                         "parameter_bytes_fp32": 4 * params,
                         "parameter_gradient_adam_moments_bytes_fp32": 16 * params,
                         "head_modules": sum(n.startswith("action_head.") for n in selected),
                         "language_modules": sum(".language_model.layers." in n for n in selected)}
    return {"base": str(Path(base).resolve()), "rank": rank, "scopes": result,
            "note": "Header inventory excludes position embeddings; runtime nn.Linear injection count is authoritative. Activations/workspaces are not included."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--rank", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(inventory(args.base, args.rank), indent=2))
