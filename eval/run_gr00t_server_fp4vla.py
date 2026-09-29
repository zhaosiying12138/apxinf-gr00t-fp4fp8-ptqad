# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys

import os as _os
_FP4_SCOPE = _os.environ.get("FP4VLA_SCOPE", "all")
if _os.environ.get("FP4VLA_QUANT") == "1":
    import sys as _sys
    _sys.path.insert(0, "/home/zhaosiying/codebase/fp4vla/rl")
    from scoped_quant import install_scoped_fakequant, mark_scope
    install_scoped_fakequant(_FP4_SCOPE)

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


def _load_json_modality_configs(config_path: Path) -> dict[str, ModalityConfig]:
    """Load a JSON file whose values are ModalityConfig field dicts.

    A dataset's ``meta/modality.json`` is a different (data-layout) schema and is
    not accepted here — point such users at a .py config instead of letting the
    ``ModalityConfig(**v)`` unpack raise a bare ``TypeError``.
    """
    with open(config_path, "r") as f:
        raw = json.load(f)
    try:
        return {k: ModalityConfig(**v) for k, v in raw.items()}
    except TypeError as exc:
        raise ValueError(
            f"{config_path} is not a ModalityConfig JSON: each value must hold ModalityConfig "
            f"fields (delta_indices, modality_keys, ...). A dataset's meta/modality.json uses a "
            f"different schema; pass a .py modality config (e.g. examples/SO100/so100_config.py) instead."
        ) from exc


@dataclass
class ServerConfig:
    """Configuration for running the GR00T inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""


def main(config: ServerConfig):
    try:
        import huggingface_hub as _hh
        class _MI:
            tags = None
        _hh.model_info = lambda *a, **k: _MI()
    except Exception:
        pass
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # Create and start the server
    if config.model_path is not None:
        # check if the model path exists
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
        if os.environ.get("FP4VLA_QUANT") == "1":
            from scoped_quant import mark_scope
            mark_scope(policy.model)
        _log_dir = os.environ.get("FP4VLA_LOG_DIR", "")
        if _log_dir:
            # RWR rollout logging: dump (obs, action) per get_action call for
            # on-policy self-imitation training (videos JPEG-compressed).
            import pickle, gzip, numpy as _np
            from PIL import Image
            import io as _io
            os.makedirs(_log_dir, exist_ok=True)
            _inner_get_action = policy.get_action
            _step = [0]

            def _logged_get_action(obs, options=None):
                act, info = _inner_get_action(obs, options)
                try:
                    ent = {"obs": {}, "action": None}
                    for k, v in (obs or {}).items():
                        if isinstance(v, dict):
                            ent["obs"][k] = {
                                kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
                                for kk, vv in v.items()}
                        elif hasattr(v, "shape") and v.ndim >= 3 and v.dtype == _np.uint8:
                            buf = _io.BytesIO()
                            Image.fromarray(v.squeeze(0) if v.ndim == 4 else v).save(buf, "JPEG", quality=90)
                            ent["obs"][k] = {"__jpeg__": buf.getvalue()}
                        elif hasattr(v, "tolist"):
                            ent["obs"][k] = v.tolist()
                        else:
                            ent["obs"][k] = v
                    if isinstance(act, dict):
                        ent["action"] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                                         for k, v in act.items()}
                    with open(f"{_log_dir}/step_{_step[0]:06d}.pkl.gz", "wb") as f:
                        f.write(gzip.compress(pickle.dumps(ent)))
                    _step[0] += 1
                except Exception as e:
                    print(f"[fp4vla-log] drop step {_step[0]}: {type(e).__name__} {e}", flush=True)
                return act, info
            policy.get_action = _logged_get_action
            print(f"[fp4vla-log] rollout logging -> {_log_dir}", flush=True)
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                # The .py file is expected to call register_modality_config()
                # as an import side-effect; resolution falls through to
                # MODALITY_CONFIGS below.
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                modality_configs = _load_json_modality_configs(config_path)
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        # For .py configs (or no config path), look up from the registry
        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    with PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    ) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
