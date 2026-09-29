"""Official LIBERO-10 bank resets and ten raw zero-action settling steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def install_bank_resets(env_class, task_name, indices, settle_steps=10):
    import numpy as np
    import torch
    from libero.libero import benchmark
    from libero.libero.utils import get_libero_path
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_ids = [i for i in range(suite.get_num_tasks()) if suite.get_task(i).name == task_name]
    if len(task_ids) != 1 or len(set(indices)) != len(indices) or not indices:
        raise ValueError("Expected one LIBERO-10 task and unique declared bank indices")
    task_id = task_ids[0]
    task = suite.get_task(task_id)
    bank_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    # Official NumPy arrays in torch.save containers; keep weights_only loading.
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray,
                                            np.dtype, type(np.dtype("float64"))]):
        bank = suite.get_task_init_states(task_id)
    if min(indices) < 0 or max(indices) >= len(bank):
        raise ValueError(f"Bank has {len(bank)} states; requested {indices}")
    bank_hash = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    original = env_class.reset

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._recovery_seed = int(seed)
            self._recovery_episode = 0
        if not hasattr(self, "_recovery_seed"):
            raise ValueError("An explicit initial seed is required")
        episode = self._recovery_episode
        actual = self._recovery_seed + episode
        original(self, seed=actual, options=options)
        # Extra terminal auto-resets remain inside the declared partition and
        # are excluded from scored episode counts.
        bank_index = indices[episode % len(indices)]
        raw = self._env.set_init_state(bank[bank_index])
        restored = self._env.sim.get_state().flatten()
        restored_hash = hashlib.sha256(restored.tobytes()).hexdigest()
        # Direct simulator zeros retain gripper=0. LiberoEnv.step would apply
        # the policy's normalized-gripper transformation to this command.
        for _ in range(settle_steps):
            raw, _, _, _ = self._env.step(np.zeros(7, dtype=np.float64))
        observation = self._process_observation(raw)
        info = {"success": self._env.check_success()}
        state = self._env.sim.get_state().flatten()
        print("FP4VLA_EPISODE_RESET " + json.dumps({
            "episode_index": episode, "seed": actual, "task_id": task_id,
            "init_state_index": bank_index, "init_state_bank_size": len(bank),
            "init_state_bank_sha256": bank_hash,
            "restored_state_sha256": restored_hash, "settle_steps": settle_steps,
            "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest()}), flush=True)
        self._recovery_episode += 1
        return observation, info

    env_class.reset = reset
    return {"task_id": task_id, "bank_size": len(bank), "bank_sha256": bank_hash,
            "indices": indices, "settle_steps": settle_steps}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--init-state-indices", required=True)
    known, remaining = parser.parse_known_args()
    indices = [int(x) for x in known.init_state_indices.split(",")]
    sys.argv = [sys.argv[0], *remaining]
    if "--n-envs" not in sys.argv or sys.argv[sys.argv.index("--n-envs") + 1] != "1":
        raise ValueError("Audited bank protocol requires --n-envs 1")
    if int(sys.argv[sys.argv.index("--n-episodes") + 1]) != len(indices):
        raise ValueError("Declare exactly one official bank index per scored episode")
    name = sys.argv[sys.argv.index("--env-name") + 1].removeprefix("libero_sim/")
    sys.path.insert(0, os.getcwd())
    from gr00t.eval.sim.LIBERO.libero_env import LiberoEnv
    import gr00t.eval.rollout_policy as rollout
    install_bank_resets(LiberoEnv, name, indices)
    script = Path(rollout.__file__)
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
