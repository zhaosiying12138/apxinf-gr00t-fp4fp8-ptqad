"""Explicit single-GPU batch contract, independent of upstream CLI naming."""


def resolve_batch(env):
    if int(env.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This recovery runner currently supports one GPU/process")
    micro = int(env.get("QAD_MICRO_BATCH", env.get("QAD_BSZ", "1")))
    explicit_acc = env.get("QAD_ACCUM_STEPS", env.get("QAD_ACC"))
    if "QAD_GLOBAL_BATCH" in env:
        effective = int(env["QAD_GLOBAL_BATCH"])
        if micro <= 0 or effective <= 0 or effective % micro:
            raise ValueError("QAD_GLOBAL_BATCH must be a positive multiple of QAD_MICRO_BATCH")
        accumulation = effective // micro
        if explicit_acc is not None and int(explicit_acc) != accumulation:
            raise ValueError("QAD_ACCUM_STEPS conflicts with GLOBAL/MICRO")
    else:
        accumulation = int(explicit_acc or "16")
        effective = micro * accumulation
    if micro < 1 or accumulation < 1:
        raise ValueError("Microbatch and accumulation must both be positive")
    return micro, accumulation, effective
