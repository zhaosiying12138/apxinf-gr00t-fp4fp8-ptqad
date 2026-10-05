"""Read measured process peaks and parameter storage without creating CUDA state."""
from collections import defaultdict
import torch


def cuda_memory_peaks():
    if not torch.cuda.is_initialized():
        return []
    results = []
    for device in range(torch.cuda.device_count()):
        if not torch.cuda.memory_stats(device):
            continue
        results.append({"device_index": device, "device_name": torch.cuda.get_device_name(device),
                        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "memory_allocated_bytes_at_end": torch.cuda.memory_allocated(device),
                        "memory_reserved_bytes_at_end": torch.cuda.memory_reserved(device)})
    return results


def parameter_storage(model):
    if model is None:
        return None
    inventory = defaultdict(lambda: {"tensors": 0, "parameters": 0, "bytes": 0,
                                     "trainable_parameters": 0})
    for parameter in model.parameters():
        group = inventory[str(parameter.dtype).removeprefix("torch.")]
        group["tensors"] += 1
        group["parameters"] += parameter.numel()
        group["bytes"] += parameter.numel() * parameter.element_size()
        if parameter.requires_grad:
            group["trainable_parameters"] += parameter.numel()
    return dict(inventory)
