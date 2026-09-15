"""Multi-GPU plumbing: one process per GPU, PyTorch DistributedDataParallel (DDP).

Launch with torchrun and every process joins here:

    torchrun --standalone --nproc_per_node=8 scripts/train.py ...

Each rank holds a full copy of the model and trains on its own slice of every
optimizer step's data; DDP averages the gradients across ranks before the optimizer
step, so the update is mathematically the single-GPU update over the same data, just
computed on N cards at once. Not launched under torchrun -> plain single process, and
every helper here becomes a no-op. NCCL on CUDA, gloo on CPU (tests).
"""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def init_distributed() -> dict:
    """Join the torchrun process group if we were launched by torchrun; else single-process.
    Pins this process to its GPU (LOCAL_RANK) so a plain "cuda" device means the right card."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1 or "RANK" not in os.environ:
        return {"rank": 0, "world": 1, "local_rank": 0, "distributed": False}
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    # a long timeout: rank 0 alone runs eval + checkpoint saves while the others wait at the
    # next all-reduce, and a 1B checkpoint to a busy shared disk can take a while
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=60))
    return {"rank": rank, "world": world, "local_rank": local_rank, "distributed": True}


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main() -> bool:
    return rank() == 0


def _flag_device() -> str:
    return "cuda" if (is_distributed() and dist.get_backend() == "nccl") else "cpu"


def barrier() -> None:
    if is_distributed():
        if _flag_device() == "cuda":
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks in place (no-op single-process)."""
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size()
    return t


def broadcast_flag(flag: bool) -> bool:
    """Rank 0's decision, delivered to everyone (stop / pause must be unanimous or the
    ranks desync at the next collective)."""
    if not is_distributed():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=_flag_device())
    dist.broadcast(t, src=0)
    return bool(int(t.item()))


def wrap_ddp(model: torch.nn.Module):
    """DDP-wrap `model` when distributed; returns the module to run forward/backward through.
    Parameters are shared with `model`, so the optimizer / checkpoints keep using `model`."""
    if not is_distributed():
        return model
    from torch.nn.parallel import DistributedDataParallel as DDP
    dev_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
    # gradient_as_bucket_view: the all-reduce buckets ARE the .grad tensors (no second copy);
    # broadcast_buffers off: the only buffers are the RoPE tables, identical everywhere already
    return DDP(model, device_ids=dev_ids, gradient_as_bucket_view=True, broadcast_buffers=False,
               find_unused_parameters=False)


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across ranks in place (no-op single-process)."""
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def reduce_device() -> str:
    """Where a tensor must live to take part in a collective (cuda under NCCL, cpu under gloo)."""
    return _flag_device()
