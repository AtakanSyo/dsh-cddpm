# ddp_utils.py
import os
import torch
import torch.distributed as dist

def init_distributed(local_rank: int | None = None):
    """
    Initialize torch.distributed if WORLD_SIZE > 1.
    Returns: (is_distributed, rank, world_size, local_rank)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if is_distributed:
        # Prefer env var provided by torchrun
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank >= 0:
            local_rank = env_local_rank
        elif local_rank is None:
            local_rank = 0

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return True, rank, world_size, local_rank

    # single-process
    return False, 0, 1, 0

def get_device(local_rank: int | None = None):
    if torch.cuda.is_available():
        # Again prefer env var if present
        lr = int(os.environ.get("LOCAL_RANK", local_rank or 0))
        return torch.device("cuda", lr)
    return torch.device("cpu")

def is_rank0(is_distributed: bool, rank: int):
    return (not is_distributed) or rank == 0

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()