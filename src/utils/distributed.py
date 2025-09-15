# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import torch
import torch.distributed as dist

import subprocess

import logging

from src.utils.logging import get_logger

logger = get_logger()



# logger = logging.getLogger(__name__)

def init_distributed(port=29500, rank_and_world_size=(None, None)):
    # tmpdir trick for some clusters
    if "SLURM_JOB_ID" in os.environ:
        tmpdir = Path(f"/scratch/slurm_tmpdir/{os.environ['SLURM_JOB_ID']}")
        if tmpdir.exists():
            os.environ["TMPDIR"] = str(tmpdir)

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))

    rank, world_size = rank_and_world_size
    master_addr = os.environ.get("MASTER_ADDR", None)
    master_port = os.environ.get("MASTER_PORT", str(port))

    # Prefer torchrun env vars if present
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Else SLURM
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        # pick first node in allocation for rendezvous if MASTER_ADDR not set
        if master_addr is None and "SLURM_NODELIST" in os.environ:
            try:
                hostnames = subprocess.check_output(
                    ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]]
                ).decode().split()
                master_addr = hostnames[0]
            except Exception:
                master_addr = os.environ.get("HOSTNAME", "127.0.0.1")
    # Else manual mode (you passed rank_and_world_size)
    elif (rank is not None) and (world_size is not None):
        local_rank = 0
        master_addr = master_addr or "127.0.0.1"
    else:
        # single-process fallback
        logger.info("No distributed env detected — running single-process")
        return 1, 0, 0

    # Ensure MASTER_ADDR/PORT are set and use IPv4 to avoid errno 97
    os.environ.setdefault("MASTER_ADDR", master_addr or "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", master_port)

    # Set CUDA device according to local_rank (ordinal in visible devices)
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(local_rank)
        except Exception as e:
            logger.error(f"Failed to set CUDA device {local_rank}. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}. Error: {e}")
            raise

    # Initialize process group if needed
    if world_size > 1:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        logger.info(f"Initialized DDP: rank={rank} world_size={world_size} local_rank={local_rank}")
    else:
        logger.info("Single process (world_size <= 1) — skipping init_process_group")

    return world_size, rank, local_rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
