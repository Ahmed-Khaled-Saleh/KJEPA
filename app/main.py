# main_distributed.py

import argparse
import logging
import os
import pprint
from pathlib import Path

import yaml

# print current directory
print(f"Current directory: {os.getcwd()}")
from app.scaffold import main as app_main
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger


parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help="name of config file to load", default="configs.yaml")
parser.add_argument("--debugmode", action="store_true", help="Run in single-process debug mode", default= False)


def main_worker(rank: int, world_size: int, fname: str):
    
    import logging

    from src.utils.logging import get_logger

    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # Load config
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    logger.info("loaded params...")

    # Log config & save params
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        folder = Path(params["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        params_path = folder / "params-pretrain.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f)

    # Init distributed from torchrun-provided env vars
    # at top of process_main / main_worker
    world_size, rank, local_rank = init_distributed()
    
    # diagnostic
    import os, torch
    logger.info(f"RANK={rank} WORLD_SIZE={world_size} LOCAL_RANK={local_rank}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logger.info(f"torch.cuda.device_count()={torch.cuda.device_count()} current_device={torch.cuda.current_device()}")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # logger.info(f"Running... (rank: {rank}/{world_size})")

    # Launch app
    app_main(params["app"], args=params)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.debugmode:
        main_worker(rank=0, world_size=1, fname=args.fname)
    else:
        # torchrun will handle ranks/world_size automatically
        # just read RANK, WORLD_SIZE env vars inside init_distributed
        main_worker(rank=int(os.environ.get("RANK", 0)),
                    world_size=int(os.environ.get("WORLD_SIZE", 1)),
                    fname=args.fname)
