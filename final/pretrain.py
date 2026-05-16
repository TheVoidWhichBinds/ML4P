# pretrain.py
#================================================================================================================================================
# First training stage after data generation: train InnerK only at PRETRAIN_K before the multi-k run.
#================================================================================================================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from config import (
    PRETRAIN_CHECKPOINT_PATH,
    PRETRAIN_K,
    PRETRAIN_LOG_PATH,
    PRETRAIN_NUM_EPOCHS,
    PRETRAIN_TEST_CHECKPOINT_PATH,
    PRETRAIN_TEST_LOG_PATH,
)
from run import train_inner








#==============================================================================================================
# PRETRAINING ENTRYPOINT
#==============================================================================================================

def make_pretrain_paths(
    test_run: bool,
    log_path = None,
    checkpoint_path = None,
):
    #------------------------------------------------------------------------------------------------------
    # Pretraining paths are controlled by config.py.
    #------------------------------------------------------------------------------------------------------

    if log_path is None:
        log_path = PRETRAIN_TEST_LOG_PATH if test_run else PRETRAIN_LOG_PATH

    if checkpoint_path is None:
        checkpoint_path = PRETRAIN_TEST_CHECKPOINT_PATH if test_run else PRETRAIN_CHECKPOINT_PATH

    return Path(log_path), Path(checkpoint_path)





def pretrain(
    test_run: bool = False,
    dataset_path: Optional[str] = None,
    log_path = None,
    checkpoint_path = None,
):
    pretrain_k = int(PRETRAIN_K)

    if pretrain_k <= 0:
        raise ValueError(f"pretraining k must be positive, but got k = {pretrain_k}.")

    log_path, checkpoint_path = make_pretrain_paths(
        test_run = test_run,
        log_path = log_path,
        checkpoint_path = checkpoint_path,
    )

    train_inner(
        test_run = test_run,
        dataset_path = dataset_path,
        k_values = [pretrain_k],
        log_path = log_path,
        checkpoint_path = checkpoint_path,
        load_pretrained = False,
        num_epochs_override = PRETRAIN_NUM_EPOCHS,
        use_slope_loss = False,
    )





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-run",
        action = "store_true",
        help = "Run one short smoke test using the test-run limits in config.py.",
    )

    parser.add_argument(
        "--dataset-path",
        type = str,
        default = None,
        help = "Optional path to the turbulent_radiative_layer_2D data directory.",
    )

    parser.add_argument(
        "--log-path",
        type = str,
        default = None,
        help = "Optional CSV log path for this pretraining run.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type = str,
        default = None,
        help = "Optional output checkpoint path for this pretraining run.",
    )

    return parser.parse_args()





if __name__ == "__main__":
    args = parse_args()

    pretrain(
        test_run = args.test_run,
        dataset_path = args.dataset_path,
        log_path = args.log_path,
        checkpoint_path = args.checkpoint_path,
    )
