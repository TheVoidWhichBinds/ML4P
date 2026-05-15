# pretrain_inner.py
#================================================================================================================================================
# First training stage after data generation: train InnerK only at k = 100 before the multi-k run.
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
from inner_run import train_inner








#==============================================================================================================
# K = 100 PRETRAINING ENTRYPOINT
#==============================================================================================================

def pretrain_inner(
    test_run: bool = False,
    dataset_path: Optional[str] = None,
):
    log_path = PRETRAIN_TEST_LOG_PATH if test_run else PRETRAIN_LOG_PATH
    checkpoint_path = PRETRAIN_TEST_CHECKPOINT_PATH if test_run else PRETRAIN_CHECKPOINT_PATH

    train_inner(
        test_run = test_run,
        dataset_path = dataset_path,
        k_values = [PRETRAIN_K],
        log_path = Path(log_path),
        checkpoint_path = Path(checkpoint_path),
        load_pretrained = False,
        num_epochs_override = PRETRAIN_NUM_EPOCHS,
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

    return parser.parse_args()





if __name__ == "__main__":
    args = parse_args()

    pretrain_inner(
        test_run = args.test_run,
        dataset_path = args.dataset_path,
    )
