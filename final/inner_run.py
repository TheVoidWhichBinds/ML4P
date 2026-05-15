# inner_run.py
#================================================================================================================================================
# Train one shared InnerK model over several temporal-history values k.
#================================================================================================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATASET_PATH,
    EPOCH_PROGRESS_DIVISIONS,
    EXPONENTIAL_FIT_WEIGHT,
    K_VALUES,
    GRAD_CLIP_MAX_NORM,
    LEARNING_RATE,
    LOG_DIR,
    MAX_FILES,
    MAX_SAMPLES,
    MAX_TRAJECTORIES,
    NUM_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    PREDICTION_WEIGHT,
    RANDOM_SEED,
    SLOPE_K,
    SLOPE_WEIGHT,
    SPATIAL_KERNEL_SIZE,
    TEST_RUN_MAX_BATCHES,
    TEST_RUN_MAX_FILES,
    TEST_RUN_MAX_SAMPLES,
    TEST_RUN_MAX_TRAJECTORIES,
    TEST_RUN_NUM_EPOCHS,
    TRAIN_SPLIT,
    VALID_SPLIT,
    WEIGHT_DECAY,
)
from data import MultiKPatchDataset, load_split
from inner_model import ExponentialSlopeLoss, InnerK








#==============================================================================================================
# UTILITIES
#==============================================================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")





def move_batch_to_device(batch, device: torch.device):
    x_by_k = {}

    for key, value in batch["x_by_k"].items():
        x_by_k[key] = value.to(device)

    y = batch["y"].to(device)
    metadata = batch["metadata"].to(device)

    return {
        "x_by_k": x_by_k,
        "y": y,
        "metadata": metadata,
    }





def format_float(value) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().item()

    return float(value)




def assert_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite tensor detected: {name}")




def assert_finite_model(model: torch.nn.Module, stage: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(
                f"Non-finite model parameter detected after {stage}: {name}"
            )





def print_epoch_metrics(
    epoch: int,
    num_epochs: int,
    split: str,
    metrics: Dict[str, float],
    k_values,
):
    print(
        f"epoch {epoch:04d}/{num_epochs:04d} | {split:>5s} | total MSE {metrics['total_loss']:.6e}",
        flush = True,
    )

    for k in k_values:
        print(
            f"    k = {k:03d} | MSE {metrics[f'mse_k_{k}']:.6e}",
            flush = True,
        )





def make_progress_checkpoints(
    total_batches: int,
    progress_divisions: int,
):
    if progress_divisions is None or progress_divisions <= 0:
        return set()

    checkpoints = set()

    for division_index in range(1, progress_divisions + 1):
        checkpoint = round(total_batches * division_index / progress_divisions)
        checkpoint = max(1, min(total_batches, checkpoint))
        checkpoints.add(checkpoint)

    return checkpoints





def print_epoch_progress(
    epoch: int,
    num_epochs: int,
    split: str,
    batch_number: int,
    total_batches: int,
    diagnostics: Dict[str, float],
    k_values,
):
    progress_percent = 100.0 * batch_number / total_batches

    print(
        f"epoch {epoch:04d}/{num_epochs:04d} | "
        f"{split:>5s} progress | "
        f"batch {batch_number}/{total_batches} "
        f"({progress_percent:5.1f}%) | "
        f"batch total MSE {diagnostics['total_loss']:.6e}",
        flush = True,
    )





def make_loader(
    split: str,
    dataset_path: Path,
    k_values,
    batch_size: int,
    max_files: Optional[int],
    max_trajectories: Optional[int],
    max_samples: Optional[int],
    shuffle: bool,
) -> DataLoader:
    states = load_split(
        split = split,
        dataset_path = dataset_path,
        max_files = max_files,
        max_trajectories = max_trajectories,
    )

    dataset = MultiKPatchDataset(
        states = states,
        k_values = k_values,
        patch_size = SPATIAL_KERNEL_SIZE,
        max_samples = max_samples,
    )

    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
    )

    return loader








#==============================================================================================================
# LOSS STEP
#==============================================================================================================

def compute_multik_loss(
    model: InnerK,
    slope_loss_fn: ExponentialSlopeLoss,
    batch,
    k_values,
    device: torch.device,
):
    batch = move_batch_to_device(
        batch = batch,
        device = device,
    )

    y = batch["y"]
    assert_finite_tensor("target y", y)

    mse_values = []
    mse_dict: Dict[int, torch.Tensor] = {}

    for k in k_values:
        x_k = batch["x_by_k"][f"k_{k}"]
        assert_finite_tensor(f"input x_k for k = {k}", x_k)

        y_hat = model(x_k)
        assert_finite_tensor(f"prediction y_hat for k = {k}", y_hat)

        mse_k = F.mse_loss(
            y_hat,
            y,
            reduction = "mean",
        )
        assert_finite_tensor(f"MSE for k = {k}", mse_k)

        mse_values.append(mse_k)
        mse_dict[k] = mse_k.detach()

    mse_values = torch.stack(mse_values)

    #------------------------------------------------------------------------------------------------------
    # Neutralized slope regularization:
    #
    # The old version called slope_loss_fn(k_values, mse_values), which made the backpropagated loss include
    # prediction loss, exponential-fit loss, and slope loss. For stability, the optimizer now sees only the
    # pooled prediction MSE across all k values.
    #------------------------------------------------------------------------------------------------------

    prediction_loss = torch.mean(mse_values)
    total_loss = prediction_loss
    assert_finite_tensor("total loss", total_loss)

    diagnostics = {
        "total_loss": total_loss.detach(),
        "prediction_loss": prediction_loss.detach(),
        "fit_loss": torch.zeros((), device = device, dtype = mse_values.dtype),
        "slope_loss": torch.zeros((), device = device, dtype = mse_values.dtype),
        "slope": torch.zeros((), device = device, dtype = mse_values.dtype),
        "p": torch.zeros((), device = device, dtype = mse_values.dtype),
        "A": torch.zeros((), device = device, dtype = mse_values.dtype),
        "k_shift": torch.zeros((), device = device, dtype = mse_values.dtype),
        "w": torch.zeros((), device = device, dtype = mse_values.dtype),
        "mse_by_k": {
            k: format_float(value)
            for k, value in mse_dict.items()
        },
    }

    return total_loss, diagnostics








#==============================================================================================================
# TRAIN / VALIDATE
#==============================================================================================================

def run_epoch(
    model: InnerK,
    slope_loss_fn: ExponentialSlopeLoss,
    loader: DataLoader,
    k_values,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
    split: Optional[str] = None,
    progress_divisions: int = 0,
):
    is_train = optimizer is not None

    model.train(is_train)
    slope_loss_fn.train(is_train)

    totals = {
        "total_loss": 0.0,
        "prediction_loss": 0.0,
        "fit_loss": 0.0,
        "slope_loss": 0.0,
        "slope": 0.0,
        "p": 0.0,
        "A": 0.0,
        "k_shift": 0.0,
        "w": 0.0,
    }

    mse_totals = {
        k: 0.0
        for k in k_values
    }

    num_batches = 0

    total_batches = len(loader)

    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    progress_checkpoints = make_progress_checkpoints(
        total_batches = total_batches,
        progress_divisions = progress_divisions,
    )

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            if is_train:
                optimizer.zero_grad(set_to_none = True)

            total_loss, diagnostics = compute_multik_loss(
                model = model,
                slope_loss_fn = slope_loss_fn,
                batch = batch,
                k_values = k_values,
                device = device,
            )

            if is_train:
                total_loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm = GRAD_CLIP_MAX_NORM,
                )

                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm before optimizer step: {format_float(grad_norm)}"
                    )

                optimizer.step()
                assert_finite_model(
                    model = model,
                    stage = "optimizer step",
                )

            for key in totals:
                totals[key] += format_float(diagnostics[key])

            for k in k_values:
                mse_totals[k] += diagnostics["mse_by_k"][k]

            num_batches += 1

            if (
                epoch is not None
                and num_epochs is not None
                and split is not None
                and num_batches in progress_checkpoints
            ):
                print_epoch_progress(
                    epoch = epoch,
                    num_epochs = num_epochs,
                    split = split,
                    batch_number = num_batches,
                    total_batches = total_batches,
                    diagnostics = diagnostics,
                    k_values = k_values,
                )

    if num_batches == 0:
        raise RuntimeError("No batches were processed.")

    averaged = {
        key: value / num_batches
        for key, value in totals.items()
    }

    for k in k_values:
        averaged[f"mse_k_{k}"] = mse_totals[k] / num_batches

    return averaged








#==============================================================================================================
# MAIN TRAINING ENTRYPOINT
#==============================================================================================================

def train_inner(
    test_run: bool = False,
    dataset_path: Optional[str] = None,
):
    torch.manual_seed(RANDOM_SEED)

    device = get_device()

    dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path is not None else DATASET_PATH

    if SLOPE_K not in K_VALUES:
        raise ValueError(
            f"SLOPE_K = {SLOPE_K} must be one of K_VALUES = {K_VALUES}."
        )

    num_epochs = TEST_RUN_NUM_EPOCHS if test_run else NUM_EPOCHS
    max_files = TEST_RUN_MAX_FILES if test_run else MAX_FILES
    max_trajectories = TEST_RUN_MAX_TRAJECTORIES if test_run else MAX_TRAJECTORIES
    max_samples = TEST_RUN_MAX_SAMPLES if test_run else MAX_SAMPLES
    max_batches = TEST_RUN_MAX_BATCHES if test_run else None

    CHECKPOINT_DIR.mkdir(parents = True, exist_ok = True)
    LOG_DIR.mkdir(parents = True, exist_ok = True)

    print(f"Using device: {device}")
    print(f"Dataset path: {dataset_path}")
    print(f"K_VALUES: {K_VALUES}")
    print(f"Learning rate: {LEARNING_RATE:.1e}")
    print(f"Gradient clipping max norm: {GRAD_CLIP_MAX_NORM:.1e}")
    print("Slope/fit loss: disabled; optimizing pooled MSE only")
    print(f"Test run: {test_run}")

    train_loader = make_loader(
        split = TRAIN_SPLIT,
        dataset_path = dataset_path,
        k_values = K_VALUES,
        batch_size = BATCH_SIZE,
        max_files = max_files,
        max_trajectories = max_trajectories,
        max_samples = max_samples,
        shuffle = True,
    )

    valid_loader = make_loader(
        split = VALID_SPLIT,
        dataset_path = dataset_path,
        k_values = K_VALUES,
        batch_size = BATCH_SIZE,
        max_files = max_files,
        max_trajectories = max_trajectories,
        max_samples = max_samples,
        shuffle = False,
    )

    model = InnerK(
        k = None,
    ).to(device)

    slope_loss_fn = ExponentialSlopeLoss(
        k_ref = SLOPE_K,
        fit_weight = EXPONENTIAL_FIT_WEIGHT,
        slope_weight = SLOPE_WEIGHT,
        prediction_weight = PREDICTION_WEIGHT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )

    log_path = LOG_DIR / ("inner_test_run_log.csv" if test_run else "inner_train_log.csv")
    checkpoint_path = CHECKPOINT_DIR / ("inner_test_run.pt" if test_run else "inner_shared.pt")

    fieldnames = [
        "epoch",
        "split",
        "total_loss",
        "prediction_loss",
    ] + [f"mse_k_{k}" for k in K_VALUES]

    with open(log_path, "w", newline = "") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames = fieldnames,
        )

        writer.writeheader()

        for epoch in range(1, num_epochs + 1):
            print(
                f"\nStarting epoch {epoch:04d}/{num_epochs:04d}",
                flush = True,
            )

            train_metrics = run_epoch(
                model = model,
                slope_loss_fn = slope_loss_fn,
                loader = train_loader,
                k_values = K_VALUES,
                device = device,
                optimizer = optimizer,
                max_batches = max_batches,
                epoch = epoch,
                num_epochs = num_epochs,
                split = "train",
                progress_divisions = EPOCH_PROGRESS_DIVISIONS,
            )

            print_epoch_metrics(
                epoch = epoch,
                num_epochs = num_epochs,
                split = "train",
                metrics = train_metrics,
                k_values = K_VALUES,
            )

            valid_metrics = run_epoch(
                model = model,
                slope_loss_fn = slope_loss_fn,
                loader = valid_loader,
                k_values = K_VALUES,
                device = device,
                optimizer = None,
                max_batches = max_batches,
                epoch = epoch,
                num_epochs = num_epochs,
                split = "valid",
                progress_divisions = EPOCH_PROGRESS_DIVISIONS,
            )

            print_epoch_metrics(
                epoch = epoch,
                num_epochs = num_epochs,
                split = "valid",
                metrics = valid_metrics,
                k_values = K_VALUES,
            )

            train_row = {
                "epoch": epoch,
                "split": "train",
                "total_loss": train_metrics["total_loss"],
                "prediction_loss": train_metrics["prediction_loss"],
            }

            valid_row = {
                "epoch": epoch,
                "split": "valid",
                "total_loss": valid_metrics["total_loss"],
                "prediction_loss": valid_metrics["prediction_loss"],
            }

            for k in K_VALUES:
                train_row[f"mse_k_{k}"] = train_metrics[f"mse_k_{k}"]
                valid_row[f"mse_k_{k}"] = valid_metrics[f"mse_k_{k}"]

            writer.writerow(train_row)
            writer.writerow(valid_row)
            log_file.flush()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "slope_loss_state_dict": slope_loss_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "k_values": K_VALUES,
            "slope_k": SLOPE_K,
        },
        checkpoint_path,
    )

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved log: {log_path}")





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

    train_inner(
        test_run = args.test_run,
        dataset_path = args.dataset_path,
    )
