# run.py
#================================================================================================================================================
# Train one shared InnerK model over several temporal-history values k.
#================================================================================================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATASET_PATH,
    EPOCH_PROGRESS_DIVISIONS,
    K_VALUES,
    GRAD_CLIP_MAX_NORM,
    LEARNING_RATE,
    LOAD_PRETRAINED_INNERK,
    LOG_DIR,
    MAX_FILES,
    MAX_SAMPLES,
    MAX_TRAJECTORIES,
    NUM_EPOCHS,
    NUM_WORKERS,
    PIN_MEMORY,
    PREDICTION_WEIGHT,
    PRETRAIN_CHECKPOINT_PATH,
    PRETRAIN_TEST_CHECKPOINT_PATH,
    RANDOM_SEED,
    SLOPE_K,
    SLOPE_WEIGHT,
    SLOPE_FIT_LR,
    SLOPE_FIT_STEPS,
    slope_loss,
    SPATIAL_KERNEL_SIZE,
    TEST_RUN_MAX_BATCHES,
    TEST_RUN_MAX_FILES,
    TEST_RUN_MAX_SAMPLES,
    TEST_RUN_MAX_TRAJECTORIES,
    TEST_RUN_NUM_EPOCHS,
    TRAIN_CHECKPOINT_PATH,
    TRAIN_LOG_PATH,
    TRAIN_SPLIT,
    VALID_SPLIT,
    VRMSE_EPS,
    WEIGHT_DECAY,
)
from data import MultiKPatchDataset, load_split
from model import ExponentialSlopeLoss, InnerK








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
        f"epoch {epoch:04d}/{num_epochs:04d} | {split:>5s} | "
        f"total loss {metrics['total_loss']:.6e} | "
        f"pooled VRMSE {metrics['prediction_loss']:.6e}",
        flush = True,
    )

    for k in k_values:
        print(
            f"    k = {k:03d} | VRMSE {metrics[f'vrmse_k_{k}']:.6e}",
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
        f"batch total loss {diagnostics['total_loss']:.6e}",
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





def compute_vrmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = VRMSE_EPS,
) -> torch.Tensor:
    #------------------------------------------------------------------------------------------------------
    # VRMSE = sqrt(MSE / Var(target)), averaged over channels.
    # For the current local model, prediction and target both have shape [B, 4], so the variance is computed
    # over the batch for each physical channel.
    #------------------------------------------------------------------------------------------------------

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, but got {prediction.shape} and {target.shape}."
        )

    channel_dim = target.ndim - 1
    reduction_dims = tuple(
        dim
        for dim in range(target.ndim)
        if dim != channel_dim
    )

    mse_by_channel = torch.mean(
        (prediction - target) ** 2,
        dim = reduction_dims,
    )

    variance_by_channel = torch.var(
        target,
        dim = reduction_dims,
        unbiased = False,
    )

    vrmse_by_channel = torch.sqrt(
        mse_by_channel / (variance_by_channel + eps)
    )

    vrmse = torch.mean(vrmse_by_channel)

    return vrmse





def load_model_checkpoint(
    model: InnerK,
    checkpoint_path: Path,
):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained InnerK checkpoint not found: {checkpoint_path}. "
            f"Run pretrain.py first, or pass --no-pretrained to start from scratch."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location = "cpu",
    )

    model_state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    incompatible_keys = model.load_state_dict(
        model_state_dict,
        strict = False,
    )

    if incompatible_keys.missing_keys:
        print(
            "Checkpoint did not contain these newly initialized model parameters: "
            f"{incompatible_keys.missing_keys}"
        )

    if incompatible_keys.unexpected_keys:
        print(
            "Checkpoint contained unused parameters: "
            f"{incompatible_keys.unexpected_keys}"
        )

    print(f"Loaded pretrained InnerK checkpoint: {checkpoint_path}")





def save_training_checkpoint(
    model: InnerK,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    k_values,
    active_slope_k: int,
    use_slope_loss: bool,
    epoch: int,
    log_path: Path,
) -> None:
    #------------------------------------------------------------------------------------------------------
    # Save the current learned model parameters, including the TaylorActivation theta values.
    # This checkpoint is updated after every completed epoch so test.py can load the latest completed state.
    #------------------------------------------------------------------------------------------------------

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "k_values": list(k_values),
            "slope_k": active_slope_k,
            "slope_loss": use_slope_loss,
            "epoch": int(epoch),
            "final_epoch": int(epoch),
            "log_path": str(log_path),
        },
        checkpoint_path,
    )








#==============================================================================================================
# LOSS STEP
#==============================================================================================================

def compute_multik_loss(
    model: InnerK,
    slope_loss_fn: ExponentialSlopeLoss,
    batch,
    k_values,
    device: torch.device,
    use_slope_loss: bool,
):
    batch = move_batch_to_device(
        batch = batch,
        device = device,
    )

    y = batch["y"]
    assert_finite_tensor("target y", y)

    vrmse_values = []
    vrmse_dict: Dict[int, torch.Tensor] = {}

    for k in k_values:
        x_k = batch["x_by_k"][f"k_{k}"]
        assert_finite_tensor(f"input x_k for k = {k}", x_k)

        y_hat = model(x_k)
        assert_finite_tensor(f"prediction y_hat for k = {k}", y_hat)

        vrmse_k = compute_vrmse(
            prediction = y_hat,
            target = y,
        )
        assert_finite_tensor(f"VRMSE for k = {k}", vrmse_k)

        vrmse_values.append(vrmse_k)
        vrmse_dict[k] = vrmse_k.detach()

    vrmse_values = torch.stack(vrmse_values)

    #------------------------------------------------------------------------------------------------------
    # Loss selection:
    #
    #     slope_loss = False:
    #         optimize pooled VRMSE only
    #
    #     slope_loss = True:
    #         optimize pooled VRMSE + exponential slope loss
    #
    # The exponential curve parameters are refit from the current VRMSE(k) values. They are not trainable
    # model parameters and are not included in the optimizer.
    #------------------------------------------------------------------------------------------------------

    if use_slope_loss:
        k_tensor = torch.tensor(
            k_values,
            device = device,
            dtype = vrmse_values.dtype,
        )

        total_loss, diagnostics = slope_loss_fn(
            k_values = k_tensor,
            vrmse_values = vrmse_values,
        )
    else:
        prediction_loss = torch.mean(vrmse_values)
        total_loss = prediction_loss

        diagnostics = {
            "total_loss": total_loss.detach(),
            "prediction_loss": prediction_loss.detach(),
            "slope_loss": torch.zeros((), device = device, dtype = vrmse_values.dtype),
            "slope": torch.zeros((), device = device, dtype = vrmse_values.dtype),
            "p": torch.zeros((), device = device, dtype = vrmse_values.dtype),
            "A": torch.zeros((), device = device, dtype = vrmse_values.dtype),
            "k_shift": torch.zeros((), device = device, dtype = vrmse_values.dtype),
            "w": torch.zeros((), device = device, dtype = vrmse_values.dtype),
        }

    assert_finite_tensor("total loss", total_loss)

    diagnostics["vrmse_by_k"] = {
        k: format_float(value)
        for k, value in vrmse_dict.items()
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
    use_slope_loss: bool = False,
):
    is_train = optimizer is not None

    model.train(is_train)
    slope_loss_fn.train(is_train)

    totals = {
        "total_loss": 0.0,
        "prediction_loss": 0.0,
        "slope_loss": 0.0,
        "slope": 0.0,
        "p": 0.0,
        "A": 0.0,
        "k_shift": 0.0,
        "w": 0.0,
    }

    vrmse_totals = {
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
                use_slope_loss = use_slope_loss,
            )

            if is_train:
                total_loss.backward()

                optimizer_parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ]

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    optimizer_parameters,
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
                vrmse_totals[k] += diagnostics["vrmse_by_k"][k]

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
        averaged[f"vrmse_k_{k}"] = vrmse_totals[k] / num_batches

    return averaged








#==============================================================================================================
# MAIN TRAINING ENTRYPOINT
#==============================================================================================================

def train_inner(
    test_run: bool = False,
    dataset_path: Optional[str] = None,
    k_values = None,
    log_path = None,
    checkpoint_path = None,
    pretrained_checkpoint_path = None,
    load_pretrained: Optional[bool] = None,
    num_epochs_override: Optional[int] = None,
    use_slope_loss: Optional[bool] = None,
):
    torch.manual_seed(RANDOM_SEED)

    device = get_device()

    dataset_path = Path(dataset_path).expanduser().resolve() if dataset_path is not None else DATASET_PATH
    k_values = list(K_VALUES) if k_values is None else [int(k) for k in k_values]
    k_values = sorted(set(k_values))

    if len(k_values) == 0:
        raise ValueError("At least one k value is required.")

    if any(k <= 0 for k in k_values):
        raise ValueError(f"All k values must be positive integers, but got {k_values}.")

    if use_slope_loss is None:
        use_slope_loss = bool(slope_loss)

    if use_slope_loss and len(k_values) < 4:
        raise ValueError(
            "slope_loss = True requires at least four k values so p, A, k_shift, and w can be fit. "
            f"Current k_values = {k_values}."
        )

    if use_slope_loss and SLOPE_K not in k_values:
        raise ValueError(
            f"slope_loss = True requires SLOPE_K = {SLOPE_K} to appear in K_VALUES = {k_values}."
        )

    active_slope_k = SLOPE_K if use_slope_loss else k_values[0]

    num_epochs = TEST_RUN_NUM_EPOCHS if test_run else (num_epochs_override or NUM_EPOCHS)
    max_files = TEST_RUN_MAX_FILES if test_run else MAX_FILES
    max_trajectories = TEST_RUN_MAX_TRAJECTORIES if test_run else MAX_TRAJECTORIES
    max_samples = TEST_RUN_MAX_SAMPLES if test_run else MAX_SAMPLES
    max_batches = TEST_RUN_MAX_BATCHES if test_run else None

    CHECKPOINT_DIR.mkdir(parents = True, exist_ok = True)
    LOG_DIR.mkdir(parents = True, exist_ok = True)

    print(f"Using device: {device}")
    print(f"Dataset path: {dataset_path}")
    print(f"K_VALUES: {k_values}")
    print(f"Learning rate: {LEARNING_RATE:.1e}")
    print(f"Gradient clipping max norm: {GRAD_CLIP_MAX_NORM:.1e}")

    if use_slope_loss:
        print(
            "Slope loss: enabled; optimizing pooled VRMSE plus exponential slope loss"
        )
        print(f"SLOPE_K: {active_slope_k}")
        print(f"SLOPE_WEIGHT: {SLOPE_WEIGHT:.1e}")
        print(f"SLOPE_FIT_STEPS: {SLOPE_FIT_STEPS}")
        print(f"SLOPE_FIT_LR: {SLOPE_FIT_LR:.1e}")
    else:
        print("Slope loss: disabled; optimizing pooled VRMSE only")

    print(f"Test run: {test_run}")

    train_loader = make_loader(
        split = TRAIN_SPLIT,
        dataset_path = dataset_path,
        k_values = k_values,
        batch_size = BATCH_SIZE,
        max_files = max_files,
        max_trajectories = max_trajectories,
        max_samples = max_samples,
        shuffle = True,
    )

    valid_loader = make_loader(
        split = VALID_SPLIT,
        dataset_path = dataset_path,
        k_values = k_values,
        batch_size = BATCH_SIZE,
        max_files = max_files,
        max_trajectories = max_trajectories,
        max_samples = max_samples,
        shuffle = False,
    )

    model = InnerK(
        k_values = k_values,
    ).to(device)

    if load_pretrained is None:
        load_pretrained = LOAD_PRETRAINED_INNERK

    if load_pretrained:
        if pretrained_checkpoint_path is None:
            pretrained_checkpoint_path = PRETRAIN_TEST_CHECKPOINT_PATH if test_run else PRETRAIN_CHECKPOINT_PATH

        load_model_checkpoint(
            model = model,
            checkpoint_path = Path(pretrained_checkpoint_path),
        )

    slope_loss_fn = ExponentialSlopeLoss(
        k_ref = active_slope_k,
        slope_weight = SLOPE_WEIGHT,
        prediction_weight = PREDICTION_WEIGHT,
        fit_steps = SLOPE_FIT_STEPS,
        fit_lr = SLOPE_FIT_LR,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )

    log_path = Path(log_path) if log_path is not None else (LOG_DIR / "inner_test_run_log.csv" if test_run else TRAIN_LOG_PATH)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else (CHECKPOINT_DIR / "inner_test_run.pt" if test_run else TRAIN_CHECKPOINT_PATH)

    log_path.parent.mkdir(parents = True, exist_ok = True)
    checkpoint_path.parent.mkdir(parents = True, exist_ok = True)

    fieldnames = [
        "epoch",
        "split",
        "total_loss",
        "prediction_loss",
        "slope_loss",
        "slope",
        "p",
        "A",
        "k_shift",
        "w",
    ] + [f"vrmse_k_{k}" for k in k_values]

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
                k_values = k_values,
                device = device,
                optimizer = optimizer,
                max_batches = max_batches,
                epoch = epoch,
                num_epochs = num_epochs,
                split = "train",
                progress_divisions = EPOCH_PROGRESS_DIVISIONS,
                use_slope_loss = use_slope_loss,
            )

            print_epoch_metrics(
                epoch = epoch,
                num_epochs = num_epochs,
                split = "train",
                metrics = train_metrics,
                k_values = k_values,
            )

            valid_metrics = run_epoch(
                model = model,
                slope_loss_fn = slope_loss_fn,
                loader = valid_loader,
                k_values = k_values,
                device = device,
                optimizer = None,
                max_batches = max_batches,
                epoch = epoch,
                num_epochs = num_epochs,
                split = "valid",
                progress_divisions = EPOCH_PROGRESS_DIVISIONS,
                use_slope_loss = use_slope_loss,
            )

            print_epoch_metrics(
                epoch = epoch,
                num_epochs = num_epochs,
                split = "valid",
                metrics = valid_metrics,
                k_values = k_values,
            )

            train_row = {
                "epoch": epoch,
                "split": "train",
                "total_loss": train_metrics["total_loss"],
                "prediction_loss": train_metrics["prediction_loss"],
                "slope_loss": train_metrics["slope_loss"],
                "slope": train_metrics["slope"],
                "p": train_metrics["p"],
                "A": train_metrics["A"],
                "k_shift": train_metrics["k_shift"],
                "w": train_metrics["w"],
            }

            valid_row = {
                "epoch": epoch,
                "split": "valid",
                "total_loss": valid_metrics["total_loss"],
                "prediction_loss": valid_metrics["prediction_loss"],
                "slope_loss": valid_metrics["slope_loss"],
                "slope": valid_metrics["slope"],
                "p": valid_metrics["p"],
                "A": valid_metrics["A"],
                "k_shift": valid_metrics["k_shift"],
                "w": valid_metrics["w"],
            }

            for k in k_values:
                train_row[f"vrmse_k_{k}"] = train_metrics[f"vrmse_k_{k}"]
                valid_row[f"vrmse_k_{k}"] = valid_metrics[f"vrmse_k_{k}"]

            writer.writerow(train_row)
            writer.writerow(valid_row)
            log_file.flush()

            save_training_checkpoint(
                model = model,
                optimizer = optimizer,
                checkpoint_path = checkpoint_path,
                k_values = k_values,
                active_slope_k = active_slope_k,
                use_slope_loss = use_slope_loss,
                epoch = epoch,
                log_path = log_path,
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

    parser.add_argument(
        "--no-pretrained",
        action = "store_true",
        help = "Start from scratch instead of loading a pretraining checkpoint.",
    )

    parser.add_argument(
        "--log-path",
        type = str,
        default = None,
        help = "Optional CSV log path. Defaults to outputs/logs/inner_train_log.csv.",
    )

    parser.add_argument(
        "--checkpoint-path",
        type = str,
        default = None,
        help = "Optional output checkpoint path. Defaults to outputs/checkpoints/inner_shared.pt.",
    )

    parser.add_argument(
        "--pretrained-checkpoint",
        type = str,
        default = None,
        help = "Optional checkpoint to load for the warm start.",
    )

    return parser.parse_args()





if __name__ == "__main__":
    args = parse_args()

    train_inner(
        test_run = args.test_run,
        dataset_path = args.dataset_path,
        k_values = None,
        log_path = args.log_path,
        checkpoint_path = args.checkpoint_path,
        pretrained_checkpoint_path = args.pretrained_checkpoint,
        load_pretrained = not args.no_pretrained,
    )
