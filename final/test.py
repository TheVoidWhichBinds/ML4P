# test.py
#================================================================================================================================================
# Tests the final InnerK checkpoint and saves VRMSE plus parity data.
#================================================================================================================================================

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import torch

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATASET_PATH,
    MAX_FILES,
    MAX_PARITY_SAMPLES,
    MAX_SAMPLES,
    MAX_TRAJECTORIES,
    TEST_LOG_PATH,
    TEST_PARITY_PATH,
    TEST_RUN_MAX_BATCHES,
    TEST_RUN_MAX_FILES,
    TEST_RUN_MAX_SAMPLES,
    TEST_RUN_MAX_TRAJECTORIES,
    TEST_SPLIT,
    TRAIN_CHECKPOINT_PATH,
)
from model import InnerK
from run import (
    assert_finite_tensor,
    compute_vrmse,
    format_float,
    get_device,
    make_loader,
    move_batch_to_device,
    print_epoch_metrics,
)








#==============================================================================================================
# CHECKPOINT LOADING
#==============================================================================================================

def load_checkpoint(checkpoint_path: Path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run run.py first, or pass --checkpoint-path to test.py."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location = "cpu",
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain model_state_dict: {checkpoint_path}"
        )

    return checkpoint_path, checkpoint





def build_model_from_checkpoint(
    checkpoint,
    device: torch.device,
):
    k_values = sorted(int(k) for k in checkpoint.get("k_values", []))

    if len(k_values) == 0:
        raise ValueError(
            "Checkpoint does not contain k_values. "
            "Use a checkpoint saved by the updated run.py."
        )

    model = InnerK(
        k_values = k_values,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict = True,
    )

    model = model.to(device)
    model.eval()

    checkpoint_epoch = int(checkpoint.get("epoch", checkpoint.get("final_epoch", 0)))

    return model, k_values, checkpoint_epoch








#==============================================================================================================
# PARITY CSV WRITING
#==============================================================================================================

def make_parity_writer(parity_path: Optional[Path]):
    if parity_path is None:
        return None, None

    parity_path = Path(parity_path).expanduser().resolve()
    parity_path.parent.mkdir(parents = True, exist_ok = True)

    parity_file = open(parity_path, "w", newline = "")

    fieldnames = [
        "k",
        "sample_index",
        "trajectory_index",
        "time_index",
        "i",
        "j",
        "channel",
        "target",
        "prediction",
        "residual",
    ]

    writer = csv.DictWriter(
        parity_file,
        fieldnames = fieldnames,
    )

    writer.writeheader()

    return parity_file, writer





def write_parity_rows(
    writer,
    k: int,
    sample_offset: int,
    metadata: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    remaining_samples: Optional[int],
) -> int:
    if writer is None:
        return 0

    batch_size = target.shape[0]

    if remaining_samples is None:
        samples_to_write = batch_size
    else:
        samples_to_write = max(0, min(batch_size, remaining_samples))

    if samples_to_write == 0:
        return 0

    metadata_cpu = metadata[:samples_to_write].detach().cpu().tolist()
    target_cpu = target[:samples_to_write].detach().cpu().tolist()
    prediction_cpu = prediction[:samples_to_write].detach().cpu().tolist()

    for local_index in range(samples_to_write):
        trajectory_index, time_index, i_index, j_index = metadata_cpu[local_index]

        for channel_index in range(len(target_cpu[local_index])):
            target_value = float(target_cpu[local_index][channel_index])
            prediction_value = float(prediction_cpu[local_index][channel_index])

            writer.writerow(
                {
                    "k": k,
                    "sample_index": sample_offset + local_index,
                    "trajectory_index": int(trajectory_index),
                    "time_index": int(time_index),
                    "i": int(i_index),
                    "j": int(j_index),
                    "channel": channel_index,
                    "target": target_value,
                    "prediction": prediction_value,
                    "residual": prediction_value - target_value,
                }
            )

    return samples_to_write








#==============================================================================================================
# TEST EVALUATION
#==============================================================================================================

def evaluate_test_split(
    model: InnerK,
    loader,
    k_values,
    device: torch.device,
    parity_path: Optional[Path],
    max_parity_samples: Optional[int],
    max_batches: Optional[int],
) -> Dict[str, float]:
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

    parity_counts = {
        k: 0
        for k in k_values
    }

    sample_offsets = {
        k: 0
        for k in k_values
    }

    num_batches = 0
    total_batches = len(loader)

    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    parity_file, parity_writer = make_parity_writer(
        parity_path = parity_path,
    )

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                batch = move_batch_to_device(
                    batch = batch,
                    device = device,
                )

                y = batch["y"]
                metadata = batch["metadata"]

                assert_finite_tensor("test target y", y)

                vrmse_values = []

                for k in k_values:
                    x_k = batch["x_by_k"][f"k_{k}"]
                    assert_finite_tensor(f"test input x_k for k = {k}", x_k)

                    y_hat = model(x_k)
                    assert_finite_tensor(f"test prediction y_hat for k = {k}", y_hat)

                    vrmse_k = compute_vrmse(
                        prediction = y_hat,
                        target = y,
                    )

                    assert_finite_tensor(f"test VRMSE for k = {k}", vrmse_k)

                    vrmse_values.append(vrmse_k)
                    vrmse_totals[k] += format_float(vrmse_k)

                    if parity_writer is not None:
                        if max_parity_samples is None:
                            remaining_samples = None
                        else:
                            remaining_samples = max_parity_samples - parity_counts[k]

                        samples_written = write_parity_rows(
                            writer = parity_writer,
                            k = k,
                            sample_offset = sample_offsets[k],
                            metadata = metadata,
                            target = y,
                            prediction = y_hat,
                            remaining_samples = remaining_samples,
                        )

                        parity_counts[k] += samples_written
                        sample_offsets[k] += int(y.shape[0])

                prediction_loss = torch.mean(torch.stack(vrmse_values))

                totals["prediction_loss"] += format_float(prediction_loss)
                totals["total_loss"] += format_float(prediction_loss)

                num_batches += 1

                print(
                    f"test progress | batch {num_batches}/{total_batches} | "
                    f"batch pooled VRMSE {format_float(prediction_loss):.6e}",
                    flush = True,
                )
    finally:
        if parity_file is not None:
            parity_file.close()

    if num_batches == 0:
        raise RuntimeError("No test batches were processed.")

    averaged = {
        key: value / num_batches
        for key, value in totals.items()
    }

    for k in k_values:
        averaged[f"vrmse_k_{k}"] = vrmse_totals[k] / num_batches

    return averaged





def save_test_log(
    log_path: Path,
    checkpoint_epoch: int,
    metrics: Dict[str, float],
    k_values,
) -> None:
    log_path = Path(log_path).expanduser().resolve()
    log_path.parent.mkdir(parents = True, exist_ok = True)

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

    row = {
        "epoch": checkpoint_epoch,
        "split": "test",
        "total_loss": metrics["total_loss"],
        "prediction_loss": metrics["prediction_loss"],
        "slope_loss": metrics["slope_loss"],
        "slope": metrics["slope"],
        "p": metrics["p"],
        "A": metrics["A"],
        "k_shift": metrics["k_shift"],
        "w": metrics["w"],
    }

    for k in k_values:
        row[f"vrmse_k_{k}"] = metrics[f"vrmse_k_{k}"]

    with open(log_path, "w", newline = "") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames = fieldnames,
        )

        writer.writeheader()
        writer.writerow(row)








#==============================================================================================================
# COMMAND LINE INTERFACE
#==============================================================================================================

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
        "--checkpoint-path",
        type = str,
        default = None,
        help = "Checkpoint from run.py. Defaults to outputs/checkpoints/inner_shared.pt.",
    )

    parser.add_argument(
        "--log-path",
        type = str,
        default = str(TEST_LOG_PATH),
        help = "Where to save the test VRMSE log CSV.",
    )

    parser.add_argument(
        "--parity-path",
        type = str,
        default = str(TEST_PARITY_PATH),
        help = "Where to save the test parity CSV.",
    )

    parser.add_argument(
        "--no-parity",
        action = "store_true",
        help = "Skip saving the parity CSV.",
    )

    parser.add_argument(
        "--max-parity-samples",
        type = int,
        default = MAX_PARITY_SAMPLES,
        help = "Maximum number of test samples per k to save for parity plotting. Use -1 for no limit.",
    )

    parser.add_argument(
        "--max-samples",
        type = int,
        default = None,
        help = "Optional override for the number of test samples loaded from the dataset.",
    )

    return parser.parse_args()





def main():
    args = parse_args()

    torch.manual_seed(0)

    device = get_device()

    checkpoint_path = args.checkpoint_path

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "inner_test_run.pt" if args.test_run else TRAIN_CHECKPOINT_PATH

    checkpoint_path, checkpoint = load_checkpoint(
        checkpoint_path = checkpoint_path,
    )

    model, k_values, checkpoint_epoch = build_model_from_checkpoint(
        checkpoint = checkpoint,
        device = device,
    )

    dataset_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path is not None else DATASET_PATH

    max_files = TEST_RUN_MAX_FILES if args.test_run else MAX_FILES
    max_trajectories = TEST_RUN_MAX_TRAJECTORIES if args.test_run else MAX_TRAJECTORIES

    if args.test_run:
        max_samples = TEST_RUN_MAX_SAMPLES
        max_batches = TEST_RUN_MAX_BATCHES
    else:
        max_samples = MAX_SAMPLES if args.max_samples is None else args.max_samples
        max_batches = None

    parity_path = None if args.no_parity else Path(args.parity_path)

    if args.max_parity_samples is None or args.max_parity_samples < 0:
        max_parity_samples = None
    else:
        max_parity_samples = int(args.max_parity_samples)

    print(f"Using device: {device}")
    print(f"Dataset path: {dataset_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint_epoch}")
    print(f"K_VALUES from checkpoint: {k_values}")
    print(f"Test split: {TEST_SPLIT}")

    test_loader = make_loader(
        split = TEST_SPLIT,
        dataset_path = dataset_path,
        k_values = k_values,
        batch_size = BATCH_SIZE,
        max_files = max_files,
        max_trajectories = max_trajectories,
        max_samples = max_samples,
        shuffle = False,
    )

    metrics = evaluate_test_split(
        model = model,
        loader = test_loader,
        k_values = k_values,
        device = device,
        parity_path = parity_path,
        max_parity_samples = max_parity_samples,
        max_batches = max_batches,
    )

    print_epoch_metrics(
        epoch = checkpoint_epoch,
        num_epochs = checkpoint_epoch,
        split = "test",
        metrics = metrics,
        k_values = k_values,
    )

    save_test_log(
        log_path = Path(args.log_path),
        checkpoint_epoch = checkpoint_epoch,
        metrics = metrics,
        k_values = k_values,
    )

    print(f"Saved test log: {Path(args.log_path).expanduser().resolve()}")

    if parity_path is not None:
        print(f"Saved parity data: {Path(parity_path).expanduser().resolve()}")





if __name__ == "__main__":
    main()
