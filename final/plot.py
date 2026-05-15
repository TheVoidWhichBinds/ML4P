# plot.py

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import K_VALUES, LOG_DIR, OUTPUT_DIR








#================================================================================================================================================
# PATHS
#================================================================================================================================================

PLOT_DIR = OUTPUT_DIR / "plots"
DEFAULT_LOG_PATH = LOG_DIR / "inner_train_log.csv"
DEFAULT_TEST_LOG_PATH = LOG_DIR / "inner_test_run_log.csv"








#================================================================================================================================================
# HELPERS
#================================================================================================================================================

def parse_k_values_from_log_columns(columns):
    #------------------------------------------------------------------------------------------------------
    # Extract k values from columns named mse_k_10, mse_k_25, ... .
    #------------------------------------------------------------------------------------------------------

    parsed_k_values = []

    for column in columns:
        if not column.startswith("mse_k_"):
            continue

        k_string = column.replace("mse_k_", "")

        try:
            parsed_k_values.append(int(k_string))
        except ValueError:
            continue

    parsed_k_values = sorted(parsed_k_values)

    return parsed_k_values





def choose_epoch_subset(epochs, divisions):
    #------------------------------------------------------------------------------------------------------
    # Choose epochs to plot using stride = total_epochs // divisions.
    # Epoch 0 is always included when it exists in the log.
    #------------------------------------------------------------------------------------------------------

    if divisions <= 0:
        raise ValueError("divisions must be a positive integer.")

    epochs = sorted(set(int(epoch) for epoch in epochs))

    if len(epochs) == 0:
        raise ValueError("No epochs were found in the log file.")

    minimum_epoch = min(epochs)
    total_epochs = max(epochs)
    stride = max(1, total_epochs // divisions)

    selected_epochs = [
        epoch
        for epoch in epochs
        if epoch == minimum_epoch or epoch % stride == 0
    ]

    if 0 in epochs and 0 not in selected_epochs:
        selected_epochs.append(0)

    if len(selected_epochs) == 0 or selected_epochs[-1] != total_epochs:
        selected_epochs.append(total_epochs)

    selected_epochs = sorted(set(selected_epochs))

    return selected_epochs, stride





def opacity_for_index(index, count, minimum_opacity = 0.25, maximum_opacity = 1.0):
    #------------------------------------------------------------------------------------------------------
    # Make later epoch curves more opaque.
    #------------------------------------------------------------------------------------------------------

    if count <= 1:
        return maximum_opacity

    fraction = index / (count - 1)
    opacity = minimum_opacity + fraction * (maximum_opacity - minimum_opacity)

    return opacity





def load_log(log_path, split):
    #------------------------------------------------------------------------------------------------------
    # Load the CSV log and keep only the requested split.
    #------------------------------------------------------------------------------------------------------

    log_path = Path(log_path).expanduser().resolve()

    if not log_path.exists():
        raise FileNotFoundError(f"Could not find log file: {log_path}")

    log_df = pd.read_csv(log_path)

    required_columns = {"epoch", "split"}
    missing_columns = required_columns.difference(log_df.columns)

    if missing_columns:
        raise ValueError(f"Log file is missing required columns: {sorted(missing_columns)}")

    split_df = log_df[log_df["split"] == split].copy()

    if split_df.empty:
        available_splits = sorted(log_df["split"].dropna().unique().tolist())
        raise ValueError(f"Split '{split}' was not found. Available splits: {available_splits}")

    split_df["epoch"] = split_df["epoch"].astype(int)
    split_df = split_df.sort_values("epoch")

    return split_df





def build_epoch_mse_table(split_df, k_values):
    #------------------------------------------------------------------------------------------------------
    # Convert wide log columns into one row per epoch and one column per k.
    #------------------------------------------------------------------------------------------------------

    mse_columns = [f"mse_k_{k}" for k in k_values]
    missing_columns = [column for column in mse_columns if column not in split_df.columns]

    if missing_columns:
        raise ValueError(f"Log file is missing MSE columns: {missing_columns}")

    mse_df = split_df[["epoch"] + mse_columns].copy()

    for column in mse_columns:
        mse_df[column] = pd.to_numeric(mse_df[column], errors = "coerce")

    return mse_df



def ensure_epoch_zero(mse_df):
    #------------------------------------------------------------------------------------------------------
    # Ensure an epoch-0 row exists for plotting.
    # If the log does not contain epoch 0, the earliest available row is copied and relabeled as epoch 0.
    # This is useful because the first logged training loss is the untrained-model baseline in the current run.
    #------------------------------------------------------------------------------------------------------

    mse_df = mse_df.copy()

    if 0 in mse_df["epoch"].astype(int).tolist():
        return mse_df, False

    earliest_epoch = int(mse_df["epoch"].min())
    epoch_zero_row = mse_df[mse_df["epoch"] == earliest_epoch].iloc[[0]].copy()
    epoch_zero_row["epoch"] = 0

    mse_df = pd.concat([epoch_zero_row, mse_df], ignore_index = True)
    mse_df = mse_df.sort_values("epoch").reset_index(drop = True)

    return mse_df, True








#================================================================================================================================================
# PLOTTING
#================================================================================================================================================

def plot_mse_curves(
    log_path,
    split,
    divisions,
    output_path,
    use_log_y,
):
    #------------------------------------------------------------------------------------------------------
    # Plot MSE(k) curves for selected epochs.
    #------------------------------------------------------------------------------------------------------

    split_df = load_log(
        log_path = log_path,
        split = split,
    )

    k_values = parse_k_values_from_log_columns(split_df.columns)

    if len(k_values) == 0:
        k_values = list(K_VALUES)

    mse_df = build_epoch_mse_table(
        split_df = split_df,
        k_values = k_values,
    )

    mse_df, used_synthetic_epoch_zero = ensure_epoch_zero(
        mse_df = mse_df,
    )

    selected_epochs, stride = choose_epoch_subset(
        epochs = mse_df["epoch"].tolist(),
        divisions = divisions,
    )

    selected_df = mse_df[mse_df["epoch"].isin(selected_epochs)].copy()

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (8.0, 5.5))

    for index, (_, row) in enumerate(selected_df.iterrows()):
        epoch = int(row["epoch"])
        mse_values = [row[f"mse_k_{k}"] for k in k_values]
        alpha = opacity_for_index(
            index = index,
            count = len(selected_df),
        )

        plt.scatter(
            k_values,
            mse_values,
            alpha = alpha,
            label = f"epoch {epoch}",
        )

    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.title(f"MSE vs. temporal history length k ({split}, scatter by epoch)")

    if use_log_y:
        plt.yscale("log")

    plt.xticks(k_values)
    plt.grid(True, alpha = 0.25)
    plt.legend(title = f"stride = {stride}")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()

    return output_path, selected_epochs, stride, used_synthetic_epoch_zero




def plot_epoch_zero_vs_final(
    log_path,
    split,
    output_path,
    use_log_y,
):
    #------------------------------------------------------------------------------------------------------
    # Plot only epoch 0 and the final epoch, with lines connecting the k values within each epoch.
    #------------------------------------------------------------------------------------------------------

    split_df = load_log(
        log_path = log_path,
        split = split,
    )

    k_values = parse_k_values_from_log_columns(split_df.columns)

    if len(k_values) == 0:
        k_values = list(K_VALUES)

    mse_df = build_epoch_mse_table(
        split_df = split_df,
        k_values = k_values,
    )

    mse_df, used_synthetic_epoch_zero = ensure_epoch_zero(
        mse_df = mse_df,
    )

    final_epoch = int(mse_df["epoch"].max())
    comparison_epochs = [0, final_epoch]
    comparison_df = mse_df[mse_df["epoch"].isin(comparison_epochs)].copy()

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (8.0, 5.5))

    for _, row in comparison_df.iterrows():
        epoch = int(row["epoch"])
        mse_values = [row[f"mse_k_{k}"] for k in k_values]

        plt.plot(
            k_values,
            mse_values,
            marker = "o",
            linewidth = 1.5,
            label = f"epoch {epoch}",
        )

    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.title(f"Epoch 0 vs. final epoch MSE ({split})")

    if use_log_y:
        plt.yscale("log")

    plt.xticks(k_values)
    plt.grid(True, alpha = 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()

    return output_path, comparison_epochs, used_synthetic_epoch_zero








#================================================================================================================================================
# COMMAND LINE INTERFACE
#================================================================================================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-path",
        type = str,
        default = str(DEFAULT_LOG_PATH),
        help = "Path to inner_run.py CSV log file.",
    )

    parser.add_argument(
        "--test-log",
        action = "store_true",
        help = "Use outputs/logs/inner_test_run_log.csv instead of inner_train_log.csv.",
    )

    parser.add_argument(
        "--split",
        type = str,
        default = "valid",
        choices = ["train", "valid"],
        help = "Which split to plot from the CSV log.",
    )

    parser.add_argument(
        "--divisions",
        type = int,
        default = 4,
        help = "Number of epoch divisions. The plotted stride is total_epochs // divisions.",
    )

    parser.add_argument(
        "--output-path",
        type = str,
        default = str(PLOT_DIR / "mse_vs_k_by_epoch.png"),
        help = "Where to save the scatter-by-epoch plot.",
    )

    parser.add_argument(
        "--comparison-output-path",
        type = str,
        default = str(PLOT_DIR / "mse_vs_k_epoch0_vs_final.png"),
        help = "Where to save the epoch-0-versus-final connected-line plot.",
    )

    parser.add_argument(
        "--linear-y",
        action = "store_true",
        help = "Use a linear y-axis. By default, the y-axis is logarithmic.",
    )

    return parser.parse_args()





def main():
    args = parse_args()

    log_path = DEFAULT_TEST_LOG_PATH if args.test_log else args.log_path

    output_path, selected_epochs, stride, used_synthetic_epoch_zero = plot_mse_curves(
        log_path = log_path,
        split = args.split,
        divisions = args.divisions,
        output_path = args.output_path,
        use_log_y = not args.linear_y,
    )

    comparison_output_path, comparison_epochs, comparison_used_synthetic_epoch_zero = plot_epoch_zero_vs_final(
        log_path = log_path,
        split = args.split,
        output_path = args.comparison_output_path,
        use_log_y = not args.linear_y,
    )

    print(f"Saved scatter plot: {output_path}")
    print(f"Plotted epochs: {selected_epochs}")
    print(f"Epoch stride: {stride}")
    print(f"Saved epoch-0-vs-final plot: {comparison_output_path}")
    print(f"Comparison epochs: {comparison_epochs}")

    if used_synthetic_epoch_zero or comparison_used_synthetic_epoch_zero:
        print("Note: epoch 0 was not present in the log, so the earliest logged epoch was copied and relabeled as epoch 0.")





if __name__ == "__main__":
    main()
