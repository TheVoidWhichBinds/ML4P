# plot.py

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from config import (
    K_VALUES,
    LOG_DIR,
    N_EPOCHS,
    OUTPUT_DIR,
    PARITY_PLOT_MAX_POINTS,
    TEST_PARITY_PATH,
)








#================================================================================================================================================
# PATHS
#================================================================================================================================================

PLOT_DIR = OUTPUT_DIR / "plots"
DEFAULT_LOG_PATH = LOG_DIR / "inner_train_log.csv"
DEFAULT_TEST_RUN_LOG_PATH = LOG_DIR / "inner_test_run_log.csv"
DEFAULT_TEST_LOG_PATH = LOG_DIR / "inner_test_log.csv"








#================================================================================================================================================
# CSV HELPERS
#================================================================================================================================================

def read_csv_rows(csv_path):
    csv_path = Path(csv_path).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV file: {csv_path}")

    with open(csv_path, "r", newline = "") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    return rows, fieldnames





def parse_float(value):
    if value is None or value == "":
        return float("nan")

    return float(value)





def parse_k_values_from_log_columns(columns):
    #------------------------------------------------------------------------------------------------------
    # Extract k values from columns named vrmse_k_10, vrmse_k_25, ... .
    #------------------------------------------------------------------------------------------------------

    parsed_k_values = []

    for column in columns:
        if not column.startswith("vrmse_k_"):
            continue

        k_string = column.replace("vrmse_k_", "")

        try:
            parsed_k_values.append(int(k_string))
        except ValueError:
            continue

    parsed_k_values = sorted(parsed_k_values)

    return parsed_k_values





def load_log_rows(log_path, split):
    #------------------------------------------------------------------------------------------------------
    # Load the CSV log and keep only the requested split.
    #------------------------------------------------------------------------------------------------------

    rows, fieldnames = read_csv_rows(log_path)

    required_columns = {"epoch", "split"}
    missing_columns = required_columns.difference(fieldnames)

    if missing_columns:
        raise ValueError(f"Log file is missing required columns: {sorted(missing_columns)}")

    split_rows = [
        row
        for row in rows
        if row.get("split") == split
    ]

    if len(split_rows) == 0:
        available_splits = sorted(set(row.get("split", "") for row in rows))
        raise ValueError(f"Split '{split}' was not found. Available splits: {available_splits}")

    for row in split_rows:
        row["epoch"] = int(float(row["epoch"]))

    split_rows = sorted(
        split_rows,
        key = lambda row: row["epoch"],
    )

    return split_rows, fieldnames





def choose_evenly_spaced_epochs(epochs, n_epochs):
    #------------------------------------------------------------------------------------------------------
    # Choose n evenly spaced epochs from the epochs present in the saved log.
    # The final epoch is always included.
    #------------------------------------------------------------------------------------------------------

    if n_epochs <= 0:
        raise ValueError("n_epochs must be a positive integer.")

    epochs = sorted(set(int(epoch) for epoch in epochs))

    if len(epochs) == 0:
        raise ValueError("No epochs were found in the log file.")

    if n_epochs >= len(epochs):
        return epochs

    if n_epochs == 1:
        return [epochs[-1]]

    selected_indices = []
    last_index = len(epochs) - 1

    for selection_index in range(n_epochs):
        fractional_index = selection_index * last_index / (n_epochs - 1)
        selected_indices.append(round(fractional_index))

    selected_indices = sorted(set(selected_indices))

    while len(selected_indices) < n_epochs:
        for candidate_index in range(len(epochs)):
            if candidate_index not in selected_indices:
                selected_indices.append(candidate_index)
                break

        selected_indices = sorted(set(selected_indices))

    selected_epochs = [epochs[index] for index in selected_indices]

    if epochs[-1] not in selected_epochs:
        selected_epochs[-1] = epochs[-1]

    selected_epochs = sorted(set(selected_epochs))

    return selected_epochs





def opacity_for_index(index, count, minimum_opacity = 0.25, maximum_opacity = 1.0):
    #------------------------------------------------------------------------------------------------------
    # Make later epoch curves more opaque.
    #------------------------------------------------------------------------------------------------------

    if count <= 1:
        return maximum_opacity

    fraction = index / (count - 1)
    opacity = minimum_opacity + fraction * (maximum_opacity - minimum_opacity)

    return opacity





def downsample_xy(x_values, y_values, max_points):
    if max_points is None or max_points <= 0:
        return x_values, y_values

    if len(x_values) <= max_points:
        return x_values, y_values

    stride = max(1, len(x_values) // max_points)

    return x_values[::stride][:max_points], y_values[::stride][:max_points]








#================================================================================================================================================
# VRMSE PLOTTING
#================================================================================================================================================

def plot_vrmse_curves(
    log_path,
    split,
    n_epochs,
    output_path,
    use_log_y,
):
    #------------------------------------------------------------------------------------------------------
    # Plot VRMSE(k) curves for n evenly spaced epochs from the saved log.
    #------------------------------------------------------------------------------------------------------

    split_rows, fieldnames = load_log_rows(
        log_path = log_path,
        split = split,
    )

    k_values = parse_k_values_from_log_columns(fieldnames)

    if len(k_values) == 0:
        k_values = list(K_VALUES)

    missing_columns = [
        f"vrmse_k_{k}"
        for k in k_values
        if f"vrmse_k_{k}" not in fieldnames
    ]

    if missing_columns:
        raise ValueError(f"Log file is missing VRMSE columns: {missing_columns}")

    selected_epochs = choose_evenly_spaced_epochs(
        epochs = [row["epoch"] for row in split_rows],
        n_epochs = n_epochs,
    )

    selected_rows = [
        row
        for row in split_rows
        if row["epoch"] in selected_epochs
    ]

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (8.0, 5.5))

    for index, row in enumerate(selected_rows):
        epoch = int(row["epoch"])
        vrmse_values = [parse_float(row[f"vrmse_k_{k}"]) for k in k_values]
        alpha = opacity_for_index(
            index = index,
            count = len(selected_rows),
        )

        plt.scatter(
            k_values,
            vrmse_values,
            alpha = alpha,
            label = f"epoch {epoch}",
        )

    plt.xlabel("k")
    plt.ylabel("VRMSE")
    plt.title(f"VRMSE vs. temporal history length k ({split})")

    if use_log_y:
        plt.yscale("log")

    plt.xticks(k_values)
    plt.grid(True, alpha = 0.25)
    plt.legend(title = f"{len(selected_epochs)} epochs")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()

    return output_path, selected_epochs





def plot_epoch_first_vs_final(
    log_path,
    split,
    output_path,
    use_log_y,
):
    #------------------------------------------------------------------------------------------------------
    # Plot the first saved epoch and the final saved epoch, with lines connecting the k values within each
    # epoch.
    #------------------------------------------------------------------------------------------------------

    split_rows, fieldnames = load_log_rows(
        log_path = log_path,
        split = split,
    )

    k_values = parse_k_values_from_log_columns(fieldnames)

    if len(k_values) == 0:
        k_values = list(K_VALUES)

    first_epoch = min(row["epoch"] for row in split_rows)
    final_epoch = max(row["epoch"] for row in split_rows)
    comparison_epochs = sorted(set([first_epoch, final_epoch]))

    comparison_rows = [
        row
        for row in split_rows
        if row["epoch"] in comparison_epochs
    ]

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (8.0, 5.5))

    for row in comparison_rows:
        epoch = int(row["epoch"])
        vrmse_values = [parse_float(row[f"vrmse_k_{k}"]) for k in k_values]

        plt.plot(
            k_values,
            vrmse_values,
            marker = "o",
            linewidth = 1.5,
            label = f"epoch {epoch}",
        )

    plt.xlabel("k")
    plt.ylabel("VRMSE")
    plt.title(f"First saved epoch vs. final epoch VRMSE ({split})")

    if use_log_y:
        plt.yscale("log")

    plt.xticks(k_values)
    plt.grid(True, alpha = 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()

    return output_path, comparison_epochs








#================================================================================================================================================
# PARITY PLOTTING
#================================================================================================================================================

def load_parity_points(
    parity_path,
    k_value = None,
    channel = None,
):
    rows, fieldnames = read_csv_rows(parity_path)

    required_columns = {"k", "channel", "target", "prediction"}
    missing_columns = required_columns.difference(fieldnames)

    if missing_columns:
        raise ValueError(f"Parity file is missing required columns: {sorted(missing_columns)}")

    targets = []
    predictions = []

    for row in rows:
        row_k = int(float(row["k"]))
        row_channel = int(float(row["channel"]))

        if k_value is not None and row_k != int(k_value):
            continue

        if channel is not None and row_channel != int(channel):
            continue

        targets.append(parse_float(row["target"]))
        predictions.append(parse_float(row["prediction"]))

    if len(targets) == 0:
        raise ValueError(
            "No parity points matched the requested filters. "
            f"k_value = {k_value}, channel = {channel}."
        )

    return targets, predictions





def plot_parity(
    parity_path,
    output_path,
    k_value = None,
    channel = None,
    max_points = PARITY_PLOT_MAX_POINTS,
):
    #------------------------------------------------------------------------------------------------------
    # Plot prediction vs. target from the parity CSV produced by test.py.
    #------------------------------------------------------------------------------------------------------

    targets, predictions = load_parity_points(
        parity_path = parity_path,
        k_value = k_value,
        channel = channel,
    )

    targets, predictions = downsample_xy(
        x_values = targets,
        y_values = predictions,
        max_points = max_points,
    )

    min_axis_value = min(min(targets), min(predictions))
    max_axis_value = max(max(targets), max(predictions))

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (6.5, 6.5))
    plt.scatter(
        targets,
        predictions,
        s = 8,
        alpha = 0.35,
    )
    plt.plot(
        [min_axis_value, max_axis_value],
        [min_axis_value, max_axis_value],
        linewidth = 1.5,
    )

    title_parts = ["Parity plot"]

    if k_value is not None:
        title_parts.append(f"k = {int(k_value)}")

    if channel is not None:
        title_parts.append(f"channel = {int(channel)}")

    plt.xlabel("Target")
    plt.ylabel("Prediction")
    plt.title(" | ".join(title_parts))
    plt.grid(True, alpha = 0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()

    return output_path, len(targets)








#================================================================================================================================================
# COMMAND LINE INTERFACE
#================================================================================================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-path",
        type = str,
        default = str(DEFAULT_LOG_PATH),
        help = "Path to run.py CSV log file.",
    )

    parser.add_argument(
        "--test-run-log",
        action = "store_true",
        help = "Use outputs/logs/inner_test_run_log.csv instead of inner_train_log.csv.",
    )

    parser.add_argument(
        "--test-log",
        action = "store_true",
        help = "Use outputs/logs/inner_test_log.csv from test.py instead of inner_train_log.csv.",
    )

    parser.add_argument(
        "--split",
        type = str,
        default = "valid",
        choices = ["train", "valid", "test"],
        help = "Which split to plot from the CSV log.",
    )

    parser.add_argument(
        "--n-epochs",
        type = int,
        default = N_EPOCHS,
        help = "Number of evenly spaced saved epochs to plot. The final saved epoch is always included.",
    )

    parser.add_argument(
        "--output-path",
        type = str,
        default = str(PLOT_DIR / "vrmse_vs_k_by_epoch.png"),
        help = "Where to save the scatter-by-epoch plot.",
    )

    parser.add_argument(
        "--comparison-output-path",
        type = str,
        default = str(PLOT_DIR / "vrmse_vs_k_first_vs_final.png"),
        help = "Where to save the first-saved-epoch-versus-final connected-line plot.",
    )

    parser.add_argument(
        "--parity-path",
        type = str,
        default = str(TEST_PARITY_PATH),
        help = "Path to the parity CSV produced by test.py.",
    )

    parser.add_argument(
        "--parity-output-path",
        type = str,
        default = str(PLOT_DIR / "parity_plot.png"),
        help = "Where to save the parity plot.",
    )

    parser.add_argument(
        "--parity-k",
        type = int,
        default = None,
        help = "Optional k value to isolate in the parity plot.",
    )

    parser.add_argument(
        "--parity-channel",
        type = int,
        default = None,
        help = "Optional physical channel index to isolate in the parity plot.",
    )

    parser.add_argument(
        "--max-parity-points",
        type = int,
        default = PARITY_PLOT_MAX_POINTS,
        help = "Maximum number of points shown on the parity plot.",
    )

    parser.add_argument(
        "--only-parity",
        action = "store_true",
        help = "Only make the parity plot.",
    )

    parser.add_argument(
        "--no-parity",
        action = "store_true",
        help = "Do not make the parity plot, even if the parity CSV exists.",
    )

    parser.add_argument(
        "--linear-y",
        action = "store_true",
        help = "Use a linear y-axis. By default, the y-axis is logarithmic.",
    )

    return parser.parse_args()





def main():
    args = parse_args()

    if args.test_log:
        log_path = DEFAULT_TEST_LOG_PATH
    elif args.test_run_log:
        log_path = DEFAULT_TEST_RUN_LOG_PATH
    else:
        log_path = args.log_path

    split = args.split

    if args.test_log and args.split == "valid":
        split = "test"

    if not args.only_parity:
        output_path, selected_epochs = plot_vrmse_curves(
            log_path = log_path,
            split = split,
            n_epochs = args.n_epochs,
            output_path = args.output_path,
            use_log_y = not args.linear_y,
        )

        comparison_output_path, comparison_epochs = plot_epoch_first_vs_final(
            log_path = log_path,
            split = split,
            output_path = args.comparison_output_path,
            use_log_y = not args.linear_y,
        )

        print(f"Saved VRMSE plot: {output_path}")
        print(f"Plotted epochs: {selected_epochs}")
        print(f"Saved first-vs-final plot: {comparison_output_path}")
        print(f"Comparison epochs: {comparison_epochs}")

    parity_path = Path(args.parity_path).expanduser().resolve()

    if not args.no_parity:
        if parity_path.exists():
            parity_output_path, num_points = plot_parity(
                parity_path = parity_path,
                output_path = args.parity_output_path,
                k_value = args.parity_k,
                channel = args.parity_channel,
                max_points = args.max_parity_points,
            )

            print(f"Saved parity plot: {parity_output_path}")
            print(f"Parity points plotted: {num_points}")
        elif args.only_parity:
            raise FileNotFoundError(f"Could not find parity CSV: {parity_path}")
        else:
            print(f"Parity CSV not found, so parity plot was skipped: {parity_path}")





if __name__ == "__main__":
    main()
