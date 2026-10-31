
import csv
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = INPUT_DIR / "figures"
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
MARKERS = ("o", "s", "^", "D", "v")
METRICS = ("train_loss", "dev_loss", "train_acc", "dev_acc", "train_seconds")


def read_history(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"epoch", *METRICS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        records = []
        for row in reader:
            record = {"epoch": int(row["epoch"])}
            for metric in METRICS:
                value = float(row[metric])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{path.name}: invalid {metric}: {value}")
                if metric.endswith("_acc") and value > 1:
                    raise ValueError(f"{path.name}: accuracy must be between 0 and 1")
                record[metric] = value
            records.append(record)

    if not records:
        raise ValueError(f"{path.name}: no epoch records")
    if [record["epoch"] for record in records] != list(range(1, len(records) + 1)):
        raise ValueError(f"{path.name}: expected consecutive epochs starting at 1")
    return records


def read_histories():
    candidates = {}
    pattern = re.compile(r"metrics_batch_(\d+)(?:_epochs_(\d+))?\.csv")

    for path in sorted(INPUT_DIR.glob("metrics*.csv")):
        match = pattern.fullmatch(path.name)
        if path.name == "metrics_baseline.csv":
            batch_size = 1
        elif match:
            batch_size = int(match.group(1))
            if batch_size < 1:
                raise ValueError(f"{path.name}: batch size must be positive")
        else:
            continue

        records = read_history(path)
        if match and match.group(2) and len(records) != int(match.group(2)):
            raise ValueError(f"{path.name}: recorded epochs do not match the filename")

        # Prefer an explicit batch run over the old baseline when lengths match.
        rank = (len(records), path.name != "metrics_baseline.csv")
        if batch_size not in candidates or rank > candidates[batch_size][0]:
            candidates[batch_size] = (rank, path, records)

    if not candidates:
        raise ValueError(f"No metric histories found in {INPUT_DIR}")

    histories = {}
    for batch_size, (_, path, records) in sorted(candidates.items()):
        histories[batch_size] = records
        print(f"Batch size {batch_size}: {path.name} ({len(records)} epochs)")
    return histories


def format_axes(ax, histories, ylabel):
    epochs = sorted({row["epoch"] for records in histories.values() for row in records})
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if len(epochs) > 1:
        ax.set_xlim(epochs[0], epochs[-1])
    else:
        ax.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
    if len(epochs) <= 15:
        ax.set_xticks(epochs)
    ax.grid(alpha=0.2)


def save_plot(fig, legend_ax, title, filename):
    handles, labels = legend_ax.get_legend_handles_labels()
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=12, y=0.99)
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.99 - 0.22 / height),
               ncol=min(4, len(handles)), frameon=False, borderaxespad=0)
    fig.tight_layout()
    legend_rows = math.ceil(len(handles) / 4)
    fig.subplots_adjust(top=1 - (0.75 + 0.2 * (legend_rows - 1)) / height)
    for extension in ("pdf", "png"):
        path = OUTPUT_DIR / f"{filename}.{extension}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def plot_losses(histories):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i, (batch_size, records) in enumerate(histories.items()):
        epochs = [row["epoch"] for row in records]
        for split, linestyle in (("train", "-"), ("dev", "--")):
            values = [row[f"{split}_loss"] for row in records]
            ax.plot(epochs, values, label=f"Batch {batch_size} | {split.title()}",
                    color=COLORS[i % len(COLORS)], linestyle=linestyle,
                    marker=MARKERS[i % len(MARKERS)], markersize=3)
    format_axes(ax, histories, "Mean negative log-likelihood per sentence")
    save_plot(fig, ax, "Deep averaging network | Training and development loss",
              "loss_comparison")


def plot_dev_accuracy(histories):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for i, (batch_size, records) in enumerate(histories.items()):
        epochs = [row["epoch"] for row in records]
        values = [100 * row["dev_acc"] for row in records]
        ax.plot(epochs, values, label=f"Batch size {batch_size}",
                color=COLORS[i % len(COLORS)], linestyle="--",
                marker=MARKERS[i % len(MARKERS)], markersize=3)
    ax.axhline(77, color="0.4", linestyle="--", linewidth=1)
    ax.text(0.02, 77.15, "77%", transform=ax.get_yaxis_transform(),
            color="0.4", fontsize=9)
    format_axes(ax, histories, "Development accuracy (%)")
    save_plot(fig, ax, "Deep averaging network | Development accuracy",
              "dev_accuracy_comparison")


def write_results_table(histories):
    if 1 not in histories:
        raise ValueError("Batch size 1 is required to calculate training speedups")
    epoch_counts = {len(records) for records in histories.values()}
    if len(epoch_counts) != 1:
        raise ValueError("The results table requires the same epoch count for every run")

    num_epochs = epoch_counts.pop()
    baseline_seconds = sum(row["train_seconds"] for row in histories[1])
    if baseline_seconds <= 0:
        raise ValueError("Training time must be positive to calculate speedups")
    best_accuracy = max(row["dev_acc"] for records in histories.values() for row in records)
    lines = [
        r"% Generated by p2-distrib/plot_history.py from the selected metric CSVs.",
        r"\begin{table}[!ht]",
        r"\centering",
        r"\small",
        rf"\caption{{DAN batching over {num_epochs} epochs (seed 42). Best epoch maximizes "
        r"dev accuracy; final dev uses the last epoch. Training time excludes evaluation. "
        r"Speedups are relative to batch size 1 and use one run per configuration.}",
        r"\label{tab:batch-results}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{rrrrrr}",
        r"\hline",
        r"Batch & Best dev (\%) & Epoch & Final dev (\%) & Train (s) & Speedup \\",
        r"\hline",
    ]
    for batch_size, records in histories.items():
        best = max(records, key=lambda row: row["dev_acc"])
        train_seconds = sum(row["train_seconds"] for row in records)
        if train_seconds <= 0:
            raise ValueError(f"Batch size {batch_size}: training time must be positive")
        best_value = f"{100 * best['dev_acc']:.2f}"
        if best["dev_acc"] == best_accuracy:
            best_value = r"\cellcolor{gray!20}\textbf{" + best_value + "}"
        lines.append(
            f"{batch_size} & {best_value} & {best['epoch']} & "
            f"{100 * records[-1]['dev_acc']:.2f} & {train_seconds:.2f} & "
            f"{baseline_seconds / train_seconds:.2f}" + r"$\times$ \\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    path = OUTPUT_DIR / "batch_results.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    histories = read_histories()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_losses(histories)
    plot_dev_accuracy(histories)
    write_results_table(histories)
