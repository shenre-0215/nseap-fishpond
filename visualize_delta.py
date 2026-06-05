"""Visualize delta(C) convergence — network maturity over time."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_delta_convergence(
    trace_dir: str = "results_delta",
    output_dir: str = "results_delta",
):
    os.makedirs(output_dir, exist_ok=True)

    # Load traces
    fixed_trace = None
    delta_trace = None
    for fname in os.listdir(trace_dir):
        path = os.path.join(trace_dir, fname)
        if not fname.endswith("_trace.json"):
            continue
        with open(path) as f:
            data = json.load(f)
        if "NSEAP-Fixed" in fname:
            fixed_trace = data
        elif "NSEAP-Delta" in fname:
            delta_trace = data

    if delta_trace is None:
        print("No delta trace found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- Panel 1: delta(C) over time ----
    ax = axes[0, 0]
    d_vals = delta_trace["delta"]
    cycles = list(range(len(d_vals)))
    ax.plot(cycles, d_vals, color="#27AE60", linewidth=1.5)
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.4, label="minimum depth (2)")
    ax.axhline(y=10.0, color="gray", linestyle=":", alpha=0.4, label="max depth (10)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("delta(C)")
    ax.set_title("delta(C) Convergence Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Population + delta(C) dual axis ----
    ax = axes[0, 1]
    pop = delta_trace["population"]
    ax.plot(cycles, d_vals, color="#27AE60", linewidth=1.5, label="delta(C)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("delta(C)", color="#27AE60")
    ax.tick_params(axis="y", labelcolor="#27AE60")
    ax2 = ax.twinx()
    ax2.plot(range(len(pop)), pop, color="#3498DB", linewidth=1, alpha=0.6, label="population")
    ax2.set_ylabel("Population", color="#3498DB")
    ax2.tick_params(axis="y", labelcolor="#3498DB")
    ax2.axhline(y=150, color="gray", linestyle="--", alpha=0.4, label="critical threshold")
    ax.set_title("delta(C) vs Population")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Fixed-depth (5) population trace ----
    ax = axes[1, 0]
    if fixed_trace:
        f_pop = fixed_trace["population"]
        ax.plot(range(len(f_pop)), f_pop, color="#8E44AD", linewidth=1.5)
        ax.axhline(y=150, color="gray", linestyle="--", alpha=0.5, label="critical threshold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Population")
        ax.set_title("NSEAP-Fixed (depth=5): Population Trace")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ---- Panel 4: delta(C) histogram ----
    ax = axes[1, 1]
    ax.hist(d_vals, bins=30, color="#27AE60", alpha=0.7, edgecolor="white")
    avg_d = np.mean(d_vals)
    ax.axvline(x=avg_d, color="red", linestyle="--", linewidth=2,
               label=f"mean = {avg_d:.2f}")
    ax.set_xlabel("delta(C) value")
    ax.set_ylabel("Frequency")
    ax.set_title("delta(C) Distribution Over Entire Run")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "delta_convergence.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ---- Additional: delta(C) stats ----
    print(f"\ndelta(C) statistics:")
    print(f"  Mean:      {np.mean(d_vals):.3f}")
    print(f"  Std:       {np.std(d_vals):.3f}")
    print(f"  Min:       {np.min(d_vals):.3f}")
    print(f"  Max:       {np.max(d_vals):.3f}")
    print(f"  First 10%: {np.mean(d_vals[:max(1, len(d_vals)//10)]):.3f}  (early)")
    print(f"  Last  10%: {np.mean(d_vals[-max(1, len(d_vals)//10):]):.3f}  (late)")
    print(f"  Final:     {d_vals[-1]:.3f}")


def plot_delta_vs_edges(
    trace_dir: str = "results_delta",
    output_dir: str = "results_delta",
):
    """Multi-run analysis: how delta(C) evolves as edges accumulate."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all delta traces
    all_delta_runs = []
    for fname in os.listdir(trace_dir):
        if "NSEAP-Delta" in fname and fname.endswith("_trace.json"):
            with open(os.path.join(trace_dir, fname)) as f:
                all_delta_runs.append(json.load(f))

    if len(all_delta_runs) < 2:
        print("Need multiple runs for multi-run analysis.")
        return

    # Load summary for edge counts
    summary_path = os.path.join(trace_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Panel 1: Overlay multiple delta(C) traces ----
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, min(10, len(all_delta_runs))))
    for i, run in enumerate(all_delta_runs[:10]):
        d = run["delta"]
        ax.plot(range(len(d)), d, color=colors[i], linewidth=0.8, alpha=0.7)
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("delta(C)")
    ax.set_title(f"delta(C) Convergence ({min(10, len(all_delta_runs))} Sample Runs)")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Avg delta(C) by edge count ----
    ax = axes[1]
    # For each run, compute final delta and final edge count
    final_deltas = [run["delta"][-1] for run in all_delta_runs]
    final_edges = [run["edges"] for run in all_delta_runs]
    ax.scatter(final_edges, final_deltas, color="#27AE60", alpha=0.5, s=30)
    ax.set_xlabel("Final Edge Count")
    ax.set_ylabel("Final delta(C)")
    ax.set_title("Final delta(C) vs Small Web Size")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "delta_vs_edges.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_delta_convergence()
    plot_delta_vs_edges()
