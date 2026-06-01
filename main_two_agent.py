"""两智能体公共资源博弈 — NSEAP 对比实验。

运行方式：
    python main_two_agent.py            # 默认 100 次运行
    python main_two_agent.py --runs 50  # 快速测试
"""

import argparse
import json
import os
from experiment_two_agent import run_experiment_two_agent, summarize_two_agent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_two_agent(results, output_path: str = "two_agent_comparison.png"):
    """绘制两智能体结果对比。"""
    combos = list(results.keys())
    names = [f"{c[0]}\nvs\n{c[1]}" for c in combos]
    avg_harvest = [sum(r.total_harvest for r in results[c]) / len(results[c]) for c in combos]
    collapse_rates = [sum(1 for r in results[c] if r.collapsed) / len(results[c]) * 100 for c in combos]
    avg_cycles = [sum(r.cycles_survived for r in results[c]) / len(results[c]) for c in combos]

    colors = ["#E74C3C", "#3498DB", "#F39C12", "#27AE60"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：平均总收获
    bars = ax1.bar(names, avg_harvest, color=colors, alpha=0.7, edgecolor="white")
    ax1.set_ylabel("Average Total Harvest over All Runs")
    ax1.set_title("Total Harvest (higher = better)")
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 右图：崩溃率
    bars = ax2.bar(names, collapse_rates, color=colors, alpha=0.7, edgecolor="white")
    ax2.set_ylabel("Collapse Rate % (lower = better)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Collapse Percentage")
    ax2.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="NSEAP Two-Agent Common Pool Experiment")
    parser.add_argument("--runs", type=int, default=100, help="Number of independent runs")
    parser.add_argument("--cycles", type=int, default=200, help="Max cycles per run")
    parser.add_argument("--output", type=str, default="results_two_agent", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Running two-agent experiment: {args.runs} runs x {args.cycles} cycles...")
    results = run_experiment_two_agent(num_runs=args.runs, max_cycles=args.cycles)

    summary = summarize_two_agent(results)
    print("\n" + summary)

    # 保存
    summary_data = {}
    for combo, runs in results.items():
        summary_data[f"{combo[0]}_vs_{combo[1]}"] = {
            "avg_cycles": sum(r.cycles_survived for r in runs) / len(runs),
            "avg_total_harvest": sum(r.total_harvest for r in runs) / len(runs),
            "collapse_rate": sum(1 for r in runs if r.collapsed) / len(runs) * 100,
        }
    with open(os.path.join(args.output, "summary_two_agent.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 绘图
    plot_two_agent(results, os.path.join(args.output, "two_agent_comparison.png"))

    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()
