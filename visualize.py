"""结果可视化 — 生成对比曲线图。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiment import RunResult


def plot_comparison(
    results: dict[str, list[RunResult]],
    output_path: str = "comparison.png",
    sample_run_idx: int = 0,
):
    """绘制多面板对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "Greedy": "#E74C3C",
        "Conservative": "#3498DB",
        "QLearning": "#F39C12",
        "NSEAP": "#27AE60",
    }

    # 面板 1：单次运行的种群数量曲线
    ax = axes[0, 0]
    for name, runs in results.items():
        run = runs[min(sample_run_idx, len(runs) - 1)]
        cycles = list(range(len(run.population_trace)))
        ax.plot(cycles, run.population_trace, color=colors[name], label=name, linewidth=1.5)
    ax.axhline(y=150, color="gray", linestyle="--", alpha=0.5, label="critical threshold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Population")
    ax.set_title(f"Population over Time (Run #{sample_run_idx})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 面板 2：累计捕捞量曲线
    ax = axes[0, 1]
    for name, runs in results.items():
        run = runs[min(sample_run_idx, len(runs) - 1)]
        cumulative = np.cumsum(run.harvest_trace)
        cycles = list(range(len(cumulative)))
        ax.plot(cycles, cumulative, color=colors[name], label=name, linewidth=1.5)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cumulative Harvest")
    ax.set_title("Cumulative Harvest over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 面板 3：存活周期分布（箱线图）
    ax = axes[1, 0]
    survival_data = []
    labels = []
    for name in ["Greedy", "Conservative", "QLearning", "NSEAP"]:
        runs_list = results.get(name, [])
        survival_data.append([r.cycles_survived for r in runs_list])
        labels.append(name)
    bp = ax.boxplot(survival_data, labels=labels, patch_artist=True)
    for patch, name in zip(bp["boxes"], labels):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.6)
    ax.set_ylabel("Cycles Survived")
    ax.set_title("Survival Distribution (across all runs)")
    ax.grid(True, alpha=0.3)

    # 面板 4：崩溃率 vs 平均累计捕捞量（散点图）
    ax = axes[1, 1]
    for name in ["Greedy", "Conservative", "QLearning", "NSEAP"]:
        runs_list = results.get(name, [])
        collapse_rate = sum(1 for r in runs_list if r.collapsed) / len(runs_list) * 100
        avg_harvest = sum(r.total_harvested for r in runs_list) / len(runs_list)
        ax.scatter(collapse_rate, avg_harvest, color=colors[name], s=200, label=name, zorder=5)
        ax.annotate(name, (collapse_rate, avg_harvest), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)
    ax.set_xlabel("Collapse Rate (%)")
    ax.set_ylabel("Avg Total Harvest")
    ax.set_title("Efficiency Frontier: Harvest vs Safety")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_nseap_internals(
    results: dict[str, list[RunResult]],
    output_path: str = "nseap_internals.png",
):
    """绘制 NSEAP 内部状态：小网边数增长、置信度演变。"""
    nseap_runs = results.get("NSEAP", [])
    if not nseap_runs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：小网边数增长
    ax = axes[0]
    # 取存活最长的几次运行
    sorted_runs = sorted(nseap_runs, key=lambda r: r.cycles_survived, reverse=True)
    for run in sorted_runs[:5]:
        ax.plot(range(len(run.population_trace)), run.population_trace,
                alpha=0.7, linewidth=1)
    ax.axhline(y=150, color="gray", linestyle="--", alpha=0.5, label="threshold")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Population")
    ax.set_title("NSEAP: Top 5 Longest Runs")
    ax.grid(True, alpha=0.3)

    # 右：小网边数分布
    ax = axes[1]
    edge_counts = [r.edge_count for r in nseap_runs]
    ax.hist(edge_counts, bins=20, color="#27AE60", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Number of Edges in Small Web")
    ax.set_ylabel("Frequency")
    ax.set_title("NSEAP: Small Web Size Distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
