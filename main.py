"""鱼塘生态系统管理 — NSEAP 原型验证。

运行方式：
    python main.py            # 默认 200 次运行
    python main.py --runs 50  # 快速测试
"""

import argparse
import json
import os
from experiment import run_experiment, summarize
from visualize import plot_comparison, plot_nseap_internals


def main():
    parser = argparse.ArgumentParser(description="NSEAP Fish Pond Experiment")
    parser.add_argument("--runs", type=int, default=200, help="Number of independent runs")
    parser.add_argument("--cycles", type=int, default=200, help="Max cycles per run")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Running experiment: {args.runs} runs x {args.cycles} cycles...")
    results = run_experiment(num_runs=args.runs, max_cycles=args.cycles)

    # 打印摘要
    summary = summarize(results)
    print("\n" + summary)

    # 保存详细结果
    summary_data = {}
    for name, runs in results.items():
        summary_data[name] = {
            "avg_survival": sum(r.cycles_survived for r in runs) / len(runs),
            "avg_harvest": sum(r.total_harvested for r in runs) / len(runs),
            "collapse_rate": sum(1 for r in runs if r.collapsed) / len(runs) * 100,
            "avg_edges": sum(r.edge_count for r in runs) / len(runs),
        }

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # 绘图
    print("\nGenerating plots...")
    plot_comparison(results, os.path.join(args.output, "comparison.png"))
    plot_nseap_internals(results, os.path.join(args.output, "nseap_internals.png"))

    # 保存一次典型的运行轨迹用于详细分析
    sample_run = {
        name: {
            "population_trace": runs[0].population_trace,
            "harvest_trace": runs[0].harvest_trace,
        }
        for name, runs in results.items()
    }
    with open(os.path.join(args.output, "sample_trace.json"), "w", encoding="utf-8") as f:
        json.dump(sample_run, f, indent=2)

    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()
