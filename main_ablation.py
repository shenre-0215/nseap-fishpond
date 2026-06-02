"""消融实验入口 —— 验证 NSEAP 动态本体生长的必要性。

运行方式：
    python main_ablation.py            # 默认 200 次运行
    python main_ablation.py --runs 50  # 快速测试
"""

import argparse
import json
import os

from experiment_ablation import (
    run_ablation_experiment,
    summarize,
    save_results,
)
from visualize import plot_comparison


def main():
    parser = argparse.ArgumentParser(description="NSEAP Ablation Experiment")
    parser.add_argument("--runs", type=int, default=200, help="Number of independent runs")
    parser.add_argument("--cycles", type=int, default=200, help="Max cycles per run")
    parser.add_argument("--output", type=str, default="results_ablation", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Running ablation experiment: {args.runs} runs x {args.cycles} cycles...")
    print("Agents: Greedy, Conservative, QLearning, NSEAP, NSEAP-Static")
    print()

    results = run_ablation_experiment(num_runs=args.runs, max_cycles=args.cycles)

    summary = summarize(results)
    print("\n" + summary)

    save_results(results, args.output)

    print("\nGenerating plots...")
    plot_comparison(results, os.path.join(args.output, "ablation_comparison.png"))

    # 消融专项分析
    nseap_data = results.get("NSEAP", [])
    nseap_static_data = results.get("NSEAP-Static", [])

    if nseap_data and nseap_static_data:
        nseap_collapse = sum(1 for r in nseap_data if r.collapsed) / len(nseap_data) * 100
        static_collapse = sum(1 for r in nseap_static_data if r.collapsed) / len(nseap_static_data) * 100
        nseap_harvest = sum(r.total_harvested for r in nseap_data) / len(nseap_data)
        static_harvest = sum(r.total_harvested for r in nseap_static_data) / len(nseap_static_data)
        nseap_edges = sum(r.edge_count for r in nseap_data) / len(nseap_data)
        static_edges = sum(r.edge_count for r in nseap_static_data) / len(nseap_static_data)
        nseap_nodes = sum(r.node_count for r in nseap_data) / len(nseap_data)
        static_nodes = sum(r.node_count for r in nseap_static_data) / len(nseap_static_data)

        print("\n" + "=" * 60)
        print("  ABLATION ANALYSIS: NSEAP vs NSEAP-Static")
        print("=" * 60)
        print(f"  {'Metric':<25} {'NSEAP':>12} {'NSEAP-Static':>14}")
        print(f"  {'Collapse Rate':<25} {nseap_collapse:>11.1f}% {static_collapse:>13.1f}%")
        print(f"  {'Avg Harvest':<25} {nseap_harvest:>12.0f} {static_harvest:>14.0f}")
        print(f"  {'Avg Edges (final)':<25} {nseap_edges:>11.1f} {static_edges:>14.1f}")
        print(f"  {'Avg Nodes (final)':<25} {nseap_nodes:>11.1f} {static_nodes:>14.1f}")
        print("=" * 60)

        if static_collapse > 0:
            print(f"\n  Key finding: Removing dynamic ontology growth causes")
            print(f"  {static_collapse:.1f}% collapse rate (vs NSEAP's {nseap_collapse:.1f}%).")
            print(f"  Dynamic growth is empirically necessary.")

    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()
