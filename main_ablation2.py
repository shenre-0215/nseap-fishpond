"""第二组消融实验入口 —— 分别移除悬置层和闭环检测。

运行方式：
    python main_ablation2.py            # 默认 200 次运行
    python main_ablation2.py --runs 50  # 快速测试
"""

import argparse
import os

from experiment_ablation2 import (
    run_ablation2_experiment,
    summarize,
    save_results,
)
from visualize import plot_comparison


def main():
    parser = argparse.ArgumentParser(description="NSEAP Component Ablation Experiment 2")
    parser.add_argument("--runs", type=int, default=200, help="Number of independent runs")
    parser.add_argument("--cycles", type=int, default=200, help="Max cycles per run")
    parser.add_argument("--output", type=str, default="results_ablation2", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Running component ablation: {args.runs} runs x {args.cycles} cycles...")
    print("Agents: NSEAP, NSEAP-NoSusp (no suspension), NSEAP-NoClose (no closure)")
    print()

    results = run_ablation2_experiment(num_runs=args.runs, max_cycles=args.cycles)

    summary = summarize(results)
    print("\n" + summary)

    save_results(results, args.output)

    print("\nGenerating plots...")
    plot_comparison(results, os.path.join(args.output, "ablation2_comparison.png"))

    # 组件贡献分析
    nseap_data = results.get("NSEAP", [])
    nosusp_data = results.get("NSEAP-NoSusp", [])
    noclose_data = results.get("NSEAP-NoClose", [])

    if nseap_data:
        nseap_collapse = sum(1 for r in nseap_data if r.collapsed) / len(nseap_data) * 100
        nseap_harvest = sum(r.total_harvested for r in nseap_data) / len(nseap_data)

        print("\n" + "=" * 60)
        print("  COMPONENT CONTRIBUTION ANALYSIS")
        print("=" * 60)

        for name, data, base_collapse, base_harvest in [
            ("NSEAP", nseap_data, nseap_collapse, nseap_harvest),
        ]:
            for ablated_name, ablated_data in [
                ("NSEAP-NoSusp", nosusp_data),
                ("NSEAP-NoClose", noclose_data),
            ]:
                if not ablated_data:
                    continue
                a_collapse = sum(1 for r in ablated_data if r.collapsed) / len(ablated_data) * 100
                a_harvest = sum(r.total_harvested for r in ablated_data) / len(ablated_data)
                delta_collapse = a_collapse - base_collapse
                delta_harvest = a_harvest - base_harvest
                print(f"\n  Removing {ablated_name.split('-')[1]}:")
                print(f"    Collapse: {base_collapse:.1f}% → {a_collapse:.1f}% (Δ = {delta_collapse:+.1f}%)")
                print(f"    Harvest:  {base_harvest:.0f} → {a_harvest:.0f} (Δ = {delta_harvest:+.0f})")

        print("=" * 60)

    print(f"\nAll results saved to: {args.output}/")


if __name__ == "__main__":
    main()
