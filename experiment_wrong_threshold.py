"""实验：初始临界点猜测严重错误，NSEAP能否自我修正？

真实临界点 = 150
NSEAP初始猜测安全边界 = 400（错得离谱）
看NSEAP能否通过校准慢慢把边界降下来
"""

import random
from dataclasses import dataclass
from environment import FishPond
from agents.greedy import GreedyAgent
from agents.conservative import ConservativeAgent
from agents.qlearning import QLearningAgent
from agents.nseap import NSEAPAgent


@dataclass
class RunResult:
    agent_name: str
    seed: int
    cycles_survived: int
    total_harvest: int
    collapsed: bool
    final_boundary: float


def run_single(agent_cls, seed: int, max_cycles: int = 200, init_boundary: float = 400) -> RunResult:
    """Run one experiment with wrong initial threshold guess."""
    # 真实临界点是150，NSEAP初始猜测是400
    pond = FishPond(
        carrying_capacity=1000,
        growth_rate=0.3,
        critical_threshold=150,  # 真实临界点
        noise=0.05,
        seed=seed,
    )

    if agent_cls == NSEAPAgent:
        # NSEAPAgent构造函数里没有直接传init_boundary，我们创建后改
        agent = agent_cls()
        agent.suspension.min_safe_boundary = init_boundary  # 强制设置错误初始猜测
    else:
        agent = agent_cls()

    total_harvest = 0

    for cycle in range(max_cycles):
        # Get observation
        if pond.cycle == 0:
            # First cycle, observation before any action
            obs = pond.step(0)
        harvest = agent.decide(obs) if not obs.collapsed else 0
        obs = pond.step(harvest)
        total_harvest += harvest
        if obs.collapsed:
            break

    final_boundary = agent.suspension.min_safe_boundary if agent_cls == NSEAPAgent else init_boundary

    return RunResult(
        agent_name=agent.name,
        seed=seed,
        cycles_survived=pond.cycle,
        total_harvest=total_harvest,
        collapsed=pond.collapsed,
        final_boundary=final_boundary,
    )


def run_experiment(num_runs: int = 100, max_cycles: int = 200, init_boundary: float = 400):
    """Run experiment with wrong initial threshold guess."""
    results = {
        "NSEAP": [],
        "Conservative": [],
        "QLearning": [],
        "Greedy": [],
    }

    print(f"Running experiment: {num_runs} runs, true threshold = 150, initial guess = {init_boundary}")
    print("-" * 80)

    for i in range(num_runs):
        if (i + 1) % 10 == 0:
            print(f"  Run {i+1}/{num_runs}...")

        # NSEAP with wrong initial guess
        res = run_single(NSEAPAgent, i, max_cycles, init_boundary)
        results["NSEAP"].append(res)

        # Baselines
        results["Greedy"].append(run_single(GreedyAgent, i, max_cycles, init_boundary))
        results["Conservative"].append(run_single(ConservativeAgent, i, max_cycles, init_boundary))
        results["QLearning"].append(run_single(QLearningAgent, i, max_cycles, init_boundary))

    return results


def summarize(results):
    """Print summary."""
    print("\n" + "=" * 80)
    print(f"{'Agent':<15} {'Survival':>10} {'Harvest':>12} {'Collapse%':>10} {'Avg Final Boundary':>20}")
    print("-" * 80)

    for agent_name, runs in results.items():
        avg_cycles = sum(r.cycles_survived for r in runs) / len(runs)
        avg_harvest = sum(r.total_harvest for r in runs) / len(runs)
        collapse_rate = sum(1 for r in runs if r.collapsed) / len(runs) * 100
        avg_boundary = sum(r.final_boundary for r in runs) / len(runs)
        print(f"{agent_name:<15} {avg_cycles:>10.1f} {avg_harvest:>12.0f} {collapse_rate:>9.1f}% {avg_boundary:>20.1f}")

    print("=" * 80)

    return {
        agent_name: {
            "avg_cycles": sum(r.cycles_survived for r in runs) / len(runs),
            "avg_harvest": sum(r.total_harvest for r in runs) / len(runs),
            "collapse_rate": sum(1 for r in runs if r.collapsed) / len(runs) * 100,
            "avg_final_boundary": sum(r.final_boundary for r in runs) / len(runs),
        }
        for agent_name, runs in results.items()
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--init-boundary", type=float, default=400)
    args = parser.parse_args()

    results = run_experiment(args.runs, args.cycles, args.init_boundary)
    summary = summarize(results)
