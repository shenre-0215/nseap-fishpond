"""两智能体公共资源博弈实验 — 对比不同组合。

四种组合：
1. Greedy vs Greedy
2. Conservative vs Conservative
3. QLearning vs QLearning
4. NSEAP vs NSEAP
"""

import random
from dataclasses import dataclass, field

from two_agent_env import TwoAgentFishPond, TwoAgentObservation
from agents.greedy import GreedyAgent
from agents.conservative import ConservativeAgent
from agents.qlearning import QLearningAgent
from agents.nseap import NSEAPAgent


@dataclass
class TwoAgentRunResult:
    agent1_name: str
    agent2_name: str
    seed: int
    cycles_survived: int
    total_harvest1: int
    total_harvest2: int
    total_harvest: int
    collapsed: bool


def run_two_agent(
    agent1_cls,
    agent2_cls,
    seed: int,
    max_cycles: int = 200,
) -> TwoAgentRunResult:
    pond = TwoAgentFishPond(
        total_carrying_capacity=2000,
        growth_rate=0.3,
        critical_threshold=300,
        noise=0.05,
        seed=seed,
    )
    a1 = agent1_cls()
    a2 = agent2_cls()
    total1 = 0
    total2 = 0

    for _ in range(max_cycles):
        # 初始观测
        if pond.cycle == 0:
            obs1 = TwoAgentObservation(
                my_population=int(pond.population),
                my_last_harvest=0,
                collapsed=False,
                cycle=0,
            )
            obs2 = TwoAgentObservation(
                my_population=int(pond.population),
                my_last_harvest=0,
                collapsed=False,
                cycle=0,
            )

        # 决策
        h1 = a1.decide(obs1) if not pond.collapsed else 0
        h2 = a2.decide(obs2) if not pond.collapsed else 0

        # 执行
        obs1, obs2 = pond.step(h1, h2)

        total1 += h1
        total2 += h2

        if pond.collapsed:
            break

    return TwoAgentRunResult(
        agent1_name=a1.name,
        agent2_name=a2.name,
        seed=seed,
        cycles_survived=pond.cycle,
        total_harvest1=total1,
        total_harvest2=total2,
        total_harvest=total1 + total2,
        collapsed=pond.collapsed,
    )


def run_experiment_two_agent(
    num_runs: int = 100,
    max_cycles: int = 200,
) -> dict[tuple[str, str], list[TwoAgentRunResult]]:

    combinations = [
        (GreedyAgent, GreedyAgent),
        (ConservativeAgent, ConservativeAgent),
        (QLearningAgent, QLearningAgent),
        (NSEAPAgent, NSEAPAgent),
    ]

    all_results: dict[tuple[str, str], list[TwoAgentRunResult]] = {}

    for (a1_cls, a2_cls) in combinations:
        combo = (a1_cls().name, a2_cls().name)
        all_results[combo] = []
        print(f"Running {combo[0]} vs {combo[1]}...")

        for run_idx in range(num_runs):
            seed = run_idx * 100 + 42
            result = run_two_agent(a1_cls, a2_cls, seed, max_cycles)
            all_results[combo].append(result)

    return all_results


def summarize_two_agent(results: dict[tuple[str, str], list[TwoAgentRunResult]]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(
        f"{'Combination':<25} {'Avg Cycles':>12} {'Avg Harvest':>12} {'Collapse%':>10}"
    )
    lines.append("-" * 80)

    for combo, runs in results.items():
        name = f"{combo[0]} vs {combo[1]}"
        avg_cycles = sum(r.cycles_survived for r in runs) / len(runs)
        avg_harvest = sum(r.total_harvest for r in runs) / len(runs)
        collapse_pct = sum(1 for r in runs if r.collapsed) / len(runs) * 100
        lines.append(
            f"{name:<25} {avg_cycles:>11.1f}  {avg_harvest:>11.0f}  {collapse_pct:>8.1f}%"
        )

    lines.append("=" * 80)
    return "\n".join(lines)
