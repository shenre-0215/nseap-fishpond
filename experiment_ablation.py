"""消融实验运行器 —— 对比 NSEAP vs NSEAP-Static，验证动态本体生长的必要性。"""

import json
import os
from dataclasses import dataclass, field

from environment import FishPond, Observation
from agents.nseap import NSEAPAgent
from agents.nseap_static import NSEAPStaticAgent
from agents.greedy import GreedyAgent
from agents.conservative import ConservativeAgent
from agents.qlearning import QLearningAgent


@dataclass
class RunResult:
    agent_name: str
    seed: int
    cycles_survived: int
    total_harvested: int
    collapsed: bool
    population_trace: list[int] = field(default_factory=list)
    harvest_trace: list[int] = field(default_factory=list)
    edge_count: int = 0
    node_count: int = 0


def run_single_agent(agent, seed: int, max_cycles: int = 200) -> RunResult:
    pond = FishPond(seed=seed)
    pop_trace = [int(pond.population)]
    harvest_trace = [0]

    for _ in range(max_cycles):
        current_pop = int(pond.population)
        dummy_obs = pond.history[-1] if pond.history else None
        if dummy_obs is None:
            dummy_obs = Observation(
                population=current_pop, harvested=0, collapsed=False, cycle=0
            )
        harvest = agent.decide(dummy_obs)
        obs = pond.step(harvest)
        pop_trace.append(obs.population)
        harvest_trace.append(obs.harvested)
        if obs.collapsed:
            break

    edge_count = len(agent.small_web.edges) if hasattr(agent, 'small_web') else 0
    node_count = len(agent.small_web.nodes) if hasattr(agent, 'small_web') else 0

    return RunResult(
        agent_name=agent.name,
        seed=seed,
        cycles_survived=pond.cycle,
        total_harvested=sum(harvest_trace),
        collapsed=pond.collapsed,
        population_trace=pop_trace,
        harvest_trace=harvest_trace,
        edge_count=edge_count,
        node_count=node_count,
    )


def run_ablation_experiment(
    num_runs: int = 200,
    max_cycles: int = 200,
    verbose: bool = True,
) -> dict[str, list[RunResult]]:
    agents = [
        GreedyAgent,
        ConservativeAgent,
        QLearningAgent,
        NSEAPAgent,
        NSEAPStaticAgent,
    ]
    all_results: dict[str, list[RunResult]] = {a().name: [] for a in agents}

    for run_idx in range(num_runs):
        seed = run_idx * 100 + 42
        if verbose and run_idx % 50 == 0:
            print(f"  Run {run_idx}/{num_runs}...")
        for agent_cls in agents:
            agent = agent_cls()
            result = run_single_agent(agent, seed, max_cycles)
            all_results[agent.name].append(result)

    return all_results


def summarize(results: dict[str, list[RunResult]]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(
        f"{'Agent':<18} {'Survival':>8}  {'Harvest':>10}  "
        f"{'Collapse%':>10}  {'Edges':>7}  {'Nodes':>7}"
    )
    lines.append("-" * 80)

    for name, runs in results.items():
        survivals = [r.cycles_survived for r in runs]
        harvests = [r.total_harvested for r in runs]
        collapses = sum(1 for r in runs if r.collapsed)
        avg_edges = sum(r.edge_count for r in runs) / len(runs) if runs else 0
        avg_nodes = sum(r.node_count for r in runs) / len(runs) if runs else 0

        lines.append(
            f"{name:<18} {sum(survivals)/len(survivals):>7.1f}  "
            f"{sum(harvests)/len(harvests):>9.0f}  "
            f"{collapses/len(runs)*100:>9.1f}%  "
            f"{avg_edges:>6.1f}  {avg_nodes:>6.1f}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def save_results(results: dict[str, list[RunResult]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    summary_data = {}
    for name, runs in results.items():
        summary_data[name] = {
            "avg_survival": sum(r.cycles_survived for r in runs) / len(runs),
            "avg_harvest": sum(r.total_harvested for r in runs) / len(runs),
            "collapse_rate": sum(1 for r in runs if r.collapsed) / len(runs) * 100,
            "avg_edges": sum(r.edge_count for r in runs) / len(runs),
            "avg_nodes": sum(r.node_count for r in runs) / len(runs),
        }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir}/summary.json")
    return summary_data
