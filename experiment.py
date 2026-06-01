"""实验运行器 — 对比 4 个智能体在相同环境中的表现。"""

import time
from dataclasses import dataclass, field

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
    total_harvested: int
    collapsed: bool
    population_trace: list[int] = field(default_factory=list)
    harvest_trace: list[int] = field(default_factory=list)
    edge_count: int = 0  # NSEAP 特有：小网边数


def run_single(agent, seed: int, max_cycles: int = 200) -> RunResult:
    pond = FishPond(seed=seed)
    agent_instance = agent()  # 每个 run 创建新的 agent 实例

    pop_trace = []
    harvest_trace = []

    for _ in range(max_cycles):
        obs = pond.step(0)  # 先感知当前状态
        if obs.collapsed:
            break

        # 注意：先感知（step(0)），再决策，再执行
        # 重新设计：直接获取当前状态
        pass

    return RunResult(
        agent_name=agent_instance.name,
        seed=seed,
        cycles_survived=pond.cycle,
        total_harvested=sum(h.harvested for h in pond.history),
        collapsed=pond.collapsed,
        population_trace=[h.population for h in pond.history],
        harvest_trace=[h.harvested for h in pond.history],
    )


def run_single_agent(agent, seed: int, max_cycles: int = 200) -> RunResult:
    """运行单个智能体一个完整 episode。

    流程：先观察当前状态 → 决策 → 执行捕捞 → 循环。
    注意：第一个周期的 step(0) 只让鱼自然增长，不捕捞。
    """
    pond = FishPond(seed=seed)

    pop_trace = [int(pond.population)]
    harvest_trace = [0]

    for _ in range(max_cycles):
        # 1. 观察当前状态
        current_pop = int(pond.population)
        dummy_obs = pond.history[-1] if pond.history else None

        # 2. 智能体决策
        if dummy_obs is None:
            # 第一个周期
            from environment import Observation
            dummy_obs = Observation(population=current_pop, harvested=0, collapsed=False, cycle=0)

        harvest = agent.decide(dummy_obs)

        # 3. 执行
        obs = pond.step(harvest)
        pop_trace.append(obs.population)
        harvest_trace.append(obs.harvested)

        if obs.collapsed:
            break

    edge_count = 0
    if hasattr(agent, 'small_web'):
        edge_count = len(agent.small_web.edges)

    return RunResult(
        agent_name=agent.name,
        seed=seed,
        cycles_survived=pond.cycle,
        total_harvested=sum(harvest_trace),
        collapsed=pond.collapsed,
        population_trace=pop_trace,
        harvest_trace=harvest_trace,
        edge_count=edge_count,
    )


def run_experiment(
    num_runs: int = 200,
    max_cycles: int = 200,
    verbose: bool = True,
) -> dict[str, list[RunResult]]:
    """运行完整对比实验。

    每个 agent 在相同的 num_runs 个随机种子下独立运行。
    """
    agents = [
        GreedyAgent,
        ConservativeAgent,
        QLearningAgent,
        NSEAPAgent,
    ]

    all_results: dict[str, list[RunResult]] = {a().name: [] for a in agents}

    for run_idx in range(num_runs):
        seed = run_idx * 100 + 42  # 可复现的种子
        if verbose and run_idx % 50 == 0:
            print(f"  Run {run_idx}/{num_runs}...")

        for agent_cls in agents:
            agent = agent_cls()
            result = run_single_agent(agent, seed, max_cycles)
            all_results[agent.name].append(result)

    return all_results


def summarize(results: dict[str, list[RunResult]]) -> str:
    """生成实验结果摘要。"""
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Agent':<16} {'Survival':>8}  {'Harvest':>10}  {'Collapse%':>10}  {'Edges':>6}")
    lines.append("-" * 70)

    for name, runs in results.items():
        survivals = [r.cycles_survived for r in runs]
        harvests = [r.total_harvested for r in runs]
        collapses = sum(1 for r in runs if r.collapsed)
        avg_edges = sum(r.edge_count for r in runs) / len(runs) if runs else 0

        lines.append(
            f"{name:<16} {sum(survivals)/len(survivals):>7.1f}  "
            f"{sum(harvests)/len(harvests):>9.0f}  "
            f"{collapses/len(runs)*100:>9.1f}%  "
            f"{avg_edges:>5.1f}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)
