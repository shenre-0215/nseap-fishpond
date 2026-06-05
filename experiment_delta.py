"""Compare NSEAP with fixed depth vs dynamic delta(C)."""

import argparse
import json
import os
from dataclasses import dataclass, field

from environment import FishPond
from suspension import SuspensionLayer
from small_web import SmallWeb


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
    delta_trace: list[float] = field(default_factory=list)


class NSEAPAgentFixed:
    """Original NSEAP with hardcoded max_depth=5."""

    def __init__(self):
        self.name = "NSEAP-Fixed"
        self.suspension = SuspensionLayer()
        self.small_web = SmallWeb(max_depth=5)
        self.last_prediction = None
        self.total_harvested = 0

    def decide(self, obs) -> int:
        if obs.collapsed:
            return 0
        pop = obs.population

        suspended = self.suspension.suspend(pop)
        reasoning = self.small_web.reason(pop, suspended["candidate_actions"])

        if self.last_prediction:
            self._calibrate(pop)

        chosen = self._choose_action(reasoning, suspended["neutral_attributes"])

        if chosen.get("closure_path") and chosen["can_close"]:
            self.small_web.solidify_path(chosen["closure_path"])

        safe_nodes = [n for n in self.small_web.nodes.values() if n.is_safe]
        if safe_nodes:
            lowest_safe = min(n.lo for n in safe_nodes)
            self.suspension.update_boundary(lowest_safe)

        self.last_prediction = {
            "action": chosen["action"],
            "predicted_to": chosen["predicted_to"],
            "edge_id": chosen["edge_id"],
            "from_population": pop,
        }

        return self._action_to_amount(chosen["action"], pop)

    def _choose_action(self, reasoning, attrs):
        candidates = reasoning["candidates"]
        if not reasoning["current_is_safe"]:
            closable = [c for c in candidates if c["can_close"]]
            if closable:
                return min(closable, key=lambda c: c["risk"])
        if attrs.uncertainty > 0.5:
            for c in candidates:
                if c["action"] in ("none", "light") and c["can_close"]:
                    return c
        safe_candidates = [c for c in candidates if c["risk"] < 0.5]
        if safe_candidates:
            return max(safe_candidates, key=lambda c: c["confidence"])
        return min(candidates, key=lambda c: c["risk"])

    def _calibrate(self, actual_population):
        pred = self.last_prediction
        if not pred:
            return
        actual_node = self.small_web.find_node(float(actual_population))
        predicted_node = self.small_web.nodes.get(pred["predicted_to"])
        success = predicted_node is not None and predicted_node.contains(float(actual_population))
        if not success:
            edge = self.small_web.get_or_create_edge(
                self.small_web.find_node(float(pred["from_population"])).name,
                pred["action"],
                actual_node.name,
            )
            self.small_web.update_edge(edge, True)
            old_edge = self.small_web.edges.get(pred["edge_id"])
            if old_edge:
                self.small_web.update_edge(old_edge, False)
        else:
            edge = self.small_web.edges.get(pred["edge_id"])
            if edge:
                self.small_web.update_edge(edge, True)

    def _action_to_amount(self, action, population):
        ratios = {"none": 0, "light": 0.15, "moderate": 0.35, "heavy": 0.55}
        return max(0, int(population * ratios.get(action, 0.1)))


class NSEAPAgentDelta:
    """NSEAP with dynamic delta(C) for adaptive closure depth."""

    def __init__(self):
        self.name = "NSEAP-Delta"
        self.suspension = SuspensionLayer()
        self.small_web = SmallWeb(max_depth=10)
        self.last_prediction = None
        self.total_harvested = 0

    def decide(self, obs) -> int:
        if obs.collapsed:
            return 0
        pop = obs.population

        suspended = self.suspension.suspend(pop)
        reasoning = self.small_web.reason(pop, suspended["candidate_actions"])

        if self.last_prediction:
            self._calibrate(pop)

        chosen = self._choose_action(reasoning, suspended["neutral_attributes"])

        if chosen.get("closure_path") and chosen["can_close"]:
            self.small_web.solidify_path(chosen["closure_path"])

        safe_nodes = [n for n in self.small_web.nodes.values() if n.is_safe]
        if safe_nodes:
            lowest_safe = min(n.lo for n in safe_nodes)
            self.suspension.update_boundary(lowest_safe)

        self.last_prediction = {
            "action": chosen["action"],
            "predicted_to": chosen["predicted_to"],
            "edge_id": chosen["edge_id"],
            "from_population": pop,
        }

        return self._action_to_amount(chosen["action"], pop)

    def _choose_action(self, reasoning, attrs):
        candidates = reasoning["candidates"]
        if not reasoning["current_is_safe"]:
            closable = [c for c in candidates if c["can_close"]]
            if closable:
                return min(closable, key=lambda c: c["risk"])
        if attrs.uncertainty > 0.5:
            for c in candidates:
                if c["action"] in ("none", "light") and c["can_close"]:
                    return c
        safe_candidates = [c for c in candidates if c["risk"] < 0.5]
        if safe_candidates:
            return max(safe_candidates, key=lambda c: c["confidence"])
        return min(candidates, key=lambda c: c["risk"])

    def _calibrate(self, actual_population):
        pred = self.last_prediction
        if not pred:
            return
        actual_node = self.small_web.find_node(float(actual_population))
        predicted_node = self.small_web.nodes.get(pred["predicted_to"])
        success = predicted_node is not None and predicted_node.contains(float(actual_population))
        if not success:
            edge = self.small_web.get_or_create_edge(
                self.small_web.find_node(float(pred["from_population"])).name,
                pred["action"],
                actual_node.name,
            )
            self.small_web.update_edge(edge, True)
            old_edge = self.small_web.edges.get(pred["edge_id"])
            if old_edge:
                self.small_web.update_edge(old_edge, False)
        else:
            edge = self.small_web.edges.get(pred["edge_id"])
            if edge:
                self.small_web.update_edge(edge, True)

    def _action_to_amount(self, action, population):
        ratios = {"none": 0, "light": 0.15, "moderate": 0.35, "heavy": 0.55}
        return max(0, int(population * ratios.get(action, 0.1)))


def run_single(agent, seed, max_cycles=200):
    pond = FishPond(seed=seed)

    pop_trace = [int(pond.population)]
    harvest_trace = [0]
    delta_trace = []

    for _ in range(max_cycles):
        current_pop = int(pond.population)
        dummy_obs = pond.history[-1] if pond.history else None
        if dummy_obs is None:
            from environment import Observation
            dummy_obs = Observation(population=current_pop, harvested=0, collapsed=False, cycle=0)

        harvest = agent.decide(dummy_obs)
        obs = pond.step(harvest)
        pop_trace.append(obs.population)
        harvest_trace.append(obs.harvested)

        if hasattr(agent, 'small_web'):
            delta_trace.append(agent.small_web.compute_delta())

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
        delta_trace=delta_trace,
    )


def main():
    parser = argparse.ArgumentParser(description="NSEAP: Fixed-depth vs Delta(C)")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--output", type=str, default="results_delta")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    agent_classes = [NSEAPAgentFixed, NSEAPAgentDelta]
    all_results = {}

    for agent_cls in agent_classes:
        agent_name = agent_cls().name
        runs = []
        print(f"\nRunning {agent_name} ({args.runs} runs x {args.cycles} cycles)...")
        for run_idx in range(args.runs):
            seed = run_idx * 100 + 42
            agent = agent_cls()
            result = run_single(agent, seed, args.cycles)
            runs.append(result)
            if run_idx % 50 == 0:
                print(f"  Run {run_idx}/{args.runs}...")
        all_results[agent_name] = runs

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Agent':<16} {'Survival':>8}  {'Harvest':>10}  {'Collapse%':>10}  {'Edges':>6}  {'AvgDelta':>8}")
    print("-" * 70)
    for name, runs in all_results.items():
        survivals = [r.cycles_survived for r in runs]
        harvests = [r.total_harvested for r in runs]
        collapses = sum(1 for r in runs if r.collapsed)
        avg_edges = sum(r.edge_count for r in runs) / len(runs)
        avg_delta = sum(sum(r.delta_trace) / max(1, len(r.delta_trace)) for r in runs) / len(runs)
        print(
            f"{name:<16} {sum(survivals)/len(survivals):>7.1f}  "
            f"{sum(harvests)/len(harvests):>9.0f}  "
            f"{collapses/len(runs)*100:>9.1f}%  "
            f"{avg_edges:>5.1f}  "
            f"{avg_delta:>7.2f}"
        )
    print("=" * 70)

    # Save detailed results
    summary = {}
    for name, runs in all_results.items():
        summary[name] = {
            "avg_survival": sum(r.cycles_survived for r in runs) / len(runs),
            "avg_harvest": sum(r.total_harvested for r in runs) / len(runs),
            "collapse_rate": sum(1 for r in runs if r.collapsed) / len(runs) * 100,
            "avg_edges": sum(r.edge_count for r in runs) / len(runs),
        }

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save sample delta trace
    for name, runs in all_results.items():
        sample = runs[0]
        with open(os.path.join(args.output, f"{name}_trace.json"), "w", encoding="utf-8") as f:
            json.dump({
                "population": sample.population_trace,
                "harvest": sample.harvest_trace,
                "delta": sample.delta_trace,
                "edges": sample.edge_count,
            }, f, indent=2)

    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()
