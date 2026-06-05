"""Small Web - a relation-prioritized dynamic ontology.

Nodes = state intervals (e.g. "low": [100, 300])
Edges = action-induced state transitions with probability and confidence

Core mechanisms:
- Edges grow dynamically (tentative -> closure verification -> solidify/reject)
- Closure detection: DFS from current node, can we reach any safe node within delta(C) steps?
- delta(C): adaptive closure tolerance - sparse network -> deep search (exploratory),
  dense network -> shallow search (conservative)
- Confidence grows with observation count
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    name: str
    lo: float
    hi: float
    is_safe: bool = False

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


@dataclass
class Edge:
    id: str
    from_node: str
    action: str
    to_node: str
    probability: float = 0.5
    observation_count: int = 0
    success_count: int = 0
    confidence: float = 0.0
    status: str = "tentative"  # tentative | confirmed | rejected


@dataclass
class SmallWeb:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    max_depth: int = 10
    confirm_threshold: int = 3

    # delta(C) parameters
    _alpha: float = 5.0
    _epsilon: float = 0.1
    _M0: float = 10.0
    _N_closed: int = 0
    _N_trials: int = 0

    def __post_init__(self):
        if not self.nodes:
            self.nodes = {
                "critical_low": Node("critical_low", 0, 150, is_safe=False),
                "low": Node("low", 150, 300, is_safe=True),
                "moderate": Node("moderate", 300, 600, is_safe=True),
                "abundant": Node("abundant", 600, 1000, is_safe=True),
            }

    # ---- delta(C): adaptive closure tolerance ----

    def compute_delta(self) -> float:
        """Adaptive closure tolerance -> dynamic search depth.

        Young/sparse network -> high delta -> deep search (exploratory)
        Mature/dense network -> low delta -> shallow search (conservative)

        delta(C) = alpha / (rho_adj + epsilon) * log(1 + N_closed / N_trials)

        rho_adj = (|E|/|V|) * min(1, total_obs/M0) + epsilon
        """
        total_obs = sum(e.observation_count for e in self.edges.values())

        # Insufficient data -> use max depth (full exploration)
        if total_obs < 3:
            return float(self.max_depth)

        # rho(C): edge density = |E| / |V|
        n_nodes = max(1, len(self.nodes))
        rho = len(self.edges) / n_nodes

        # M(C)/M0: observation sufficiency calibration
        M_ratio = min(1.0, total_obs / self._M0)
        rho_adj = rho * M_ratio + self._epsilon

        # Historical closure success rate
        attempts = max(1, self._N_trials)
        success_term = math.log(1 + self._N_closed / attempts)

        delta = (self._alpha / rho_adj) * success_term

        # Clamp to [2, max_depth]
        return max(2.0, min(float(self.max_depth), delta))

    # ---- node operations ----

    def find_node(self, population: float) -> Node:
        for node in self.nodes.values():
            if node.contains(population):
                return node
        return self._create_node_for(population)

    def _create_node_for(self, population: float) -> Node:
        lo = max(0, population * 0.8)
        hi = population * 1.2
        name = f"zone_{int(population)}"
        node = Node(name, lo, hi, is_safe=population > 150)
        self.nodes[name] = node
        return node

    # ---- edge operations ----

    def get_or_create_edge(
        self, from_name: str, action: str, to_name: str
    ) -> Edge:
        for e in self.edges.values():
            if e.from_node == from_name and e.action == action and e.to_node == to_name:
                return e
        edge = Edge(
            id=str(uuid.uuid4())[:8],
            from_node=from_name,
            action=action,
            to_node=to_name,
        )
        self.edges[edge.id] = edge
        return edge

    def update_edge(self, edge: Edge, success: bool):
        edge.observation_count += 1
        if success:
            edge.success_count += 1
        edge.probability = edge.success_count / edge.observation_count
        edge.confidence = 1.0 - 1.0 / (edge.observation_count + 1)
        if edge.observation_count >= self.confirm_threshold and edge.probability > 0.5:
            edge.status = "confirmed"
        elif edge.observation_count >= self.confirm_threshold * 2 and edge.probability < 0.3:
            edge.status = "rejected"

    # ---- closure detection ----

    def detect_closure(self, start_name: str) -> Optional[list[str]]:
        """DFS from start_node using delta(C) as dynamic depth limit."""
        depth_limit = int(self.compute_delta())
        self._N_trials += 1

        def dfs(current: str, depth: int, visited: set[str]) -> Optional[list[str]]:
            if depth > depth_limit:
                return None
            current_node = self.nodes.get(current)
            if current_node and current_node.is_safe and depth > 0:
                return [current]
            for edge in self.edges.values():
                if (
                    edge.from_node == current
                    and edge.status in ("confirmed", "tentative")
                    and edge.to_node not in visited
                ):
                    result = dfs(edge.to_node, depth + 1, visited | {current})
                    if result:
                        return [current] + result
            return None

        result = dfs(start_name, 0, set())
        if result is not None:
            self._N_closed += 1
        return result

    def solidify_path(self, path: list[str]):
        for i in range(len(path) - 1):
            for edge in self.edges.values():
                if (
                    edge.from_node == path[i]
                    and edge.to_node == path[i + 1]
                    and edge.status == "tentative"
                ):
                    edge.status = "confirmed"

    # ---- reasoning ----

    def reason(self, population: int, actions: list[str]) -> dict:
        current = self.find_node(float(population))
        results = []

        for action in actions:
            candidates = [
                e
                for e in self.edges.values()
                if e.from_node == current.name and e.action == action
            ]
            if candidates:
                edge = max(candidates, key=lambda e: e.confidence)
                results.append(
                    {
                        "action": action,
                        "predicted_to": edge.to_node,
                        "confidence": edge.confidence,
                        "probability": edge.probability,
                        "status": edge.status,
                        "edge_id": edge.id,
                    }
                )
            else:
                ratio = self._action_ratio(action)
                predicted_pop = population * (1 + ratio)
                to_node = self.find_node(predicted_pop)
                edge = self.get_or_create_edge(current.name, action, to_node.name)
                results.append(
                    {
                        "action": action,
                        "predicted_to": to_node.name,
                        "confidence": 0.0,
                        "probability": 0.5,
                        "status": "tentative",
                        "edge_id": edge.id,
                    }
                )

        for r in results:
            closure = self.detect_closure(r["predicted_to"])
            r["closure_path"] = closure
            r["can_close"] = closure is not None
            to_node = self.nodes.get(r["predicted_to"])
            r["risk"] = 0.0 if (to_node and to_node.is_safe) else 0.5
            if not r["can_close"]:
                r["risk"] += 0.3

        return {
            "current_node": current.name,
            "current_is_safe": current.is_safe,
            "delta": self.compute_delta(),
            "candidates": sorted(
                results, key=lambda r: (r["risk"], -r["confidence"])
            ),
        }

    def _action_ratio(self, action: str) -> float:
        ratios = {
            "none": 0.15,
            "light": 0.05,
            "moderate": -0.05,
            "heavy": -0.20,
        }
        return ratios.get(action, -0.05)

    # ---- serialization ----

    def snapshot(self) -> dict:
        return {
            "nodes": {
                k: {"lo": v.lo, "hi": v.hi, "is_safe": v.is_safe}
                for k, v in self.nodes.items()
            },
            "edges": {
                k: {
                    "from": v.from_node,
                    "action": v.action,
                    "to": v.to_node,
                    "prob": v.probability,
                    "conf": v.confidence,
                    "obs": v.observation_count,
                    "status": v.status,
                }
                for k, v in self.edges.items()
            },
        }
