"""小网（Small Web）— 可闭环、自洽、可映射的关系图。

节点 = 状态区域（如 "low": [100, 300]）
边   = 行动导致的状态转移，附带概率和置信度

核心机制：
- 边是动态生成的（试探 → 闭环验证 → 固化/丢弃）
- 闭环检测：从当前节点沿 confirmed 边走 N 步，能否回到安全节点
- 置信度随观察次数增长
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    name: str
    lo: float  # 数量下界
    hi: float  # 数量上界
    is_safe: bool = False  # 已知安全节点

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
    max_depth: int = 5
    confirm_threshold: int = 3  # 多少次成功观察后确认

    def __post_init__(self):
        # 预设初始状态节点（range 是示例值，会在运行中被调整）
        if not self.nodes:
            self.nodes = {
                "critical_low": Node("critical_low", 0, 150, is_safe=False),
                "low": Node("low", 150, 300, is_safe=True),
                "moderate": Node("moderate", 300, 600, is_safe=True),
                "abundant": Node("abundant", 600, 1000, is_safe=True),
            }

    # ---- 节点操作 ----

    def find_node(self, population: float) -> Node:
        for node in self.nodes.values():
            if node.contains(population):
                return node
        # 扩展边界
        return self._create_node_for(population)

    def _create_node_for(self, population: float) -> Node:
        """为新出现的种群数量动态创建节点。"""
        existing = sorted([(n.lo, n.hi) for n in self.nodes.values()])
        lo = max(0, population * 0.8)
        hi = population * 1.2
        name = f"zone_{int(population)}"
        node = Node(name, lo, hi, is_safe=population > 150)
        self.nodes[name] = node
        return node

    # ---- 边操作 ----

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

    # ---- 闭环检测 ----

    def detect_closure(self, start_name: str) -> Optional[list[str]]:
        """从 start_node 出发 DFS，检查能否回到任意 is_safe 节点。"""

        def dfs(current: str, depth: int, visited: set[str]) -> Optional[list[str]]:
            if depth > self.max_depth:
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

        return dfs(start_name, 0, set())

    def solidify_path(self, path: list[str]):
        """将闭环路径上的 tentative 边固化为 confirmed。"""
        for i in range(len(path) - 1):
            for edge in self.edges.values():
                if (
                    edge.from_node == path[i]
                    and edge.to_node == path[i + 1]
                    and edge.status == "tentative"
                ):
                    edge.status = "confirmed"

    # ---- 推演 ----

    def reason(self, population: int, actions: list[str]) -> dict:
        """给定当前状态和候选行动，预测每条行动的结果。"""
        current = self.find_node(float(population))
        results = []

        for action in actions:
            # 找到或创建边
            candidates = [
                e
                for e in self.edges.values()
                if e.from_node == current.name and e.action == action
            ]
            if candidates:
                edge = max(candidates, key=lambda e: e.confidence)
                to_node = self.nodes.get(edge.to_node)
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
                # 试探性预测：假设行动会按比例改变状态
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

        # 为每条路径检测闭环
        for r in results:
            closure = self.detect_closure(r["predicted_to"])
            r["closure_path"] = closure
            r["can_close"] = closure is not None
            # 计算风险
            to_node = self.nodes.get(r["predicted_to"])
            r["risk"] = 0.0 if (to_node and to_node.is_safe) else 0.5
            if not r["can_close"]:
                r["risk"] += 0.3

        return {
            "current_node": current.name,
            "current_is_safe": current.is_safe,
            "candidates": sorted(
                results, key=lambda r: (r["risk"], -r["confidence"])
            ),
        }

    def _action_ratio(self, action: str) -> float:
        """行动对种群数量的近似影响比例。"""
        ratios = {
            "none": 0.15,
            "light": 0.05,
            "moderate": -0.05,
            "heavy": -0.20,
        }
        return ratios.get(action, -0.05)

    # ---- 序列化 ----

    def snapshot(self) -> dict:
        return {
            "nodes": {k: {"lo": v.lo, "hi": v.hi, "is_safe": v.is_safe} for k, v in self.nodes.items()},
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
