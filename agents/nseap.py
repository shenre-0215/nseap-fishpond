"""NSEAP 智能体 — 悬置层 + 小网闭环。"""

from environment import FishPond, Observation
from suspension import SuspensionLayer
from small_web import SmallWeb


class NSEAPAgent:
    def __init__(self, max_depth: int = 5, confirm_threshold: int = 3):
        self.name = "NSEAP"
        self.suspension = SuspensionLayer()
        self.small_web = SmallWeb(max_depth=max_depth, confirm_threshold=confirm_threshold)
        self.last_prediction: dict | None = None
        self.total_harvested = 0
        self.collapse_count = 0

    def decide(self, obs) -> int:
        if obs.collapsed:
            return 0

        pop = getattr(obs, 'population', None)
        if pop is None:
            pop = getattr(obs, 'my_population', 0)

        # 1. 悬置：提取中性属性
        suspended = self.suspension.suspend(pop)
        attrs = suspended["neutral_attributes"]

        # 2. 小网推演
        reasoning = self.small_web.reason(
            pop, suspended["candidate_actions"]
        )

        # 3. 校准上一次预测
        if self.last_prediction:
            self._calibrate(pop)

        # 4. 决策：基于推理结果
        chosen = self._choose_action(reasoning, attrs)
        action_name = chosen["action"]

        # 5. 闭环检测与固化
        if chosen.get("closure_path") and chosen["can_close"]:
            self.small_web.solidify_path(chosen["closure_path"])

        # 6. 更新安全边界
        safe_nodes = [n for n in self.small_web.nodes.values() if n.is_safe]
        if safe_nodes:
            lowest_safe = min(n.lo for n in safe_nodes)
            self.suspension.update_boundary(lowest_safe)

        # 7. 记录预测用于下次校准
        amount = self._action_to_amount(action_name, pop)
        self.last_prediction = {
            "action": action_name,
            "predicted_to": chosen["predicted_to"],
            "edge_id": chosen["edge_id"],
            "from_population": pop,
        }

        return amount

    def _choose_action(self, reasoning: dict, attrs) -> dict:
        """基于悬置属性和小网推演选择行动。

        核心逻辑：
        - 不确定性高 → 偏好保守行动
        - 当前不在安全节点 → 优先选能闭环回安全节点的路径
        - 在安全节点内 → 选择置信度高的高收益行动
        """
        candidates = reasoning["candidates"]

        # 如果不安全，强制选择能闭环的路径
        if not reasoning["current_is_safe"]:
            closable = [c for c in candidates if c["can_close"]]
            if closable:
                return min(closable, key=lambda c: c["risk"])

        # 在安全节点内：平衡风险与收益
        if attrs.uncertainty > 0.5:
            # 高不确定性 → 偏好保守
            for c in candidates:
                if c["action"] in ("none", "light") and c["can_close"]:
                    return c

        # 低不确定性 → 选置信度最高的非高风险行动
        safe_candidates = [c for c in candidates if c["risk"] < 0.5]
        if safe_candidates:
            return max(safe_candidates, key=lambda c: c["confidence"])

        # 回退：选风险最低的
        return min(candidates, key=lambda c: c["risk"])

    def _calibrate(self, actual_population: int):
        """对比预测状态 vs 实际状态，更新小网边的权重。"""
        pred = self.last_prediction
        if not pred:
            return

        # 查找实际到达的节点
        actual_node = self.small_web.find_node(float(actual_population))
        predicted_node = self.small_web.nodes.get(pred["predicted_to"])

        # 如果预测节点存在，检查实际是否匹配
        success = predicted_node is not None and predicted_node.contains(
            float(actual_population)
        )

        # 如果实际到达了不同的节点，更新或创建新边
        if not success:
            edge = self.small_web.get_or_create_edge(
                self.small_web.find_node(float(pred["from_population"])).name,
                pred["action"],
                actual_node.name,
            )
            self.small_web.update_edge(edge, True)
            # 降低原预测边的置信度
            old_edge = self.small_web.edges.get(pred["edge_id"])
            if old_edge:
                self.small_web.update_edge(old_edge, False)
        else:
            edge = self.small_web.edges.get(pred["edge_id"])
            if edge:
                self.small_web.update_edge(edge, True)

    def _action_to_amount(self, action: str, population: int) -> int:
        ratios = {"none": 0, "light": 0.15, "moderate": 0.35, "heavy": 0.55}
        return max(0, int(population * ratios.get(action, 0.1)))
