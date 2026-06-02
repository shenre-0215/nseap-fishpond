"""NSEAP 消融变体 v2 —— 激进版，确保行为真正不同。

- NSEAP-NoSusp: 完全跳过悬置，直接选最高收益行动（忽略不确定性和安全）
- NSEAP-NoClose: 跳过闭环检测和固化，纯粹基于边置信度贪心决策
"""

from agents.nseap import NSEAPAgent


class NSEAPNoSuspensionAgent(NSEAPAgent):
    """移除悬置层 —— 激进版：不做中性提取，不考虑不确定性，直接选最高收益。"""

    def __init__(self, max_depth=5, confirm_threshold=3):
        super().__init__(max_depth=max_depth, confirm_threshold=confirm_threshold)
        self.name = "NSEAP-NoSusp"

    def decide(self, obs):
        if obs.collapsed:
            return 0

        pop = getattr(obs, 'population', None)
        if pop is None:
            pop = getattr(obs, 'my_population', 0)

        # 无悬置：直接推理
        actions = ["none", "light", "moderate", "heavy"]
        reasoning = self.small_web.reason(pop, actions)

        if self.last_prediction:
            self._calibrate(pop)

        # 激进决策：总是选最高收益（heavy > moderate > light > none）
        chosen = self._choose_action_greedy(reasoning)

        if chosen.get("closure_path") and chosen["can_close"]:
            self.small_web.solidify_path(chosen["closure_path"])

        safe_nodes = [n for n in self.small_web.nodes.values() if n.is_safe]
        if safe_nodes:
            lowest_safe = min(n.lo for n in safe_nodes)
            self.suspension.update_boundary(lowest_safe)

        amount = self._action_to_amount(chosen["action"], pop)
        self.last_prediction = {
            "action": chosen["action"],
            "predicted_to": chosen["predicted_to"],
            "edge_id": chosen["edge_id"],
            "from_population": pop,
        }
        return amount

    def _choose_action_greedy(self, reasoning):
        """完全忽略不确定性和安全，选最高收益。"""
        action_order = ["heavy", "moderate", "light", "none"]
        candidates = reasoning["candidates"]
        for action in action_order:
            for c in candidates:
                if c["action"] == action:
                    return c
        return candidates[0]


class NSEAPNoClosureAgent(NSEAPAgent):
    """移除闭环检测 —— 激进版：小网正常生长，但决策不检查闭环，也不固化路径。"""

    def __init__(self, max_depth=5, confirm_threshold=3):
        super().__init__(max_depth=max_depth, confirm_threshold=confirm_threshold)
        self.name = "NSEAP-NoClose"

    def decide(self, obs):
        if obs.collapsed:
            return 0

        pop = getattr(obs, 'population', None)
        if pop is None:
            pop = getattr(obs, 'my_population', 0)

        suspended = self.suspension.suspend(pop)
        attrs = suspended["neutral_attributes"]
        actions = suspended["candidate_actions"]
        reasoning = self.small_web.reason(pop, actions)

        if self.last_prediction:
            self._calibrate(pop)

        # 不检查闭环，不固化，基于不确定性和置信度决策
        chosen = self._choose_action_no_closure(reasoning, attrs)

        # 仍然更新安全边界
        safe_nodes = [n for n in self.small_web.nodes.values() if n.is_safe]
        if safe_nodes:
            lowest_safe = min(n.lo for n in safe_nodes)
            self.suspension.update_boundary(lowest_safe)

        amount = self._action_to_amount(chosen["action"], pop)
        self.last_prediction = {
            "action": chosen["action"],
            "predicted_to": chosen["predicted_to"],
            "edge_id": chosen["edge_id"],
            "from_population": pop,
        }
        return amount

    def _choose_action_no_closure(self, reasoning, attrs):
        """决策不要求闭环，只看风险和置信度。"""
        candidates = reasoning["candidates"]

        if not reasoning["current_is_safe"]:
            # 不在安全节点：选风险最低的（但不要求闭环）
            safe_candidates = [c for c in candidates if c["risk"] < 0.5]
            if safe_candidates:
                return min(safe_candidates, key=lambda c: c["risk"])
            return min(candidates, key=lambda c: c["risk"])

        if attrs.uncertainty > 0.5:
            for c in candidates:
                if c["action"] in ("none", "light"):
                    return c

        # 低不确定性：选置信度最高的非高风险行动
        safe_candidates = [c for c in candidates if c["risk"] < 0.5]
        if safe_candidates:
            return max(safe_candidates, key=lambda c: c["confidence"])

        return min(candidates, key=lambda c: c["risk"])
