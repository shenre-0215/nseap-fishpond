"""贪心智能体 — 没有悬置层，默认"越多越好"。"""

from environment import FishPond, Observation


class GreedyAgent:
    def __init__(self):
        self.name = "Greedy"

    def decide(self, obs) -> int:
        if obs.collapsed:
            return 0
        pop = getattr(obs, 'population', None)
        if pop is None:
            pop = getattr(obs, 'my_population', 0)
        return max(1, pop // 2)
