"""保守固定智能体 — 每次固定捕捞，永不冒险，但也永不学习。"""

from environment import FishPond, Observation


class ConservativeAgent:
    def __init__(self, fixed_harvest: int = 50):
        self.name = "Conservative"
        self.fixed_harvest = fixed_harvest

    def decide(self, obs) -> int:
        if obs.collapsed:
            return 0
        pop = getattr(obs, 'population', None)
        if pop is None:
            pop = getattr(obs, 'my_population', 0)
        return min(self.fixed_harvest, pop)
