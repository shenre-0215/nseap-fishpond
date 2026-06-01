"""鱼塘生态系统模拟器 — 一个简单的复杂系统。

鱼群按 Logistic 曲线增长，存在未知的崩溃临界点。
智能体不知道任何参数，只能在每次捕捞后观察结果。
"""

import random
from dataclasses import dataclass, field


@dataclass
class Observation:
    population: int
    harvested: int
    collapsed: bool
    cycle: int


class FishPond:
    def __init__(
        self,
        carrying_capacity: float = 1000.0,
        growth_rate: float = 0.3,
        critical_threshold: float = 150.0,
        noise: float = 0.05,
        seed: int | None = None,
    ):
        self.K = carrying_capacity
        self.r = growth_rate
        self.threshold = critical_threshold
        self.noise = noise
        self.population = 600.0
        self.collapsed = False
        self.cycle = 0
        self.history: list[Observation] = []
        self._rng = random.Random(seed)

    def step(self, harvest_amount: int) -> Observation:
        if self.collapsed:
            obs = Observation(population=0, harvested=0, collapsed=True, cycle=self.cycle)
            self.history.append(obs)
            return obs

        # Logistic 增长
        growth = self.r * self.population * (1 - self.population / self.K)
        env_noise = self._rng.gauss(0, self.noise * self.population)
        self.population += growth + env_noise
        self.population = max(0.0, self.population)

        # 执行捕捞
        actual = min(harvest_amount, int(self.population))
        self.population -= actual

        # 检查崩溃
        if self.population < self.threshold:
            self.collapsed = True
            self.population = 0.0

        self.cycle += 1
        obs = Observation(
            population=int(self.population),
            harvested=actual,
            collapsed=self.collapsed,
            cycle=self.cycle,
        )
        self.history.append(obs)
        return obs

    def reset(self):
        self.population = 600.0
        self.collapsed = False
        self.cycle = 0
        self.history.clear()
