"""两智能体共享一个大湖 — 公共资源博弈场景。

总鱼群由两个智能体共同捕捞。如果总捕捞量太大，跌破临界点，一起崩溃。
"""

from dataclasses import dataclass, field
import random


@dataclass
class TwoAgentObservation:
    my_population: int      # 你感知到的当前总鱼群（不知道对方捞了多少，只知道剩下多少）
    my_last_harvest: int     # 我上一轮捞了多少
    collapsed: bool          # 是不是已经崩了
    cycle: int


class TwoAgentFishPond:
    def __init__(
        self,
        total_carrying_capacity: float = 2000.0,
        growth_rate: float = 0.3,
        critical_threshold: float = 300.0,
        noise: float = 0.05,
        seed: int | None = None,
    ):
        self.K = total_carrying_capacity
        self.r = growth_rate
        self.threshold = critical_threshold
        self.noise = noise
        self.population = 1200.0  # 初始总鱼群
        self.collapsed = False
        self.cycle = 0
        self._rng = random.Random(seed)

    def step(self, harvest1: int, harvest2: int) -> tuple[TwoAgentObservation, TwoAgentObservation]:
        """两个智能体各捞一次，返回各自的观测。"""
        if self.collapsed:
            obs1 = TwoAgentObservation(0, harvest1, True, self.cycle)
            obs2 = TwoAgentObservation(0, harvest2, True, self.cycle)
            return obs1, obs2

        # Logistic 增长
        growth = self.r * self.population * (1 - self.population / self.K)
        env_noise = self._rng.gauss(0, self.noise * self.population)
        self.population += growth + env_noise
        self.population = max(0.0, self.population)

        # 两个智能体都捞
        actual1 = min(harvest1, int(self.population))
        actual2 = min(harvest2, int(self.population - actual1))
        self.population -= actual1 + actual2

        # 检查崩溃
        if self.population < self.threshold:
            self.collapsed = True
            self.population = 0.0

        self.cycle += 1

        # 返回观测（每个智能体只知道总剩余和自己上次捞了多少）
        obs1 = TwoAgentObservation(
            my_population=int(self.population),
            my_last_harvest=actual1,
            collapsed=self.collapsed,
            cycle=self.cycle,
        )
        obs2 = TwoAgentObservation(
            my_population=int(self.population),
            my_last_harvest=actual2,
            collapsed=self.collapsed,
            cycle=self.cycle,
        )
        return obs1, obs2

    def reset(self):
        self.population = 1200.0
        self.collapsed = False
        self.cycle = 0
