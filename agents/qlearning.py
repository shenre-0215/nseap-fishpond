"""Q-Learning 智能体 — 有学习，但没有显式悬置层。"""

from collections import defaultdict
import random

from environment import FishPond, Observation


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
    ):
        self.name = "QLearning"
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.Q: dict[tuple, dict[str, float]] = defaultdict(
            lambda: {"none": 0, "light": 0, "moderate": 0, "heavy": 0}
        )
        self.actions = ["none", "light", "moderate", "heavy"]
        self.last_state = None
        self.last_action = None

    def _discretize(self, population: int) -> int:
        """将连续种群数量离散化为状态桶。"""
        if population == 0:
            return 0
        return (population // 100) * 100

    def decide(self, obs) -> int:
        if obs.collapsed:
            return 0

        pop = getattr(obs, 'population', None)
        if pop is None:
            pop = getattr(obs, 'my_population', 0)
        last_harvested = getattr(obs, 'harvested', None)
        if last_harvested is None:
            last_harvested = getattr(obs, 'my_last_harvest', 0)

        state = self._discretize(pop)

        # Q-learning 更新（基于上一次行动的结果）
        if self.last_state is not None and self.last_action is not None:
            reward = last_harvested - (1000 if obs.collapsed else 0)
            best_next = max(self.Q[state].values()) if self.Q[state] else 0
            td_target = reward + self.gamma * best_next
            td_error = td_target - self.Q[self.last_state][self.last_action]
            self.Q[self.last_state][self.last_action] += self.alpha * td_error

        # ε-greedy 选择
        self.epsilon *= self.epsilon_decay
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.Q[state]
            action = max(q_values, key=q_values.get)

        self.last_state = state
        self.last_action = action

        return self._action_to_amount(action, pop)

    def _action_to_amount(self, action: str, population: int) -> int:
        ratios = {"none": 0, "light": 0.15, "moderate": 0.35, "heavy": 0.55}
        return max(0, int(population * ratios.get(action, 0.1)))
