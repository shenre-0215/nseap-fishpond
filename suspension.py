"""悬置层（Suspension Layer）— 刺激与响应之间的认知缓冲。

不做的事情：
- 不输出带有预设标签的判断（"鱼少了，多捕"）
- 不直接推荐任何行动

做的事情：
- 提取中性属性（数量、趋势、变化率、与危险边界的距离）
- 生成候选行动列表
- 将决策权交给小网的图推演
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NeutralAttributes:
    quantity: int
    trend: str  # "rising" | "falling" | "stable"
    rate_of_change: float
    distance_to_collapse: float  # 距离已知最低安全边界的距离
    uncertainty: float  # 0-1，不确定性越高越需要谨慎


@dataclass
class SuspensionLayer:
    history: list[int] = field(default_factory=list)
    trend_window: int = 5
    min_safe_boundary: float = 150.0  # 初始安全边界猜测，会在运行中更新

    ACTIONS = ["none", "light", "moderate", "heavy"]

    def suspend(self, population: int) -> dict:
        self.history.append(population)

        attrs = NeutralAttributes(
            quantity=population,
            trend=self._compute_trend(),
            rate_of_change=self._compute_roc(),
            distance_to_collapse=population - self.min_safe_boundary,
            uncertainty=self._compute_uncertainty(population),
        )

        return {
            "neutral_attributes": attrs,
            "evaluation_request": True,
            "candidate_actions": self.ACTIONS,
        }

    def _compute_trend(self) -> str:
        if len(self.history) < 3:
            return "stable"
        recent = self.history[-self.trend_window :]
        if len(recent) < 3:
            return "stable"
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        avg_diff = sum(diffs) / len(diffs)
        if avg_diff > 10:
            return "rising"
        elif avg_diff < -10:
            return "falling"
        return "stable"

    def _compute_roc(self) -> float:
        if len(self.history) < 2:
            return 0.0
        prev = self.history[-2]
        curr = self.history[-1]
        if prev == 0:
            return 0.0
        return (curr - prev) / prev

    def _compute_uncertainty(self, population: int) -> float:
        """不确定性 = 组合信号：接近边界 + 趋势不稳定 + 数据不足"""
        u = 0.0
        # 接近已知危险边界
        if population < self.min_safe_boundary * 2:
            u += 0.3
        if population < self.min_safe_boundary * 1.5:
            u += 0.3
        # 数据不足
        if len(self.history) < 10:
            u += 0.2
        # 趋势波动大
        if len(self.history) >= 5:
            recent = self.history[-5:]
            if max(recent) > 0:
                volatility = (max(recent) - min(recent)) / max(recent)
                if volatility > 0.3:
                    u += 0.2
        return min(u, 1.0)

    def update_boundary(self, new_boundary: float):
        """根据小网的发现更新安全边界估计。"""
        if new_boundary > self.min_safe_boundary:
            self.min_safe_boundary = new_boundary
