"""NSEAP-Static 消融变体 —— 预定义完整本体，无动态生长。

与完整 NSEAP 的唯一区别：小网在初始化时就包含所有可能的节点和边，
运行时不再创建新节点或新边。只能更新已有边的置信度。
这测试"动态本体生长"是否不可替代。
"""

from agents.nseap import NSEAPAgent
from small_web import SmallWeb, Node, Edge
import uuid


class SmallWebStatic(SmallWeb):
    """预定义完整本体的静态小网 —— 不会动态生长。"""

    def __init__(self, max_depth=5, confirm_threshold=3):
        # 不调用父类 __init__，手动构建完整本体
        # 绕过 SmallWeb.__post_init__ 的默认节点创建
        self.nodes = {}
        self.edges = {}
        self.max_depth = max_depth
        self.confirm_threshold = confirm_threshold

        # 预定义所有节点（与默认 SmallWeb 一致）
        node_defs = [
            ("critical_low", 0, 150, False),
            ("low", 150, 300, True),
            ("moderate", 300, 600, True),
            ("abundant", 600, 1000, True),
        ]
        for name, lo, hi, safe in node_defs:
            self.nodes[name] = Node(name, lo, hi, is_safe=safe)

        # 预定义所有可能的边（4 nodes × 4 actions = 16 edges）
        actions = ["none", "light", "moderate", "heavy"]
        action_ratios = {
            "none": 0.15,
            "light": 0.05,
            "moderate": -0.05,
            "heavy": -0.20,
        }

        for from_name, from_node in self.nodes.items():
            for action in actions:
                # 预测目标节点：基于行动比率
                mid_from = (from_node.lo + from_node.hi) / 2
                predicted = mid_from * (1 + action_ratios[action])
                to_node = None
                for n in self.nodes.values():
                    if n.contains(predicted):
                        to_node = n
                        break
                if to_node is None:
                    to_node = self.nodes["critical_low"] if predicted < 150 else self.nodes["abundant"]

                edge = Edge(
                    id=str(uuid.uuid4())[:8],
                    from_node=from_name,
                    action=action,
                    to_node=to_node.name,
                    probability=0.5,
                    observation_count=0,
                    success_count=0,
                    confidence=0.0,
                    status="tentative",
                )
                self.edges[edge.id] = edge

    def _create_node_for(self, population):
        """静态版本：不创建新节点，返回最近的已有节点。"""
        best_node = None
        best_dist = float('inf')
        mid = population
        for node in self.nodes.values():
            dist = abs((node.lo + node.hi) / 2 - mid)
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return best_node

    def get_or_create_edge(self, from_name, action, to_name):
        """静态版本：不创建新边，返回最匹配的已有边。"""
        for e in self.edges.values():
            if e.from_node == from_name and e.action == action and e.to_node == to_name:
                return e
        # 回退：返回同 from + action 的第一条边
        for e in self.edges.values():
            if e.from_node == from_name and e.action == action:
                return e
        # 最终回退：返回任意第一条边（不太可能发生）
        return list(self.edges.values())[0]


class NSEAPStaticAgent(NSEAPAgent):
    """NSEAP 静态变体 —— 预定义完整本体，无动态生长能力。"""

    def __init__(self, max_depth=5, confirm_threshold=3):
        # 绕过父类 __init__，使用静态小网
        self.name = "NSEAP-Static"
        self.suspension = __import__('suspension').SuspensionLayer()
        self.small_web = SmallWebStatic(
            max_depth=max_depth, confirm_threshold=confirm_threshold
        )
        self.last_prediction = None
        self.total_harvested = 0
        self.collapse_count = 0
