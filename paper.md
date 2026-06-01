# NSEAP: Neural-Symbolic Evolutionary Agent Platform for Safe Learning in Complex Critical Systems

**Author:** shen  
**Affiliation:** Zhengzhou Sias University, Zhengzhou, China  
**Email:** 359005185@qq.com

---

## Abstract

> **问题：** 在存在不可逆临界点的复杂系统中，传统强化学习面临"探索即崩溃"的根本困境——随机探索可能直接跌破临界点导致系统永久崩溃，无法完成学习。  
> **方法：** 本文提出 NSEAP（神经符号进化智能体平台），一种全新的智能体架构：(1) **悬置层**插入感知与推理之间，剥离预设标签，仅提取中性属性；(2) **小网**是一个动态生长的关系图，节点表示状态区间，边表示行动导致的状态转移，边的置信度通过观察逐步更新；(3) **实用闭环**机制，只有能在有限步内闭环回到已知安全节点的路径才被固化为知识。本体从交互中动态生长，而非预先给定。  
> **结果：** 在鱼塘生态管理基准问题上的实验表明，NSEAP 在 100 次独立运行中实现了 **0% 崩溃率**，累计收获比保守固定策略高出 **30.6%**，相比 Q-Learning（50% 崩溃率）鲁棒性显著提升。在两智能体公共资源博弈实验中，NSEAP 在初始不确定性极高时自动选择零捕捞策略，实现 **0% 崩溃率**，严格遵守"不确定性管理优先"的设计原则。  
> **结论：** NSEAP 为复杂临界系统中的安全学习提供了一条新路——不追求全局最优，通过保守试探、闭环验证、动态生长，在保证系统不崩溃的前提下逐步积累知识、最大化收益。

**Keywords:** 复杂系统, 临界态, 强化学习, 神经符号AI, 动态本体, 安全探索

---

## 1 Introduction

复杂系统在自然界和人类社会中无处不在：生态系统、金融市场、电力电网、气候变化都表现出非线性动力学特征，存在**临界点**——一旦系统状态跌破临界点，会发生不可逆崩溃，无法恢复 [1,2]。

在这类系统中做出自主决策，对人工智能提出了根本性挑战：

1. **探索即崩溃**：传统强化学习基于"探索-利用"框架，必须通过随机探索来学习价值函数，但在临界系统中，一次错误探索就可能导致系统永久崩溃，学习无法继续 [3]。

2. **预设本体偏见**：传统符号AI依赖人类专家预定义本体，但在开放环境中，专家也无法预知所有可能的状态和关系，预设本体必然存在偏见 [4]。

3. **不完备性困境**：追求全局一致性的知识系统会陷入哥德尔不完备性陷阱，无法在开放环境中持续生长 [5]。

本文提出 **NSEAP (Neural-Symbolic Evolutionary Agent Platform)**，从架构层面回应这三个挑战：

- **悬置层 (Suspension Layer)**：插入感知与推理之间，不立即做出预设判断，仅提取中性属性（数量、趋势、不确定性、到安全边界的距离），延迟本体承诺。

- **小网 (Small Web)**：关系优先的动态关系图。节点表示状态区间，边表示行动导致的状态转移。边不是预定义的，而是在交互中**动态生长**——试探→验证→固化/丢弃。

- **实用闭环 (Practical Closure)**：不追求全局一致性，只要求在有限深度内能闭环回到已知安全节点。能闭环的路径才被固化为知识，避开不完备性陷阱。

NSEAP 的核心设计原则是：**不确定性管理优先于收益最大化**。在复杂临界系统中，活下来比一时的收益更重要。

本文在两个经典问题上验证 NSEAP：

1. **单智能体鱼塘生态管理**：智能体管理一个鱼塘，存在未知崩溃临界点，目标是在不崩溃的前提下最大化累计捕捞量。

2. **两智能体公共资源博弈**（公地悲剧）：两个智能体共享一个鱼塘，各自决策，过度捕捞会导致集体崩溃。

实验结果表明：
- NSEAP 在单智能体任务中实现零崩溃，收获显著高于保守策略
- 在公共资源博弈中，当初始不确定性极高时，NSEAP 自动选择最保守策略，保持零崩溃
- 结果验证了 NSEAP 架构设计的有效性

## 2 Related Work

### 2.1 Safe Reinforcement Learning

安全强化学习研究如何在学习过程中避免危险行为。主要方法包括：
- **约束马尔可夫决策过程** [6]：将安全性表示为约束，在约束下优化奖励
- **风险敏感RL** [7]：在目标函数中加入风险项，惩罚不确定性
- **贝叶斯RL** [8]：估计不确定性，根据不确定性调整探索率

与这些方法相比，NSEAP 走了一条完全不同的路：NSEAP 不通过调整奖励或惩罚来约束探索，而是通过**拓扑闭环检测**来筛选安全路径——只有能回到已知安全状态的行动才被认为是安全的。

### 2.2 Neural-Symbolic AI

神经符号AI结合神经网络和符号推理，目标是得到可解释、可推广的系统 [9]。传统神经符号AI通常依赖预定义符号本体 [10]。

NSEAP 的不同之处在于：**本体不是预定义的，而是在与环境的交互中动态生长出来的**。每一条知识（边）都附带置信度，只有经过闭环验证才被固化。

### 2.3 Dynamic Ontology

动态本体论认为本体应该随时间和经验进化 [11]。现有工作主要关注离线更新 [12]。

NSEAP 实现了**在线交互生长**：智能体每一步交互都可能产生新节点、新边，验证后固化。这是一个持续的进化过程。

### 2.4 Commonsense Reasoning & Closure

哲学和认知科学早就讨论过"悬置判断"（epoché）的概念 [13]，现象学认为在认识事物之前需要悬置预设判断。NSEAP 将这个哲学概念实现为AI架构中的一个显式组件。

实用主义认为"有用就是真理" [14]。NSEAP 的闭环标准就是实用主义的——能闭环回安全节点的知识就是可靠知识，不需要全局一致性。

## 3 NSEAP Architecture

### 3.1 Core Design Principles

NSEAP 基于三个核心设计原则：

1. **悬置判断，延迟本体承诺**：不预设"什么是对的"，先提取中性属性，再推理
2. **关系优先，动态生长**：先学习"行动→状态"的关系，再基于关系做决策；关系从试探中生长出来
3. **有限闭环，实用可靠**：不追求全局一致性，只要能在有限步回到安全节点，就接受这个知识

### 3.2 Three-Layer Architecture

```
Perception → Suspension Layer → Small Web Reasoning → Decision → Execution → Calibration → Closure → ...
```

### 3.3 Suspension Layer

悬置层位于感知和推理之间。它**不输出决策**，只做三件事：
1. 从原始感知中提取**中性属性**
2. 生成候选行动列表，但不推荐任何一个
3. 将决策权交给小网推演

提取的中性属性包括：
| Attribute | Meaning |
|-----------|---------|
| `quantity` | 当前状态数量 |
| `trend` | 趋势：rising / falling / stable |
| `rate_of_change` | 变化率 |
| `distance_to_collapse` | 到已知安全边界的距离 |
| `uncertainty` | 综合不确定性 (0-1) |

不确定性计算公式：
```
uncertainty = sum of:
  +0.3 if within 2x of safety boundary
  +0.3 if within 1.5x of safety boundary
  +0.2 if fewer than 10 observations
  +0.2 if high volatility
capped at 1.0
```

算法 1 描述了悬置过程：

```
Algorithm 1: Suspend(population)
  Add population to history
  Compute trend from recent history
  Compute rate of change
  Compute distance to known safe boundary
  Compute uncertainty
  return {
    neutral_attributes: (quantity, trend, rate_of_change, distance, uncertainty),
    candidate_actions: [none, light, moderate, heavy]
  }
```

### 3.4 Small Web: Dynamic Relation Graph

小网是一个动态增长的关系图：

- **Node**：表示一个状态区间 `[lo, hi]`，标记是否已知安全
- **Edge**：表示从节点 A 采取行动 X 会转移到节点 B，附带：
  - `confidence`：置信度，随观察次数增长
  - `probability`：成功转移概率
  - `status`：tentative / confirmed / rejected

置信度计算公式：
```
confidence = 1 - 1 / (observation_count + 1)
```
观察次数越多，置信度越高。

边的状态转换规则：
- 初始：`tentative`
- 如果 `observation_count >= confirm_threshold` 且 `probability > 0.5` → `confirmed`
- 如果 `observation_count >= 2 * confirm_threshold` 且 `probability < 0.3` → `rejected`

小网支持动态生长：如果当前状态不属于任何已有节点，自动创建新节点。

### 3.5 Closure Detection and Solidification

**闭环检测**：从当前节点出发，沿边深度优先搜索，检查能否在 `max_depth` 步内回到任意已知安全节点。

```
Algorithm 2: DetectClosure(start_node, max_depth)
  def DFS(current, depth, visited):
    if depth > max_depth:
      return None
    if current is safe and depth > 0:
      return [current]
    for each edge from current:
      if edge.to not in visited:
        result = DFS(edge.to, depth + 1, visited ∪ {current})
        if result != None:
          return [current] + result
    return None
  return DFS(start_node, 0, ∅)
```

如果找到闭环路径，将路径上所有 `tentative` 边固化为 `confirmed`：

```
Algorithm 3: SolidifyPath(path)
  for i from 0 to len(path)-2:
    edge = get_edge(path[i], path[i+1])
    if edge.status == tentative:
      edge.status = confirmed
```

这个机制实现了：**只有经过闭环验证的知识才被正式接受**。

### 3.6 Decision Making

决策逻辑：

1. 如果当前节点不安全，优先选择能闭环的低风险路径
2. 如果在安全节点内：
   - 不确定性 > 0.5 → 优先选择保守行动（none / light）且能闭环
   - 低不确定性 → 选择置信度最高的低风险行动
3. 回退：选择风险最低的行动

这一逻辑保证了：**不确定性越高，越保守**。

### 3.7 Calibration

每一步执行后，对比预测状态和实际状态，更新边权重：

- 如果预测正确 → 提高对应边的置信度
- 如果预测错误 → 创建新边反映真实转移，降低原边置信度

这个校准过程保证了：**错误预测会被快速纠正**。

### 3.8 Complete Algorithm Flow

```
Algorithm 4: Complete NSEAP Cycle(observation)
  1. Get current population from observation
  2. Suspend: extract neutral attributes and candidate actions
  3. Reason: for each candidate action, predict next state and check closure
  4. Decide: select action based on risk and confidence
  5. If action can close, solidify the closure path
  6. Update safety boundary based on lowest known safe node
  7. Execute action in environment
  8. Calibrate: compare predicted state with actual state, update edge confidence
  9. Record prediction for next calibration
  10. Return harvest amount
```

## 4 Experiment 1: Single-Agent Fish Pond Management

### 4.1 Problem Description

我们用鱼塘生态管理作为基准问题：

- 鱼群按 Logistic 曲线增长
- 存在未知临界点：跌破阈值后不可逆崩溃
- 智能体初始不知道临界点、增长率、承载力
- 每轮决定捕捞量
- 目标：在不崩溃的前提下最大化累计捕捞量

环境动力学：

```
Logistic growth:
dP/dt = r * P * (1 - P/K) + ε
```

其中：
- r = 增长率
- K = 环境承载力
- ε ~ N(0, σ²P²) = 随机噪声

默认参数：
- K = 1000
- r = 0.3
- 临界点 = 150
- σ = 0.05
- 初始 P = 600

### 4.2 Compared Agents

我们对比四种智能体：

| Agent | Description |
|-------|-------------|
| **Greedy** | 每次捕捞当前鱼群的 50%，没有学习 |
| **Conservative** | 每次固定捕捞 50 条，固定策略，没有学习 |
| **QLearning** | 标准 Q-Learning，离散状态桶，ε-greedy 探索 |
| **NSEAP** | 本文提出的方法，max_depth=5, confirm_threshold=3 |

### 4.3 Evaluation Metrics

- 平均存活周期数
- 平均累计收获
- 崩溃率（100次运行中崩溃的比例）

### 4.4 Experimental Setup

- 独立运行次数：100
- 最大周期数：200
- 每个运行使用不同随机种子

### 4.5 Results

Table 1: 单智能体实验结果（100次运行）

| Agent | Avg Survival Cycles | Avg Total Harvest | Collapse Rate |
|-------|---------------------|-------------------|---------------|
| Greedy | 4.2 | 728 | **100.0%** |
| Conservative | 200.0 | 10,000 | 0.0% |
| QLearning | 139.1 | 9,193 | 50.0% |
| **NSEAP** | **200.0** | **13,059** | **0.0%** |

Figure 1: 结果对比图  
![Comparison](results/comparison.png)

### 4.6 Discussion

从结果可以看出：

1. **贪心策略**：短期收益高，但几乎必然崩溃，验证了在临界系统中贪心的危险性
2. **保守固定策略**：永不崩溃，但收获远低于可持续上限
3. **QLearning**：需要探索，约一半概率在探索中崩溃
4. **NSEAP**：实现了零崩溃，同时收获比保守策略高出 **30.6%**，证明了 NSEAP 能在安全的前提下有效学习并提高收益

### 4.7 Parameter Sensitivity Analysis

我们测试了不同环境参数下 NSEAP 的表现：

Table 2: 参数敏感性测试（崩溃率）

| Condition | NSEAP | Conservative | QLearning |
|-----------|-------|--------------|-----------|
| Default | **0%** | 0% | 50% |
| Double noise (σ=0.1) | **0%** | 0% | 58% |
| Lower growth (r=0.2) | **0%** | 12% | 67% |

结论：
- 在各种环境条件下，NSEAP 的崩溃率都低于或等于对比方法
- 当系统变脆弱（增长率降低），保守策略自己也开始崩溃，NSEAP 仍保持零崩溃

### 4.8 Dynamic Boundary Correction Experiment

We further test whether NSEAP can **self-correct** when the initial guess of the safety boundary is wrong. True critical threshold is 150, we set different initial guesses.

Table 3: Self-correction experiment results (100 runs)

| Initial safety boundary guess | NSEAP collapse rate | Final average boundary | Result |
|-------------------------------|---------------------|------------------------|--------|
| 100 (guess lower than true) | **0.0%** | **150.0** | ✅ **Perfect self-correction**: NSEAP automatically corrects the boundary from 100 to the true value 150 through exploration and calibration, still maintains zero collapse |
| 150 (correct) | 0.0% | 150.0 | Works as expected |
| 250 (guess higher than true) | 0.0% | 250.0 | NSEAP never enters the region < 250, remains conservative, zero harvest zero collapse |
| 400 (guess much higher) | 0.0% | 400.0 | Same as above, remains conservative |

Discussion:

- When initial guess is **aggressive (guess lower than true)**: NSEAP can discover the true critical threshold through exploration, dynamically adjust the safety boundary, and still maintain zero collapse. This validates the **dynamic ontology correction capability** of NSEAP.

- When initial guess is **conservative (guess higher than true)**: NSEAP keeps zero collapse but never explores the unknown region below the guessed boundary. This is consistent with the design principle: **when uncertainty exceeds threshold, prioritize system stability over收益**.

- In all cases, NSEAP maintains **zero collapse** — the "uncertainty first" design principle works as intended.

## 5 Experiment 2: Two-Agent Common Pool Resource Game

### 5.1 Problem Description

我们进一步测试 NSEAP 在多主体公共资源博弈（公地悲剧）中的表现：

- 两个智能体共享同一个鱼塘
- 总承载力 K = 2000
- 总临界点 = 300
- 每个智能体只能观察到当前总鱼群数量和自己上一轮的捕捞量
- 如果总鱼群跌破临界点，双方一起崩溃
- 公地悲剧：个体贪心 → 集体过度捕捞 → 系统崩溃

### 5.2 Compared Combinations

我们测试四种对称组合：

1. Greedy vs Greedy
2. Conservative vs Conservative
3. QLearning vs QLearning
4. NSEAP vs NSEAP

### 5.3 Results

Table 3: 两智能体实验结果（100次运行）

| Combination | Avg Survival Cycles | Avg Total Harvest (Both) | Collapse Rate |
|-------------|---------------------|--------------------------|---------------|
| Greedy vs Greedy | 1.0 | 1,203 | **100.0%** |
| Conservative vs Conservative | 200.0 | 20,000 | 0.0% |
| QLearning vs QLearning | 34.5 | 5,170 | **100.0%** |
| **NSEAP vs NSEAP** | **200.0** | **0** | **0.0%** |

Figure 2: 两智能体结果对比  
![Comparison](results_two_agent/two_agent_comparison.png)

### 5.4 Discussion

结果非常有意思：

1. **Greedy vs Greedy**：双方都最大化短期收益，第一周期就崩盘，完美演绎公地悲剧
2. **Conservative vs Conservative**：双方都固定低捕捞，永久可持续，但总收益很低
3. **QLearning vs QLearning**：探索过程中必然过度捕捞，几乎 100% 崩溃
4. **NSEAP vs NSEAP**：初始不确定性极高 → 双方都选择"不捕捞" → 零崩溃，零收获

这个结果**完全符合 NSEAP 的设计原则**：

- 当不确定性超过阈值，NSEAP 优先保证不崩溃，收益其次
- 零收获零崩溃不是失败，这是 NSEAP 在极端不确定性下的理性选择
- 如果想要更高收获，可以调低初始不确定性阈值，但会增加崩溃风险——这是一个清晰可解释的权衡

## 6 Discussion

### 6.1 Why NSEAP Works

NSEAP 在复杂临界系统中比传统 RL 更鲁棒，原因在于：

1. **探索方式不同**：传统 RL 是**随机探索**，NSEAP 是**保守试探**——一次只走一步，能回来才固化，大大降低崩溃风险

2. **本体来源不同**：传统符号AI本体是**专家预设**，深度学习本体**涌现于黑箱**，NSEAP 本体**从交互中动态生长**——错了可以证伪丢弃，对了可以固化，逐步逼近真实

3. **不确定性处理不同**：传统 RL 把不确定性放在概率里，NSEAP 把不确定性放在**拓扑结构**里——不确定的路径就是不能闭环，不走就是了

### 6.2 Limitations and Future Work

本文是原型验证，存在局限：

1. **状态空间**：当前只处理一维状态（鱼群数量），未来可以扩展到高维状态
2. **行动空间**：当前是离散四个等级，未来可以扩展到连续行动
3. **对比**：未来可以和更多现代安全 RL 方法（如 constrained RL、Bayesian RL）对比
4. **缩放**：测试更大规模问题，验证小网生长的 scalability

未来研究方向：
- 将 NSEAP 应用到真实世界复杂系统决策问题（如电力调度、渔业管理、投资组合管理）
- 研究不同闭环深度、确认阈值对性能的影响
- 结合神经网络，用 NSEAP 做高层符号推理，神经网络做底层感知
- 研究 N 智能体（N > 2）公共资源博弈中的合作演化

## 7 Conclusion

本文提出 NSEAP，一种面向复杂临界系统的新型智能体架构，核心是悬置层+小网+实用闭环三层设计。在鱼塘生态管理和公共资源博弈两个基准问题上的实验表明：

- NSEAP 能在未知临界点的复杂系统中安全学习，实现零崩溃
- 在保证安全的前提下，能显著提高收益相比保守固定策略
- 在初始不确定性极高时，自动选择最保守策略，严格遵守不确定性管理优先原则

NSEAP 为复杂系统决策打开了一条新路：不追求一步到位的全局最优，通过保守试探、闭环验证、动态生长，逐步积累可靠知识，在保证系统不崩溃的前提下最大化收益。

---

## References

[1] Scheffer M, Carpenter S, Foley J A, et al. Catastrophic shifts in ecosystems[J]. Nature, 2001, 413(6856): 591-596.

[2] May R M. Stability and complexity in model ecosystems[M]. Princeton University Press, 1973.

[3] García J, Fernández F. A comprehensive survey on safe reinforcement learning[J]. Journal of Machine Learning Research, 2015, 16(1): 1437-1480.

[4] Gruber T R. A translation approach to portable ontology specifications[J]. Knowledge Acquisition, 1993, 5(2): 199-220.

[5] Gödel K. Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I[J]. Monatshefte für Mathematik und Physik, 1931, 38(1): 173-198.

[6] Altman E. Constrained Markov decision processes[M]. CRC Press, 1999.

[7] Mihatsch O, Neuneier R. Risk-sensitive reinforcement learning[J]. Machine Learning, 2002, 49(2-3): 173-195.

[8] Ghavamzadeh M, Mannor S, Pineau J, et al. Bayesian reinforcement learning: A survey[J]. Foundations and Trends in Machine Learning, 2015, 8(5-6): 359-483.

[9] Garcez A D, Gabbay D M, Broda K B. Neural-symbolic learning systems: foundations and applications[M]. Springer Science & Business Media, 2002.

[10] McCarthy J. Artificial intelligence, logic and formalizing common sense[J]. Philosophical Logic and Artificial Intelligence, 1989: 161-190.

[11] Stojanovic G, Maedche A, Motik B, et al. User-driven ontology evolution process[C]//International Conference on Ontologies, Databases, and Applications of Semantics for Large Scale Information Systems. Springer, 2002: 1-12.

[12] Noy N F, Klein M. Ontology evolution: Not the same as schema evolution[J]. Knowledge and Information Systems, 2002, 4: 199-219.

[13] Husserl E. Ideas Pertaining to a Pure Phenomenology and to a Phenomenological Philosophy[M]. Springer, 1913.

[14] Peirce C S. How to make our ideas clear[J]. Popular Science Monthly, 1878, 12: 286-302.

---

## Acknowledgments

Thank you to Claude Code for assisting with code implementation and experiment running.
