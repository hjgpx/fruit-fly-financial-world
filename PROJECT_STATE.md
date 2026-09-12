# PROJECT_STATE.md — Project Save Point

> **任何拿到本仓库 + 本文档的 AI，请先读完此文件再动手。**
>
> 本文是项目截至 **2026-09-12 17:10 GMT+8** 的完整存档快照。
> 它已经包含：项目目标、不可违反的原则、数据版本、关键文件、已完成的全部实验与数值结果、发现并修复的 bug、已被否定的路线、MaleCNS / KC–MBON–DAN / NT 审计的核心结论、当前的技术判断，以及下一步要做的工作。
>
> 如果你是从零开始的新会话，只需对本文件说"继续"即可，**不需要重讲整个故事**。

---

## 0. 项目目标

**核心目标**：搭建一个让"果蝇连接组在金融市场中生存与自我进化"的科研环境。整条链路是：

```
BTC-USDT 1h K 线（OHLCV）
  → 因果归一化为 1,624 个真实果蝇感觉神经元（R7p/R7y/R8p/R8y 组等）的脉冲
  → MaleCNS v1.0 连接组（162,517 有效神经元、25,120,209 边）做 K 步线性传播
  → MBON（97 个）层做 BUY/SELL 二分类读出
  → 与后续价格变化比对，得到 reward
  → 用奖励调制的 Hebbian 学习改 P 中的部分权重
  → 在不删除已有软件的前提下，反复迭代，让策略存活
```

> 现阶段只走到了**链路的物理管道与诊断阶段**：传播可以跑、读出可以给决策、学习回路骨架存在（v0），但**真实学习尚未启用**（用户在每一轮都明确说"不学习、不 renorm、不实现 LIF、不改 P"）。

---

## 1. 不可违反的原则（所有未来 AI 必须照做）

### 1.1 用户操作原则
1. **不得删除任何软件和文件**。所有清理操作必须用"临时脚本 + 用完即删"的模式，仓库内不能留临时文件。
2. **保守谨慎**：能只读不写就只读；能不改正式代码就不改。
3. **接管而非重装**：用户已登录的浏览器会话、已安装的工具、已下载的数据**一律保留**，不重新部署。
4. **Bash 10 分钟硬杀**：任何 Bash 命令 `timeout` 上限 600000 ms（10 分钟）；长任务必须 `run_in_background=true` 轮询或拆分。

### 1.2 研究方法原则
5. **一次加一条规则，拒绝范围扩展**。每次实验只引入一个新变量（一个 NT 字段、一种 NT 符号、一个传播步数、一种注入通道）。
6. **严格遵守既定参数**：用户给定的 seed、threshold、step 数、文件路径**一字不改**。
7. **绝不臆测数据**：缺什么查什么，找不到就明说"找不到"，不能凭感觉编。
8. **临时脚本 = 一次性工具**：写在一个临时文件里，跑完即 `rm`，不进版本控制。

### 1.3 神经科学原则（项目特别声明）
9. **不实现 LIF**：所有动力学仅基于原始 `P_csr` 做**线性传播**或**带符号线性传播**；不引入积分-发放神经元。
10. **不擅自赋 NT 符号**：`ACh/GABA/Glu/DA/5HT/OA/HA/unclear` 在不同阶段可能给不同符号，但**仅在用户明确指令下**才允许；任何"自行决定兴奋/抑制"的建议都必须先得到用户授权。
11. **不修改 P_csr**：连接组权重来自官方 MaleCNS v1.0 的清理版本（25,120,209 条去重边），对它做的是 `tocsc + 列乘符号` 这种**只读改写**（不落盘）。
12. **不 renorm**：传播后的 L1 质量自然衰减（如 1.0 → 0.42 → 0.14–0.17），不要重新归一。
13. **数据覆盖率优先于精确度**：神经科学上"有没有数据"比"有多准"更重要，做任何动力学决策前必须先确认源数据的覆盖范围与字段一致性。

### 1.4 沟通/工作流原则
14. **结构化提示词**：用户偏好"编号分节 + 显式规则 + 精确定义与公式"的提问方式，新会话的提问应保持这一风格。
15. **每次实验结尾只回答一个问题**：用户经常在 prompt 末尾给出"最后只回答：XXX"——这就是**唯一**的回答目标，中间过程数据可以给，但最终段落必须聚焦这一个答案。
16. **工作日志每日追加**：`.workbuddy/memory/YYYY-MM-DD.md` 是 APPEND-ONLY，写完一个实验立即追加一段。

---

## 2. 硬件 / 软件 / 路径

### 2.1 硬件
- CPU：双路 Xeon E5-2690 v2（约 10c/20t × 2 = 40 核 80 线程，但 NUMA 与 Windows NUMA 拓扑对 Numba 而言实际可用 ~20 线程并行）
- 内存：48 GB DDR3
- OS：Windows 10 1909（中文环境）

### 2.2 软件栈
- Python 3.11.9（必须使用 `.venv/Scripts/python.exe`）
- numpy / scipy / numba / pandas / pyarrow（无 pytest，用 `unittest`）
- git 客户端在 PortableGit 下，无凭据存储（GCM 安装但空），push 需每次提供 token
- curl（Git Bash 自带）支持 `-C -` 断点续传与 HTTP Range

### 2.3 关键路径（绝对路径，便于复制）
```
项目根目录       C:\Users\User\WorkBuddy\2026-09-11-20-43-17\fruit-fly-financial-world
Python venv      .\fruit-fly-financial-world\.venv\Scripts\python.exe
源数据           .\data\raw\malecns_v1.0\
处理后数据       .\data\processed\v1\   （边缘列表 nodes.feather / edges.feather / P_csr.npz）
市场数据         .\data\market\BTC-USDT_1h.csv
配置文件         .\config\market_input_map.json   .\config\action_map.json
源代码           .\src\market_encoder.py   .\src\brain_runner.py   .\src\learning_loop.py
测试             .\tests\test_market_encoder.py   .\tests\test_learning_loop.py
文档             .\docs\PROGRESS.md   .\README.md   .\PROJECT_STATE.md（本文件）
当日工作日志     C:\Users\User\WorkBuddy\2026-09-11-20-43-17\.workbuddy\memory\2026-09-12.md
```

---

## 3. 数据版本与规模

### 3.1 MaleCNS v1.0（核心数据）
- 来源：`https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`
- 已下载文件：
  - `body-neurotransmitters-male-cns-v1.0.feather`（**43,282,834 字节**，约 42 MB，1,835,518 行）
    - 字段：`body / cell_type / total_nt_predictions / predicted_nt_confidence / predicted_nt / ground_truth / celltype_total_nt_predictions / celltype_predicted_nt / celltype_predicted_nt_confidence / consensus_nt`
  - 其他（nodes、edges、neural\_annotations、classification 等）从 desktop-fly 项目的 `etl_malecns.py` 派生，**未下载 2.7 GB 的 tbar-neurotransmitters**
- **未下载**：2.7 GB 的 `tbar-neurotransmitters-male-cns-v1.0.feather`（用户明令禁止下载这个大文件）

### 3.2 我们的有效子集（**当前所有实验都在这个子集上做**）
- 神经元（valid）：**162,517 个**，其中：
  - R7p 332 / R8p 330 / R7y 481 / R8y 481 （**总计 1,624 个视觉输入神经元**，与 4 路市场信号 1:1 对应）
  - KC 4,064（15 个 type，如 KCg-d、KCab、KCg-s3 等）
  - MBON 97（37 个 type，**无 MBON08**——官方分类里就没）
  - PAM 316（PAM01–15）
  - PPL1 16（PPL101–108，每个 type 各 2 个）
- 边（valid）：**25,120,209 条**（已去重、`body_pre → body_post`）
- P 矩阵：`P_csr.npz`，shape = **162,517 × 162,517**，float32，**列归一化**（每列和 = 1，对应 `x = P @ x` 的传播约定）
- 索引：与 `nodes.feather.bodyId` 一一对应，用 `MarketEncoder().body_ids` 拿四组感觉元

### 3.3 关键官方字段语义（不要混用）
| 字段 | 含义 | 数据量级 |
|---|---|---|
| `predicted_nt` | neuron-level 神经递质预测（机器学习） | 全集 ~57.7% ACh、17.2% Glu、12.3% GABA、8.5% unclear、2.7% DA |
| `celltype_predicted_nt` | cell type 级别聚合后的预测 | 几乎与 consensus 一致 |
| `consensus_nt` | 综合类型 + 文献 + 分类器的最终判断 | **优先使用**，覆盖率 100%（62,517 valid 中 60 个无 NT 行） |
| `ground_truth` | 有文献/实验支持的 NT（覆盖约 51% 神经元） | 当有值时，与 consensus 100% 一致 |
| `predicted_nt_confidence` | neuron-level 分类置信度 | median 0.9348、min 0.1869、max 0.9754（**不设阈值**，语义有陷阱） |

---

## 4. 关键文件清单（每个文件"做什么 / 改过什么 / 怎么调用"）

### 4.1 源代码（**正式代码**，用户禁止修改）

#### `src/market_encoder.py`
- 作用：把 OHLCV 因果归一化为 1,624 个感觉神经元的脉冲（每根 K 线对应一个 1,624 维向量）
- 关键 API：
  - `MarketEncoder(market_csv_path, market_input_map_path)`
  - `.encode_step(features) -> np.ndarray`（1,624 维）
  - `.body_ids`（dict，按 `cfg["rules"]["market_signals"]` 中的组名 → set[int]）
- 注入通道与身体组映射（来自 `config/market_input_map.json`）：
  ```json
  {
    "price_up":    ["R7p", "R8p", ...],
    "price_down":  ["R7p", "R8p", ...],
    "volume":      ["R7y", "R8y", ...],
    "volatility":  ["R7y", "R8y", ...]
  }
  ```
  **注**：`market_input_map.json` 的最外层是 `rules.market_signals`（组名 → 通道名），`bodyIds` 的键是通道名（不是组名）——这是个反复踩坑的地方，新代码读这里要小心。

#### `src/brain_runner.py`
- 作用：5 步传播 + BUY/SELL 读出
- 核心实现：`x_next = P_csr @ x`（**列归一 P，左乘 = 突触后加权**，符合 MaleCNS 默认约定）
- 读出：把 MBON 97 个神经元的活动分别聚合到 BUY/SELL 两个总和，做 argmax
- Numba 并行 `@njit(parallel=True, fastmath=True)`，约 20 线程

#### `src/learning_loop.py`
- 作用：v0 的奖励调制 Hebbian 学习骨架（存在但**未被真实信号驱动过**）
- 包含 `step(...)` 和按 reward 调 `Δw = η * reward * (x_pre * x_post - decay * w)` 的入口
- **当前没有把 BTC 真实价格接到 reward**，学习回路处于"代码通、信号未接"状态

### 4.2 测试
- `tests/test_market_encoder.py`：unittest 覆盖因果、归一、组分布
- `tests/test_learning_loop.py`：unittest 覆盖 Hebbian 更新方向、衰减、reward 符号

### 4.3 配置 / 文档
- `config/action_map.json`：BUY/SELL/HOLD 的标签索引
- `docs/PROGRESS.md`：进展报告（本次推送上一版）
- `README.md`：项目定位（private repo，提交历史可见）

---

## 5. 已完成的实验（按时间顺序，所有跑过的诊断都在这里）

> 注：步骤编号只在当次会话内计数；下面的顺序是**跨多轮会话**的总次序。

### 实验 1：服务器体检（早期，与本仓库无直接关系）
- 验证 numba/多线程在大矩阵上的吞吐，确认 5 步全脑传播能在合理时间内跑完

### 实验 2：链路最小化搭建
- `market_encoder.py` + `brain_runner.py` 跑通 BTC 历史，给出 BUY/SELL 决策（不学习，纯前向）
- 测试覆盖 causal/normalize/Hebbian 三个关键不变量

### 实验 3：Mushroom Body 学习回路清点
- **发现**：MB 关键元件**全部天然存在**
  - KC 4,064（15 type）
  - MBON 97（37 type，无 MBON08）
  - PAM 316（PAM01–15）
  - PPL1 16（PPL101–108 各 2）
  - KC → MBON 边数：**61,210**（剪枝前）
- 主要 type pair：PAM01–PAM15 全部存在；PPL101–PPL108 全部存在；MBON07 / MBON11 / MBON13 等 key 类型全部存在
- **回答"天然存在 KC–MBON–DAN 局部奖励学习回路"**：是的，规模可观

### 实验 4：KC 稀疏性诊断
- 设置：10 个随机 candle、seed=42、原始 P、5 步、无学习无 renorm
- **结果**：
  - step1–2：KC 几乎全零（信号尚未到达）
  - step3：KC 100% 点亮（x > 0）
  - 严格阈值下 KC 极稀疏：>1e-6 仅 6%–11%，>1e-9 仅 ~15%
- **回答"5 步扩散后 KC 是稀疏还是全亮"**：从信息角度看**全亮**，从动力学角度看**严格稀疏**

### 实验 5：KC pattern separability（980 根全量）
- 设置：980 根 candle，定义 S2（step2）、S3t（step3 阈值）、S3top（step3 top-K）
- 指标：Jaccard 集合相似度
- **结果**：
  - 跨蜡烛 / 跨市场状态的 KC top-K Jaccard = **0.87–0.95**（**几乎是同一批 KC**）
  - per-KC 频率 ≈ 1.0（前 20% 的 KC 几乎每次都亮）
- **回答"KC 表征是否随经历变化"**：**没有**——静态结构稀疏，不是动态模式稀疏

### 实验 6：Single-channel KC fingerprint（四路感觉分别注入）
- 设置：仅 price_up / price_down / volume / volatility 各注入总质量 1.0，step2/3
- **结果**（top 5%）：
  - 通道两两 Jaccard step3：**0.71–0.93**
  - **up vs down J = 0.92（高度同质）**
  - 四通道共同核心 KC 集：其中 **KCg-d 占 94.5%**
- **回答"四路市场感觉是否被压成相同 fingerprint"**：**是的**，上游解剖收敛是根因

### 实验 7：我们自己 annotation 的 NT/sign audit（**结论错误，已被实验 8 修正**）
- 错误结论："数据里没有 NT 字段，兴奋/抑制动力学缺乏数据支持"
- 用户纠错：NT 在独立文件 `body-neurotransmitters-male-cns-v1.0.feather` 里

### 实验 8：官方 neurotransmitter 审计（**覆盖率 audit**）
- 操作：下载 43 MB 的 NT 文件；做只读 join（valid 神经元 162,517 ↔ NT 表 1,835,518 行）
- **结果**（这是覆盖率答案，单点不变量）：
  - **神经元覆盖率：162,457/162,517 = 99.96%**（仅 60 个神经元无 NT）
  - **边覆盖率：25,120,209/25,120,209 = 100.00%**（所有突触前 NT 都可查到）
  - confidence：median 0.9348，p10 0.7349，min 0.1869，max 0.9754（**不设阈值**）
  - 各类 NT 计数（valid）：ACh 93,804（57.7%）/ Glu 27,863（17.2%）/ GABA 20,031（12.3%）/ unclear 13,753（**8.5%**）/ DA 4,443 / HA 2,015 / 5HT 450 / OA 98
- **回答"覆盖率有多高"**：**神经元 99.96%，边 100%**，兴奋/抑制动力学在信息层面完全可行

### 实验 9：NT reliability audit（**三字段一致性**，关键决策实验）
- 对象：`predicted_nt` / `celltype_predicted_nt` / `consensus_nt` / `ground_truth` 与对应 confidence
- **总体一致率**（162,457 个有 NT 行的神经元）：
  - `predicted == consensus` = **88.78%**（144,235）
  - `predicted == celltype` = 92.00%
  - `celltype == consensus` = **96.68%**
  - 三字段全一致 = 88.76%
- ground_truth 覆盖 83,496 个神经元（约 51%），与各字段一致率：
  - `ground_truth == consensus` = **100.00%**（**consensus 已完整吸收 ground_truth**）
  - `ground_truth == celltype` = 99.22%
  - `ground_truth == predicted` = 90.57%
- **KC（4,064）的分歧最严重**：
  - `predicted` 神经元级：dopamine 4,058 / GABA 1 / unclear 5
  - `celltype_predicted`：dopamine 4,062
  - **`consensus_nt`：acetylcholine 4,064（100%）** ← 与 predicted 完全相反
  - KC ground_truth = **0**（**没有任何文献真值**）
- **MBON 97 / PAM 316 / PPL1 16**：
  - MBON：consensus ACh 50 / GABA 21 / Glu 26；65 个有 gt 全部 == consensus；pred↔cons 95/97
  - PAM：四字段全部 dopamine（311+5=316），gt 316/316
  - PPL1：四字段全部 dopamine，gt 16/16
- **R7p/R8p/R7y/R8y**：
  - neuron-level 大量 unclear（R7y 86% unclear、R7p 76% unclear）
  - 但 `celltype` / `consensus` / `ground_truth` **全部 histamine**，无冲突
- **高 confidence 冲突的真实硬错误仅 30 个**：LC30 × 15（pred ACh vs gt Glu）、AN09B017 系列 × 10、Lai × 1、IN11B018 × 2
- **最大系统性分歧：KC dopamine → consensus ACh 的 4,058 个**（KC 置信度中位数 ~0.93，属于"自信地错"）
- **最终结论**：**neuron-level predicted_nt 不能直接作为动力学依据，必须优先层级 `ground_truth > consensus_nt > celltype_predicted_nt > predicted_nt`**

### 实验 10：Signed linear propagation diagnostic（**本会话刚做完**）
- 设置：在 `P_csr`（列归一）上对每条边乘 `sign_vec[pre]`，符号来自 `consensus_nt`：
  - acetylcholine = +1；gaba = -1；histamine = -1；**glutamate = 0 / dopamine = 0 / serotonin = 0 / octopamine = 0 / unclear = 0**
- 注入：四通道分别注入总质量 1.0；传播 step 1–3；无学习、无 renorm
- **符号施加统计**：
  - 神经元符号（valid）：**+1 = 102,576 / −1 = 27,767 / 0 = 32,174**
  - 出边权重：正 102,423 / 负 26,989 / 置零 32,051（**约 20% 传播质量被置零**，主要是 glutamate=0）
- **KC 激活分布**：
  - step1：KC 全零（视觉信号未到）
  - step2：仅 2.9–4.3% KC 非零，**几乎全为负**（信号经 histamine 感觉元被翻转）
  - step3：全部 4,064 KC 非零；正 2,435–2,572 / 负 1,492–1,629；幅值比 unsigned 小一个数量级
- **top 5%（按 abs）两两 Jaccard**（**核心数字**）：

| 通道对 | signed step3 | unsigned step3 |
|---|---|---|
| up–down | 0.906 | 0.915 |
| up–volume | 0.692 | 0.758 |
| up–volatility | 0.789 | 0.897 |
| down–volume | 0.713 | 0.706 |
| down–volatility | 0.804 | 0.915 |
| volume–volatility | 0.773 | 0.742 |
| **六对均值** | **0.78** | **0.82** |

- **主导性**：signed top5% 中 **KCg-d 占 86–92%**；四通道交集 151 个 KC，其中 **KCg-d 143 个**（unsigned 交集 165 个，KCg-d 156 个）——主导性几乎没变
- **数值稳定性**：L1 质量 1.0 → 0.42 → 0.14–0.17（衰减来自置零列，不来自爆炸）；max\|x\| ~1e-3；KC 层 |sum|/sum|x| = 0.86–0.98（无大规模抵消）
- **回答"符号是否已显著改善 KC 层可分性"**：**没有**——同质化根因是解剖收敛，不是符号问题
- 涉及代码：临时脚本 `tmp_signed_prop.py`（已删）；正式 `brain_runner.py`、`learning_loop.py` **均未动**
- 完整记录：`.workbuddy/memory/2026-09-12.md` 末尾段

---

## 6. 所有重要数值结果汇总表

| 维度 | 数值 |
|---|---|
| 总神经元（valid） | 162,517 |
| 总边（valid） | 25,120,209 |
| KC 数量 | 4,064 |
| MBON 数量 | 97 |
| PAM 数量 | 316 |
| PPL1 数量 | 16 |
| 视觉输入四组合计 | 1,624（R7p 332 + R8p 330 + R7y 481 + R8y 481） |
| KC → MBON 边数 | 61,210 |
| KCg-d 在四通道共同核心中占比（top5%） | 94.5% |
| KCg-d 在 signed 四通道共同核心中占比 | 86–92% |
| NT 神经元覆盖率 | 162,457 / 162,517 = 99.96% |
| NT 边覆盖率 | 25,120,209 / 25,120,209 = 100% |
| NT 三字段一致率 | pred↔cons 88.78% / celltype↔cons 96.68% / 三全一致 88.76% |
| NT 中 ACh 占比 | 57.7% |
| NT 中 unclear 占比 | 8.47% |
| KC predicted=dopamine 中保留为 dopamine 的（consensus） | 0 / 4,058 |
| confidence 中位数 | 0.9348 |
| KC 置信度中位数 | ~0.93 |
| KC step3 严格稀疏度（x>1e-6） | 6–11% |
| KC step3 信息稀疏度（x>0） | 100% |
| 四通道 top5% Jaccard step3（unsigned）均值 | 0.82 |
| 四通道 top5% Jaccard step3（signed consensus）均值 | 0.78 |
| up vs down Jaccard step3（unsigned / signed） | 0.915 / 0.906 |
| 高 confidence 且 pred↔cons 冲突总数 | 5,864（其中 5,877 个是"unclear 但高置信"） |
| 真正硬错误（高置信且 pred vs ground_truth 冲突） | 30 个 |

---

## 7. 发现并修复的 bug / 踩过的坑

| # | 问题 | 修复 |
|---|---|---|
| 1 | curl 5 分钟超时只下到 54% | 用 `curl -sS -L -C -` 断点续传，HTTP 206 补完 |
| 2 | `pyarrow.feather.read_schema` 报错 | 改用 `pf.read_table(path).schema` |
| 3 | 第 6 节脚本输出淹没核心数字 | 拆成精简汇总脚本只打聚合 |
| 4 | `market_input_map.json` 键是通道名、组名在 `rules.market_signals` 里 | 在新代码读这里要做两层穿透 |
| 5 | 2-hop sensory→KC 路径早期取错 `pre` | 修正 `mid = post` |
| 6 | 第 61 轮我错误回答"数据里没有 NT" | 用户纠正、下载官方独立 NT 文件、加入覆盖率审计 |
| 7 | 第 7 轮细分类脚本因 25M 边扫描过久 | 拆为 `tmp_nt_part6.py` 单独跑 |
| 8 | **GitHub 推送**：网络时通时断；fine-grained PAT 未授权本仓库；密码已被 GitHub 禁用 | 改用 classic PAT（`ghp_` + `repo` scope）；每次 push 需重新提供 token（本机无凭据存储） |
| 9 | `.gitconfig` 在沙箱内无法锁定（无写权限） | 改用仓库级 `git config user.name/email` |
| 10 | `.venv/Scripts/python.exe` 路径 | 所有 Python 调用必须用绝对路径，不能用裸 `python` |

---

## 8. 已被否定的路线（不要重新尝试）

| # | 路线 | 否决理由 |
|---|---|---|
| A | 把 `predicted_nt` 直接作为动力学依据 | 与 consensus 仅 88.78% 一致，KC 系统性错，**ground_truth > consensus > celltype > predicted** 才有保障 |
| B | 用 unsigned P 做市场可分性的最终解释 | 四通道 top5% J 均值 0.82、up-down 0.915，几乎没区分度 |
| C | 把 glutamate 默认置为兴奋（+1）/ GABA 默认置为抑制（−1） | 用户明令"不擅自赋 NT 符号"；本轮诊断选 glutamate=0 是临时规则，不是最终模型 |
| D | 实现 LIF（积分-发放）神经元 | 用户明令"不实现 LIF" |
| E | 传播后做 renorm | 用户明令"不 renorm" |
| F | 用 `tmp_*.py` 留下的脚本 | 用完即删的临时脚本不进仓库；正式代码在 `src/` |
| G | 下载 2.7 GB 的 `tbar-neurotransmitters` 文件 | 用户明令"不下载 tbar 大文件" |
| H | 用 Bash 跑超过 10 分钟的任务 | 会被工具强制杀，必须 `run_in_background` 或分块 |
| I | 用 GitHub 密码 push（`Amtech888*`） | GitHub 自 2021-08 起**禁用密码认证 git 操作**；必须 PAT |
| J | 用细粒度（fine-grained）PAT 做全仓库推送 | 容易忘记在"Repository access"里勾选该仓库，导致 403；classic + `repo` scope 更省事 |

---

## 9. 当前技术判断（综合所有诊断的最终结论）

1. **数据足够支撑动力学**：覆盖率 99.96%（神经元）和 100%（边），但只能信任层级 `ground_truth > consensus_nt > celltype_predicted_nt > predicted_nt`，**绝不能直接用 predicted_nt**。
2. **KC 层的同质化是结构性问题，不是符号/动力学问题**：
   - 四通道 top5% J 0.82、up-down 0.915
   - signed 后也只降到 0.78 / 0.906（边际改善）
   - 共同核心几乎全是 KCg-d
   - **根因是上游解剖结构（谁连到 KC），不是突触权重符号**
3. **下一步的有效杠杆在输入路由 / 门控 / 结构层面**，不是继续调符号：
   - 候选：① 加入侧抑制让 KC 互斥（增加 top-K 可分性）
   - 候选：② 给不同通道注入到不同 KC 子群（人为路由，打破共同核心）
   - 候选：③ 在 layer-1 / 2 加入 channel-selective gating（让不同通道走不同中介神经元）
4. **学习回路（learning_loop.py）目前是骨架**：未真实连接 BTC 价格变化到 reward，**不应当前开启学习**——会得到噪声。
5. **可分性 > 正确性优先级**：在 KC 没分得开的现实下，先解决"区分市场状态"，再谈"用 NT 修正符号"。
6. **网络受限原则不变**：每次 push 必须临时提供 PAT；任何长 Bash 任务用 background 模式。

---

## 10. 下一步真正要做的事（**注意：实验 10 已在本会话刚做完**，下面是更新后的待办）

### 优先级 1：构造可分性（必须先于此做学习）

#### Task A.1 — 输入路由分流
- 把 R7p/R8p（强方向选择性）→ KCab、KCa'b' 等"非 g-d"型 KC；保留 R7y/R8y → KCg-d
- 目标：打破 up/down 共同核心；预期 up–down J 降至 0.5 以下
- 不实现 LIF、不加 NT 符号、不改 P 的**符号版本**——只改 `config/market_input_map.json` 的**路由**，并把注入按通道拆分（每通道总质量 1.0）
- **先做单次只改路由的诊断，验证 J 是否真的下降**

#### Task A.2 — KC 侧抑制（横向抑制）
- 在传播 step 3 后，对 KC 层做一次 top-K winner-take-all：保留前 5% 的 KC、把其余置零
- 不改 P，不引入递归；是**后处理**信号
- 预期：四通道 fingerprint 边界更清晰

#### Task A.3 — Channel-selective gating（结构层面）
- 在 P 上加一层 1,624 → 1,624 的"通道门"稀疏矩阵 `G_csr`，让 4 路市场信号被独立门控后再到 KC
- G 初始为单位矩阵的某种 1-hot 化变体——**只调 G 不调 P**
- 预期：四通道的 KC 激活图谱应该变得不一样

### 优先级 2：NT 符号规则修订

#### Task B.1 — 把 glutamate 从 0 改成 −1（待用户授权）
- 当前 glutamate = 0 导致 17.2% 神经元被置零、~20% 边质量归零
- 如果改成 −1，会减少质量损失，但需要用户先确认（这是对 NT 的擅自赋号，需要明令）

#### Task B.2 — KC 的特殊处理
- 实验 9 显示：KC 的 `consensus_nt` 强制为 acetylcholine（+1），但官方 neuron-level 把它判为 dopamine
- 是否引入"类型覆盖"：当 type 属于 KC 时强制覆盖为 ACh？这与 `consensus_nt` 已经做的事一致，无需额外操作
- 是否在 NT 字段为空时回退到 cell-type 默认？这是 audit 中 60 个神经元的情况，目前暂不处理

### 优先级 3：连接学习
- learning_loop.py v0 框架已有，等可分性变好后再接到 BTC 价格变化
- **严格禁止在同一次任务里既改可分性又开启学习**——会变成两件事都说不清

---

## 11. 完整通讯 / 操作约定（与新 AI 对话时直接照搬）

> 下面这段是给**接手的 AI**的一段提示词模板，可以原样发出：

```
读取这个仓库，先看 PROJECT_STATE.md，我们继续。
```

> 接手的 AI 读完 PROJECT_STATE.md 后，应当：
> 1. 复述"已确认：最新 HEAD = 9389bf8（如果你看到更新，以仓库实际为准）"
> 2. 复述"我当前看到：实验 1–10 已完成，下一步是 Task A.1（输入路由分流）或你在你说的"
> 3. 等用户给出新的精确指令，再开始操作
> 4. **不要重复跑任何已完成的实验**；**不要重新下载数据**；**不要修改正式 `src/` 代码，除非用户明令**

---

## 12. 元数据

- 撰写时间：2026-09-12 17:10 GMT+8
- 撰写人：本会话 AI（WorkBuddy）按用户指令
- 触发原因：用户在 2026-09-12 17:09 GMT+8 提出"做个存档点，新 AI 只看这个就能接上"
- 适用范围：fruit-fly-financial-world 仓库当前 main 分支
- 更新策略：每次有重大新实验 / 新增数据 / 新路线被否决时，在本文件相应小节追加，旧段落保留不删

---

**致接手者**：如果你读到这里并准备继续，**请先确认你理解了第 1 节（不可违反的原则）和第 9 节（当前技术判断）**，这两者是这个项目之所以走到今天的根本原因。然后告诉我你想做 Task A.1、Task A.2、Task A.3 还是其他自定义扩展——我会基于本文进入下一步。
