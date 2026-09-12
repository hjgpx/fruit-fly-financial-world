# 工作进展报告 — 果蝇在金融市场中的生存与自我进化

> 更新日期：2026-09-12 ｜ 分支：main ｜ 阶段：连接组只读诊断完成，学习循环 v0 已就绪

## 0. 项目目标

把完整的雄性果蝇脑连接组（MaleCNS v1.0，162,517 个神经元、25,120,209 条突触边）作为一个固定的"身体"，让它在金融市场中：

1. **感知**：把行情（涨/跌/量/波动率）编码进 1,624 个视觉感觉神经元；
2. **决策**：信号在固定解剖拓扑中传播，由 BUY/SELL 运动神经元群读出方向；
3. **学习**：通过 reward-modulated Hebbian 可塑性（权重非负、列归一化不变量、拓扑不动）实现自我进化；
4. **生存**：在不引入未来信息、不人工选神经元、不调阈值的前提下，检验连接组能否学会在市场中活得下去。

核心约束（贯穿所有阶段）：**拓扑不变、权重非负、决策与奖励严格因果（0% 未来信息）、0% 人工先验**。

## 1. 环境与硬件

| 项目 | 配置 |
|---|---|
| 机器 | 双路 Intel Xeon E5-2690 v2（20C/40T）、48GB DDR3、Windows 10 1909 |
| Python | 3.11.9 + numpy / scipy / numba / pandas / pyarrow（venv 隔离） |
| 测试 | unittest（项目未引入 pytest） |

## 2. 已完成的工作

### 阶段 A — 硬件基准（已完成，2026-09-11）

- `benchmark_v0.py`：SciPy 单线程 CSR 稀疏传播规模扫描；
- `benchmark_v1.py`：Numba 并行 SpMV，结论 **20 线程最优**，单步传播 162,517×162,517 矩阵在毫秒级，可支撑逐 K 线在线推理；
- 结果见 `docs/benchmark.md`。

### 阶段 B — 连接组数据管道（已完成）

- 下载 MaleCNS v1.0 官方数据（`connectome-weights ... minconf-0.5.feather`，1.0GB）；
- 处理产出 `data/processed/v1/`：
  - `nodes.feather`：162,517 个 valid 神经元（bodyId + type）；
  - `edges.feather`：25,120,209 条突触边；
  - `P_csr.npz`：162,517² 列归一化传播矩阵（P[post, pre]，float32）；
  - `body_ids.npy`：bodyId → 矩阵索引映射。
- 大文件按 `.gitignore` 约定不入库，仅本地保留。

### 阶段 C — 感觉编码器 `src/market_encoder.py`（已完成）

- BTC-USDT 1h K 线 → 4 路因果信号：
  - price_up → R7p（332 个）、price_down → R8p（330 个）、volume → R7y（481 个）、volatility → R8y（481 个）；
- 归一化使用**因果 percentile rank**（最近 100 根、不含当前根、历史不足 20 根输出 0），修改未来 K 线不会改变过去信号；
- 有完整单元测试（`tests/test_market_encoder.py`）。

### 阶段 D — 传播推理 `src/brain_runner.py`（已完成）

- 每根 K 线从零状态开始，4 路信号平均分配写入感觉神经元；
- 列归一化 P 矩阵连续传播 STEPS=5 步（Numba 20 线程）；
- BUY（36 神经元）/ SELL（164 神经元）按 type-mean 等权聚合，direction = (BUY−SELL)/(BUY+SELL)；
- 不定义 HOLD，不写数据文件；有单元测试覆盖。

### 阶段 E — 学习循环 v0 `src/learning_loop.py`（已完成）

- reward-modulated Hebbian：每步累计资格痕 e，奖励在 t+1 收盘兑现；
- 更新：`w' = max(0, w + η·r·e)`，然后按突触前列重新归一化（保持列和为 1 的不变量）；
- 探索：greedy 决策 + 10% 概率随机翻转（无 HOLD）；
- 硬性约束：拓扑 indices/indptr 永远不动、权重非负、无未来信息；
- 有单元测试（`tests/test_learning_loop.py`）。

### 阶段 F — 连接组只读诊断（已完成，本阶段全部为审计，未改任何模型文件）

1. **蘑菇体学习回路清点**：KC 4,064（15 type）、MBON 97（37 type）、PAM 316（PAM01–15）、PPL1 16（PPL101–108 各 2）、KC→MBON 边 61,210 条 —— 果蝇天然的"感觉→KC→MBON + DAN 奖励"回路在数据中**完整存在且规模可观**；
2. **KC 稀疏性诊断**（10 根随机 K 线、5 步传播、无学习）：x>0 的 KC 100% 全亮，但 >1e-6 仅 6–11% —— "结构上全连通，功能上低幅";
3. **KC pattern separability**（980 根 K 线全量）：任意两根的 top5% KC 集合 Jaccard 0.87–0.95 —— **没有形成随经历变化的动态稀疏表征**，只有静态结构稀疏；
4. **single-channel fingerprint**（四通道分别注入总质量 1.0）：step3 top5% 的通道间 Jaccard 0.71–0.93，up vs down 达 0.92，共同核心 94.5% 是 KCg-d —— **四路市场信号被上游解剖收敛压成几乎同一个 fingerprint**；
5. **官方 neurotransmitter 三字段可靠性审计**（`body-neurotransmitters-male-cns-v1.0.feather`，只读）：
   - 覆盖率：神经元 99.96%（162,457/162,517）、突触边 100%（25,120,209）；
   - 一致率：predicted==consensus 88.78%，celltype==consensus 96.68%；ground_truth（83,496 个神经元）与 consensus **100% 一致**；
   - **关键发现：KC 的 neuron-level 预测 99.85% 是 dopamine，但官方 consensus 把全部 4,064 个 KC 判为 acetylcholine**（KC 无任何 ground_truth 仲裁）；MBON/PAM/PPL1/四组视觉输入的三字段基本一致；
   - 视觉输入（R7p/R7y）neuron-level 有 53%–86% 为 unclear，但 celltype/consensus/gt 一致为 histamine；
   - 结论：**若将来引入 NT 动力学，必须使用层级 ground_truth > consensus_nt > celltype_predicted_nt > predicted_nt，不能直接用 neuron-level 预测**。

## 3. 遇到过的问题与解决

| 问题 | 处理 |
|---|---|
| MaleCNS 官方下载 URL 不在仓库文档中 | 从开源项目 desktop-fly 的 etl 脚本抓取到官方 Google Storage 地址 |
| 43MB NT 文件下载 5 分钟超时只到 54% | `curl -C -` 断点续传（HTTP 206）补完 |
| pyarrow `read_schema` 对该 feather 报错 | 改用 `read_table().schema` |
| 第一轮 NT 审计结论错误（"数据里没有 NT 字段"） | 官方把 NT 放在独立 feather 文件，重新下载审计并修正结论 |
| **核心科学问题：KC 表征同质化** | 定位到根因——解剖上游收敛：四路感觉输入经 3 步扩散后被压成同一个 top5% KC 集合（J=0.92），不是编码器或学习规则的 bug |
| KC 的 dopamine vs acetylcholine 字段冲突 | 未人工纠正，记录层级使用规则（consensus 优先），留待动力学设计时决策 |
| 硬件限制（48GB DDR3、双路 NUMA） | benchmark 确定单线程 SciPy 不可行 → Numba 20 线程方案 |

## 4. 当前进度与状态

- **已完成**：数据管道 → 编码器 → 传播推理 → 学习循环 v0（含单元测试）→ 连接组全面只读诊断（回路清点、KC 稀疏性、separability、fingerprint、NT 审计）。
- **未修改**：P_csr、market_encoder、brain_runner、learning_loop 在所有诊断中保持只读；未实现 LIF、未给 NT 指定正负号、未修改拓扑。
- **已知核心瓶颈**（诊断阶段的结论）：固定解剖拓扑下，不同市场状态在 KC 层产生的表征高度同质，v0 学习循环面对的是一个"几乎相同的输入指纹"，需要后续阶段解决表征可分性。

## 5. 下一步（候选方向，未开工）

1. 用 learning_loop v0 跑通 BTC-USDT 1h 全序列的端到端训练/评估，确认奖励驱动的权重演化行为；
2. 在不违反"拓扑不变"约束的前提下，研究提升 KC 表征可分性的途径（如传播深度、输入分配方式、资格痕时间常数）——遵循"一次只加一条规则"的迭代纪律；
3. 若引入兴奋/抑制动力学，按 NT 审计结论使用 consensus 层级，并对 8.5% unclear 神经元设计明确的处理规则。

## 6. 复现说明

```bash
python -m venv .venv
.venv\Scripts\pip install numpy scipy numba pandas pyarrow
# 数据文件（data/raw、data/processed）体积约 1.4GB，按 README 中的官方源自行下载
.venv\Scripts\python -m unittest discover tests
```
