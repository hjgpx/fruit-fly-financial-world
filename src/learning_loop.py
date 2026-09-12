# -*- coding: utf-8 -*-
"""learning_loop — 果蝇自主学习循环 v0。

每根 K 线（已收盘）:
    1. MarketEncoder 把当前 OHLCV 编成 4 路 0~1 信号, 写入 R7p/R8p/R7y/R8y
       四组感觉神经元（沿用 BrainRunner.input_idx, 共 1,624 个）。
    2. 神经状态从零开始, 用 self._data (可学习副本) 传播 5 步, 每步同时累计
       资格痕 e_k (Hebbian: e_k += x_post[post_k] * x_pre[pre_k])。
    3. 决策: 读 BUY / SELL 两侧 type-mean 等权得分
              greedy = argmax(BUY, SELL)   # tie -> BUY
              action = 1 - greedy          # 10% 概率随机翻成另一动作
              否则 = greedy                # 90% 概率保留
       不存在 HOLD 动作, 不为提升胜率人工选神经元。
    4. 等到下一根 K 线收盘, 计算持仓收益作为奖励:
              BUY  (long)  : r = close[t+1]/close[t] - 1
              SELL (short) : r = -(close[t+1]/close[t] - 1)
              fee = 0 (v0 固定)
    5. 按列应用 Hebbian+reward 更新:
              w_k' = max(0, w_k + η * r * e_k)        # 不允许负权重
              e_k' = 0                                # 资格痕一次性消耗
       然后按突触前列重新归一化, 使每列和为 1 (保持 P 的列归一化不变量:
              对每个 pre j, sum over post P[post, pre=j] = 1).

约束（再次硬性声明）:
    - 拓扑不变 (P.indices / P.indptr 永远不动).
    - 权重非负.
    - 决策只用当前根及之前, 奖励在 t+1 兑现, 无未来.
    - 0% 标签, 0% 手调阈值, 0% 人工选输入神经元.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads

# 让 `from brain_runner import ...` 和 `from market_encoder import ...`
# 在以 `src.learning_loop` 形式被外部 import 时也能解析.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from brain_runner import BrainRunner, STEPS, _spmv, DEFAULT_CSV  # noqa: E402
from market_encoder import PERCENTILE_MIN_HISTORY  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_V1 = PROJECT_ROOT / "data" / "processed" / "v1"
P_CSR = PROCESSED_V1 / "P_csr.npz"

# ---- v0 固定超参 ---- #
ETA = 1e-3         # 学习率
EPSILON = 0.10     # 探索率
FEE = 0.0          # 手续费 (v0 暂为 0)
THREADS = 20       # Numba 并行线程数


# ---------------------------------------------------------------------- #
# Numba kernels
# ---------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def _accumulate_e(e_data, ever, indptr, indices, x_pre, x_post):
    """e_k += x_post[post_k] * x_pre[pre_k], 同时 ever[k] |= (该乘积非零)."""
    n = x_pre.shape[0]
    for pre in prange(n):
        a = x_pre[pre]
        for k in range(indptr[pre], indptr[pre + 1]):
            post = indices[k]
            p = x_post[post] * a
            if p != 0.0:
                e_data[k] += p
                ever[k] = 1
    return e_data


@njit(parallel=True, cache=True)
def _apply_update(data, e_data, eta_r):
    """w_k <- max(0, w_k + eta_r * e_k); e_k <- 0.

    逐边更新, 与拓扑无关; 不再算列和 (由 _compute_col_sums 单独算).
    """
    nnz = data.shape[0]
    for k in prange(nnz):
        new_w = data[k] + eta_r * e_data[k]
        if new_w < 0.0:
            new_w = 0.0
        data[k] = new_w
        e_data[k] = 0.0


@njit(cache=True)
def _compute_col_sums(data, indices, n_cols):
    """col_sums[j] = sum_k data[k] where indices[k] == j.

    按列求和: 对每个突触前 j, 把它所有出边权重加在一起 (P[post, pre=j]).
    串行扫描 nnz=25M 大约几秒, 每根 K 线调用一次, 可接受.
    """
    col_sums = np.zeros(n_cols, dtype=np.float64)
    nnz = data.shape[0]
    for k in range(nnz):
        col_sums[indices[k]] += data[k]
    return col_sums


@njit(parallel=True, cache=True)
def _renorm_columns(data, indices, col_sums):
    """按列归一化: data[k] (indices[k]=j) /= col_sums[j].

    CSR 视角下遍历 nnz 个非零元素, 每个线程处理不同的 k;
    读 col_sums[j] 是只读, 写 data[k] 独立, 无 race condition.
    """
    nnz = data.shape[0]
    for k in prange(nnz):
        s = col_sums[indices[k]]
        if s > 0.0:
            data[k] /= s
    return data


# ---------------------------------------------------------------------- #
class FlyBrain:
    """单只果蝇: 从 P_csr 出生, 在 K 线序列上自主学习."""

    def __init__(self, threads: int = THREADS, epsilon: float = EPSILON,
                 eta: float = ETA, fee: float = FEE):
        t0 = time.perf_counter()

        # --- 1. 复制原始 P 作为可学习副本, 拓扑保留, data 可写 --- #
        from scipy.sparse import load_npz
        P = load_npz(P_CSR).tocsr()
        self._indptr = np.ascontiguousarray(P.indptr, dtype=np.int64)
        self._indices = np.ascontiguousarray(P.indices, dtype=np.int64)
        self._data = np.ascontiguousarray(P.data, dtype=np.float64)
        self._data_initial = self._data.copy()      # 用于事后报告 |Δw|
        self.n = P.shape[0]
        self.nnz = P.nnz

        # --- 2. 资格痕 + 永久激活标记 --- #
        self._e = np.zeros(self.nnz, dtype=np.float64)
        self._ever = np.zeros(self.nnz, dtype=np.uint8)

        # --- 3. 借用 BrainRunner 的输入/输出映射与编码器 --- #
        # BrainRunner 内部的 _data 仅作为只读副本, 不会被我们修改, 也不会被它修改.
        self.runner = BrainRunner(threads=threads)
        self.input_idx = self.runner.input_idx
        self.output_groups = self.runner.output_groups
        self.encoder = self.runner.encoder

        self.eta = eta
        self.epsilon = epsilon
        self.fee = fee
        set_num_threads(threads)
        self._threads = threads
        self._init_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    def _side_score(self, x: np.ndarray, side: str) -> float:
        """type 内神经元均值 -> 该侧所有 type 等权平均."""
        type_means = [float(x[g].mean()) for g in self.output_groups[side]]
        return float(np.mean(type_means))

    def decide_and_accumulate(self, encoded: Dict[str, float],
                              rng: np.random.Generator):
        """写输入 → 传播 5 步 + 累计资格痕 → 返回 (action, BUY, SELL).

        action: 0 = BUY, 1 = SELL. 无 HOLD.
        """
        n = self.n
        x = np.zeros(n, dtype=np.float64)
        for ch, idx in self.input_idx.items():
            x[idx] = encoded[ch] / len(idx)

        y = np.empty_like(x)
        for _ in range(STEPS):
            _spmv(self._indptr, self._indices, self._data, x, y)
            _accumulate_e(self._e, self._ever,
                          self._indptr, self._indices, x, y)
            x, y = y, x

        buy = self._side_score(x, "BUY")
        sell = self._side_score(x, "SELL")
        greedy = 0 if buy >= sell else 1            # argmax, tie -> BUY
        if rng.random() < self.epsilon:
            action = 1 - greedy
        else:
            action = greedy
        return action, buy, sell

    def apply_reward(self, r: float) -> None:
        """收到奖励 r, 应用 Hebbian+reward 更新 + 列归一化."""
        eta_r = self.eta * r
        _apply_update(self._data, self._e, eta_r)
        col_sums = _compute_col_sums(self._data, self._indices, self.n)
        _renorm_columns(self._data, self._indices, col_sums)

    # ------------------------------------------------------------------ #
    def run(self, df: pd.DataFrame) -> Dict[str, object]:
        """在 df 上跑完整学习循环, 返回统计 dict."""
        encoded = self.encoder.encode(
            df["open"].tolist(), df["high"].tolist(), df["low"].tolist(),
            df["close"].tolist(), df["volume"].tolist(),
        )
        ts_col = df["timestamp"].astype(np.int64).to_numpy()
        cl = df["close"].to_numpy()
        N = len(df)

        rng = np.random.default_rng(42)
        actions = np.zeros(N, dtype=np.int8)
        rewards = np.zeros(N, dtype=np.float64)
        buys = np.zeros(N, dtype=np.float64)
        sells = np.zeros(N, dtype=np.float64)

        t0 = time.perf_counter()
        # t = 0..N-2: 决策 + 资格累计 + 等下一根奖励 + 更新
        for t in range(N - 1):
            act, b, s = self.decide_and_accumulate(encoded[t], rng)
            actions[t] = act
            buys[t] = b
            sells[t] = s
            ret = cl[t + 1] / cl[t] - 1.0
            if act == 1:                            # SELL -> short
                ret = -ret
            r = ret - self.fee
            rewards[t] = r
            self.apply_reward(r)
        # t = N-1: 决策但不学
        t = N - 1
        act, b, s = self.decide_and_accumulate(encoded[t], rng)
        actions[t] = act
        buys[t] = b
        sells[t] = s
        run_seconds = time.perf_counter() - t0

        # ---------- 统计 ---------- #
        diff = np.abs(self._data - self._data_initial)
        n_changed = int((diff > 0).sum())
        total_abs = float(diff.sum())
        max_abs = float(diff.max())
        n_ever = int(self._ever.sum())

        v_start = PERCENTILE_MIN_HISTORY
        v_act = actions[v_start:]
        v_rwd = rewards[v_start:]
        cuts_a = np.array_split(v_act, 4)
        cuts_r = np.array_split(v_rwd, 4)
        buy_q = [(c == 0).mean() for c in cuts_a]
        ret_q = [float(c.mean()) for c in cuts_r]

        return {
            "n_eval": len(v_act),
            "run_seconds": run_seconds,
            "n_changed": n_changed,
            "n_ever": n_ever,
            "total_abs_change": total_abs,
            "max_change": max_abs,
            "buy_q": buy_q,
            "ret_q": ret_q,
            "actions": actions,
            "rewards": rewards,
            "ts": ts_col,
            "v_start": v_start,
        }


# ---------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="果蝇自主学习循环 v0")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"CSV: {args.csv.name}  K lines={len(df)}")
    fly = FlyBrain()
    print(f"born: n={fly.n}  nnz={fly.nnz:,}  threads={fly._threads}  "
          f"({fly._init_seconds:.1f}s)  "
          f"eta={fly.eta}  eps={fly.epsilon}  fee={fly.fee}")

    out = fly.run(df)

    print()
    print("=" * 64)
    print(f"learning loop: {out['run_seconds']:.2f}s  "
          f"(valid K lines after warm-up = {out['n_eval']})")

    print()
    print("1) synapse participation & change")
    print(f"   edges that ever saw non-zero eligibility (e != 0): "
          f"{out['n_ever']:,} / {fly.nnz:,} "
          f"({out['n_ever'] / fly.nnz * 100:.4f}%)")
    print(f"   edges whose weight changed vs initial: "
          f"{out['n_changed']:,} / {fly.nnz:,} "
          f"({out['n_changed'] / fly.nnz * 100:.4f}%)")
    print(f"   sum |Δw| over all edges: {out['total_abs_change']:.6f}")
    print(f"   max  |Δw| per edge:    {out['max_change']:.6f}")

    print()
    print("2) BUY/SELL behavior over time (valid candles, 4 quarters)")
    print(f"   {'q':>3}  {'BUY ratio':>10}  {'mean reward':>13}")
    for i, (b, r) in enumerate(zip(out["buy_q"], out["ret_q"]), 1):
        print(f"   Q{i}  {b:>10.4f}  {r * 100:>+12.4f}%")

    print()
    print("3) first 5 BUY/SELL decisions (post warm-up)")
    print(f"   {'UTC time':<16}  {'act':>5}  {'reward':>9}")
    for t in range(out["v_start"], out["v_start"] + 5):
        utc = datetime.fromtimestamp(int(out["ts"][t]), tz=timezone.utc)
        a = "BUY" if out["actions"][t] == 0 else "SELL"
        print(f"   {utc.strftime('%Y-%m-%d %H:%M'):<16}  {a:>5}  "
              f"{out['rewards'][t] * 100:>+8.4f}%")

    print()
    print("4) last 5 BUY/SELL decisions (with realized reward)")
    print(f"   {'UTC time':<16}  {'act':>5}  {'reward':>9}")
    for t in range(len(out["actions"]) - 6, len(out["actions"]) - 1):
        utc = datetime.fromtimestamp(int(out["ts"][t]), tz=timezone.utc)
        a = "BUY" if out["actions"][t] == 0 else "SELL"
        print(f"   {utc.strftime('%Y-%m-%d %H:%M'):<16}  {a:>5}  "
              f"{out['rewards'][t] * 100:>+8.4f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()