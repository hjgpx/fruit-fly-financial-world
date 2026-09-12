# -*- coding: utf-8 -*-
"""brain_runner — 果蝇连接组行情推理 runner。

对每根 K 线（因果编码后的有效 K 线）执行:

    1. 用 MarketEncoder 得到 4 路输入信号，按 market_input_map.json
       平均分配写入对应感觉神经元（R7p/R8p/R7y/R8y, 共 1,624 个）:
           x0[neuron] = channel_signal / 组内神经元数
    2. 每根 K 线都从全新的 x0 开始（不跨 K 线累积状态），
       用列归一化传播矩阵 P_csr.npz 连续传播 STEPS=5 步: x = P @ x
       （P[post, pre]，即突触前 → 突触后传播，Numba 20 线程并行 SpMV）
    3. 读出 BUY / SELL 运动神经元的信号。为避免两侧神经元数量不平衡
       （BUY 36 个 / SELL 164 个），聚合方式为:
           type 均值: 该 type 内全部神经元信号的平均
           侧得分  : 该侧所有 type 均值的平均（等权 type，不等权神经元）
    4. direction = (BUY - SELL) / (BUY + SELL + 1e-12)，范围约 -1 ~ +1

不定义 HOLD，不生成交易信号，不计算收益，不写任何数据文件。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads

from market_encoder import MarketEncoder, PERCENTILE_MIN_HISTORY

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_V1 = PROJECT_ROOT / "data" / "processed" / "v1"
DEFAULT_CSV = PROJECT_ROOT / "data" / "market" / "BTC-USDT_1h.csv"
ACTION_MAP = PROJECT_ROOT / "config" / "action_map.json"
NODES_FEATHER = PROCESSED_V1 / "nodes.feather"
P_CSR = PROCESSED_V1 / "P_csr.npz"
BODY_IDS_NPY = PROCESSED_V1 / "body_ids.npy"

STEPS = 5          # 每根 K 线的传播步数
THREADS = 20       # Numba 并行线程数（benchmark_v1 结论: 默认 20 线程）
EPS = 1e-12        # direction 分母平滑


# ---------------------------------------------------------------------- #
# Numba 并行 SpMV: out = P @ x
# ---------------------------------------------------------------------- #

@njit(parallel=True, cache=True)
def _spmv(indptr, indices, data, x, out):
    n = x.shape[0]
    for i in prange(n):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        out[i] = s
    return out


class BrainRunner:
    """加载连接组 + 输入/输出映射，对 K 线序列执行传播与读出。"""

    def __init__(self, threads: int = THREADS):
        t0 = time.perf_counter()

        # --- 传播矩阵 P（列归一化, P[post, pre] = weight/出边权重和） --- #
        from scipy.sparse import load_npz
        P = load_npz(P_CSR).tocsr()
        self._indptr = np.ascontiguousarray(P.indptr, dtype=np.int64)
        self._indices = np.ascontiguousarray(P.indices, dtype=np.int64)
        self._data = np.ascontiguousarray(P.data, dtype=np.float64)
        self.n = P.shape[0]
        self.nnz = P.nnz

        # --- bodyId → 矩阵索引 固定映射 --- #
        body_ids = np.load(BODY_IDS_NPY)
        self._id2idx = {int(b): i for i, b in enumerate(body_ids)}

        # --- 输入映射（4 路行情信号 → 感觉神经元索引） --- #
        self.encoder = MarketEncoder()
        self.input_idx: Dict[str, np.ndarray] = {
            ch: np.array(
                [self._id2idx[int(b)] for b in self.encoder.body_ids[ch]],
                dtype=np.int64,
            )
            for ch in self.encoder.body_ids
        }

        # --- 输出映射（BUY/SELL bodyId → 按 type 分组的索引） --- #
        with open(ACTION_MAP, encoding="utf-8") as f:
            action = json.load(f)
        nodes = pd.read_feather(NODES_FEATHER, columns=["bodyId", "type"])
        bid2type = dict(zip(nodes["bodyId"].astype(np.int64),
                            nodes["type"].astype(str)))
        del nodes

        self.output_groups: Dict[str, List[np.ndarray]] = {}
        for side in ("BUY", "SELL"):
            ids = [int(b) for b in action[side]["bodyIds"]]
            by_type: Dict[str, List[int]] = {}
            for b in ids:
                by_type.setdefault(bid2type[b], []).append(self._id2idx[b])
            self.output_groups[side] = [
                np.array(v, dtype=np.int64) for v in by_type.values()
            ]

        set_num_threads(threads)
        self._threads = threads
        self._load_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ #

    def run_candle(self, candle: Dict[str, float]) -> Dict[str, float]:
        """单根 K 线: 写入输入 → 传播 STEPS 步 → 读出 BUY/SELL/direction。"""
        # 1. 全新输入状态
        x = np.zeros(self.n, dtype=np.float64)
        for ch, idx in self.input_idx.items():
            x[idx] = candle[ch] / len(idx)   # 平均分配给组内神经元

        # 2. 连续传播 STEPS 步（每根 K 线独立的网络动态）
        y = np.empty_like(x)
        for _ in range(STEPS):
            _spmv(self._indptr, self._indices, self._data, x, y)
            x, y = y, x

        # 3. 双层平均读出: 先 type 内神经元均值，再 type 间等权均值
        buy = self._side_score(x, "BUY")
        sell = self._side_score(x, "SELL")
        direction = (buy - sell) / (buy + sell + EPS)
        return {"BUY": buy, "SELL": sell, "direction": direction}

    def _side_score(self, x: np.ndarray, side: str) -> float:
        type_means = [float(x[g].mean()) for g in self.output_groups[side]]
        return float(np.mean(type_means))

    # ------------------------------------------------------------------ #

    def run_series(self, encoded: List[Dict[str, float]],
                   timestamps: List[int]) -> List[dict]:
        """对一批已编码 K 线逐根执行（每根独立，无跨根状态）。"""
        rows = []
        t0 = time.perf_counter()
        for candle, ts in zip(encoded, timestamps):
            r = self.run_candle(candle)
            r["timestamp"] = ts
            rows.append(r)
        self._run_seconds = time.perf_counter() - t0
        return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="果蝇连接组行情推理 runner")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--last", type=int, default=100,
                    help="取最后 N 根有效 K 线（warm-up 之后）")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"CSV: {args.csv.name}  K lines={len(df)}")

    runner = BrainRunner()
    print(f"loaded: n={runner.n}  nnz={runner.nnz:,}  "
          f"steps={STEPS}  threads={runner._threads}  "
          f"({runner._load_seconds:.1f}s)")
    print(f"output groups: "
          + ", ".join(f"{s}={len(g)} types"
                      for s, g in runner.output_groups.items()))

    # 编码全部 K 线，取 warm-up 之后的有效 K 线的最后 N 根
    encoded = runner.encoder.encode(
        df["open"].tolist(), df["high"].tolist(), df["low"].tolist(),
        df["close"].tolist(), df["volume"].tolist(),
    )
    ts_list = df["timestamp"].astype(np.int64).tolist()
    valid = [(e, t) for i, (e, t) in enumerate(zip(encoded, ts_list))
             if i >= PERCENTILE_MIN_HISTORY]
    chosen = valid[-args.last:]
    print(f"valid candles (after {PERCENTILE_MIN_HISTORY}-bar warm-up): "
          f"{len(valid)}, running last {len(chosen)}")
    print()

    rows = runner.run_series([e for e, _ in chosen], [t for _, t in chosen])

    # ---- 每根输出 ---- #
    print(f"{'idx':>4}  {'timestamp':>10}  {'UTC time':<16}  "
          f"{'BUY':>12}  {'SELL':>12}  {'direction':>10}")
    start_i = len(valid) - len(chosen)
    for k, r in enumerate(rows):
        utc = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
        print(f"{start_i + k:>4}  {r['timestamp']:>10}  "
              f"{utc.strftime('%Y-%m-%d %H:%M'):<16}  "
              f"{r['BUY']:>12.6e}  {r['SELL']:>12.6e}  "
              f"{r['direction']:>10.6f}")

    # ---- 汇总 ---- #
    d = np.array([r["direction"] for r in rows])
    conn_total = len(rows) * STEPS * runner.nnz
    print()
    print(f"direction summary (n={len(rows)}): "
          f"min={d.min():.6f}  max={d.max():.6f}  mean={d.mean():.6f}")
    print(f"propagation: {len(rows)} candles x {STEPS} steps, "
          f"total {conn_total:,} connections in {runner._run_seconds:.2f}s "
          f"({conn_total / runner._run_seconds / 1e6:.0f}M conn/s)")
    print()
    print("last 10 candles:")
    print(f"{'idx':>4}  {'timestamp':>10}  {'UTC time':<16}  "
          f"{'BUY':>12}  {'SELL':>12}  {'direction':>10}")
    for k, r in enumerate(rows[-10:], start=len(rows) - 10):
        utc = datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
        print(f"{start_i + k:>4}  {r['timestamp']:>10}  "
              f"{utc.strftime('%Y-%m-%d %H:%M'):<16}  "
              f"{r['BUY']:>12.6e}  {r['SELL']:>12.6e}  "
              f"{r['direction']:>10.6f}")


if __name__ == "__main__":
    main()
