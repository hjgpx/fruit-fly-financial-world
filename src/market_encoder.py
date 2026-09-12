# -*- coding: utf-8 -*-
"""market_encoder — OHLCV K 线 → 果蝇连接组输入神经元信号编码器。

信号管道（对每根 K 线 i）:

    1. 原始信号（4 路，均为因果计算，只依赖 i 及之前的数据）:
         price_up   = max(close/open - 1, 0)
         price_down = max(1 - close/open, 0)
         volume     = max(volume_i / mean(前 20 根 K 线成交量) - 1, 0)
                      第 0 根 K 线无历史，定义为 0
         volatility  = (high - low) / open

    2. 因果归一化（0~1, percentile rank）:
         对每一路，用最近 PERCENTILE_WINDOW=100 根历史 raw signal
         （不包含当前根）计算当前 raw signal 的百分位:
             scaled = (# 历史值 <= 当前值) / 历史根数
         规则:
             - 当前 raw signal == 0 → 输出 0
             - 历史不足 PERCENTILE_MIN_HISTORY=20 根 → 输出 0
             - 历史超过 100 根时只取最近 100 根
         该变换只依赖当前根之前的数据（因果），因此修改未来 K 线
         不会改变过去已经生成的信号。

    3. 分配到输入神经元:
         依据 config/market_input_map.json:
             price_up   → R7p (332 个)
             price_down → R8p (330 个)
             volume     → R7y (481 个)
             volatility → R8y (481 个)
         每一路的信号值平均分配给组内全部神经元:
             neuron_value = channel_signal / 组内神经元数量

注意:
    - volume 基线取「不含当前根」的前 20 根 K 线（不足 20 根时取全部历史），
      避免当前根自身稀释基线。
    - 不做传播计算、不接交易所 API、不写任何数据文件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "market_input_map.json"

CHANNELS = ("price_up", "price_down", "volume", "volatility")

VOLUME_BASELINE_WINDOW = 20  # volume 基线窗口（前 20 根 K 线）
PERCENTILE_WINDOW = 100  # percentile rank 历史窗口（最近 100 根，不含当前）
PERCENTILE_MIN_HISTORY = 20  # 历史不足 20 根时输出 0


class MarketEncoder:
    """把 OHLCV K 线编码为输入神经元信号（bodyId -> 0~1 值）。"""

    def __init__(self, config_path: Union[str, Path] = DEFAULT_CONFIG):
        config_path = Path(config_path)
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

        self.market_signals: Dict[str, str] = dict(cfg["rules"]["market_signals"])
        self.body_ids: Dict[str, List[int]] = {
            k: list(v) for k, v in cfg["bodyIds"].items()
        }
        self.counts: Dict[str, int] = {k: len(v) for k, v in self.body_ids.items()}

        if set(self.body_ids) != set(CHANNELS):
            raise ValueError(
                f"config 频道 {sorted(self.body_ids)} 与期望 {list(CHANNELS)} 不一致"
            )

        # 校验四组互不重叠（与生成配置时的校验保持一致）
        seen: set = set()
        for ch in CHANNELS:
            ids = set(self.body_ids[ch])
            overlap = seen & ids
            if overlap:
                raise ValueError(f"频道 {ch} 与其他频道存在重叠 bodyId: {sorted(overlap)[:5]}...")
            if len(ids) != len(self.body_ids[ch]):
                raise ValueError(f"频道 {ch} 内部 bodyId 存在重复")
            seen |= ids

    # ------------------------------------------------------------------ #
    # 第 1 步: 原始信号
    # ------------------------------------------------------------------ #

    def raw_signals(
        self,
        open_: Sequence[float],
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        volume: Sequence[float],
    ) -> Dict[str, List[float]]:
        """计算每根 K 线的 4 路原始信号（未缩放）。"""
        n = self._validate(open_, high, low, close, volume)

        price_up: List[float] = []
        price_down: List[float] = []
        volatility: List[float] = []
        vol_sig: List[float] = []

        for i in range(n):
            o, h, l, c = open_[i], high[i], low[i], close[i]
            ret = c / o
            price_up.append(max(ret - 1.0, 0.0))
            price_down.append(max(1.0 - ret, 0.0))
            volatility.append((h - l) / o)

            # volume: 相对前 20 根（不含当前）平均成交量的变化，截断为非负
            if i == 0:
                vol_sig.append(0.0)
            else:
                hist = volume[max(0, i - VOLUME_BASELINE_WINDOW): i]
                baseline = sum(hist) / len(hist)
                if baseline <= 0:
                    vol_sig.append(0.0)
                else:
                    vol_sig.append(max(volume[i] / baseline - 1.0, 0.0))

        return {
            "price_up": price_up,
            "price_down": price_down,
            "volume": vol_sig,
            "volatility": volatility,
        }

    # ------------------------------------------------------------------ #
    # 第 2 步: 因果 percentile rank 归一化到 [0, 1]
    # ------------------------------------------------------------------ #

    @staticmethod
    def _causal_percentile(
        values: List[float],
        min_history: int = PERCENTILE_MIN_HISTORY,
        window: int = PERCENTILE_WINDOW,
    ) -> List[float]:
        """对整路 raw signal 做因果 percentile rank 归一化。

        对第 i 根:
            hist = values[max(0, i - window) : i]   (不含当前根)
            - len(hist) < min_history → 0.0
            - values[i] == 0.0        → 0.0
            - 否则 → (# hist 中 <= values[i] 的个数) / len(hist)

        只依赖 i 之前的数据，因果。
        """
        out: List[float] = []
        for i, v in enumerate(values):
            hist = values[max(0, i - window): i]
            if len(hist) < min_history:
                out.append(0.0)
            elif v == 0.0:
                out.append(0.0)
            else:
                rank = sum(1 for h in hist if h <= v)
                out.append(rank / len(hist))
        return out

    def encode(
        self,
        open_: Sequence[float],
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        volume: Sequence[float],
    ) -> List[Dict[str, float]]:
        """对每根 K 线输出 4 路因果归一化的 0~1 信号。

        返回: List[Dict[channel, scaled_value]]，长度 = K 线数量。
        """
        raw = self.raw_signals(open_, high, low, close, volume)
        scaled = {ch: self._causal_percentile(raw[ch]) for ch in CHANNELS}
        n = len(raw["price_up"])
        return [{ch: scaled[ch][i] for ch in CHANNELS} for i in range(n)]

    # ------------------------------------------------------------------ #
    # 第 3 步: 平均分配到输入神经元
    # ------------------------------------------------------------------ #

    def distribute(self, encoded: List[Dict[str, float]]) -> List[Dict[int, float]]:
        """把每一路总信号平均分配给对应输入神经元。

        输入: encode() 的输出。
        返回: List[bodyId -> value]，四组互不重叠，故直接合并为单个 dict。
        """
        out: List[Dict[int, float]] = []
        for candle in encoded:
            neuron_input: Dict[int, float] = {}
            for ch in CHANNELS:
                share = candle[ch] / self.counts[ch]
                for bid in self.body_ids[ch]:
                    neuron_input[bid] = share
            out.append(neuron_input)
        return out

    def encode_ohlcv(
        self,
        open_: Sequence[float],
        high: Sequence[float],
        low: Sequence[float],
        close: Sequence[float],
        volume: Sequence[float],
    ) -> List[Dict[int, float]]:
        """一步到位: OHLCV → 每根 K 线的输入神经元信号 (bodyId -> 0~1)。"""
        return self.distribute(self.encode(open_, high, low, close, volume))

    def encode_df(self, df) -> List[Dict[int, float]]:
        """pandas DataFrame 版本，要求列: open/high/low/close/volume。"""
        cols = ("open", "high", "low", "close", "volume")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame 缺少列: {missing}")
        return self.encode_ohlcv(
            df["open"].tolist(),
            df["high"].tolist(),
            df["low"].tolist(),
            df["close"].tolist(),
            df["volume"].tolist(),
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(open_, high, low, close, volume) -> int:
        seqs = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        lengths = {len(s) for s in seqs.values()}
        if len(lengths) != 1:
            raise ValueError(f"输入序列长度不一致: { {k: len(v) for k, v in seqs.items()} }")
        n = lengths.pop()
        if n == 0:
            raise ValueError("输入为空")
        for i in range(n):
            if open_[i] <= 0:
                raise ValueError(f"第 {i} 根 K 线 open <= 0: {open_[i]}")
            if high[i] < low[i]:
                raise ValueError(f"第 {i} 根 K 线 high < low: {high[i]} < {low[i]}")
            if volume[i] < 0:
                raise ValueError(f"第 {i} 根 K 线 volume < 0: {volume[i]}")
        return n
