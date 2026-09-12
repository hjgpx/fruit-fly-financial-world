# -*- coding: utf-8 -*-
"""market_encoder 单元测试 — 运行: python -m unittest discover -s tests -v"""

import math
import unittest

from src.market_encoder import MarketEncoder, CHANNELS, VOLUME_BASELINE_WINDOW


def approx(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


class TestRawSignals(unittest.TestCase):
    def setUp(self):
        self.enc = MarketEncoder()

    def test_price_up_down_raw(self):
        # open=100, close=110 → +10%；close=90 → -10%；close=100 → 0
        o = [100.0, 100.0, 100.0]
        h = [110.0, 100.0, 100.0]
        l = [100.0, 90.0, 100.0]
        c = [110.0, 90.0, 100.0]
        v = [100.0] * 3
        raw = self.enc.raw_signals(o, h, l, c, v)
        self.assertTrue(approx(raw["price_up"][0], 0.1))
        self.assertTrue(approx(raw["price_up"][1], 0.0))
        self.assertTrue(approx(raw["price_up"][2], 0.0))
        self.assertTrue(approx(raw["price_down"][0], 0.0))
        self.assertTrue(approx(raw["price_down"][1], 0.1))
        self.assertTrue(approx(raw["price_down"][2], 0.0))

    def test_price_flat_candle(self):
        # 平盘 K 线: close == open → price_up = price_down = 0
        raw = self.enc.raw_signals([50.0], [51.0], [49.0], [50.0], [10.0])
        self.assertEqual(raw["price_up"], [0.0])
        self.assertEqual(raw["price_down"], [0.0])

    def test_volatility_raw(self):
        o = [100.0, 200.0]
        h = [102.0, 210.0]
        l = [99.0, 195.0]
        c = [100.0, 200.0]
        v = [10.0, 10.0]
        raw = self.enc.raw_signals(o, h, l, c, v)
        self.assertTrue(approx(raw["volatility"][0], (102.0 - 99.0) / 100.0))
        self.assertTrue(approx(raw["volatility"][1], (210.0 - 195.0) / 200.0))

    def test_volume_first_candle_zero(self):
        # 第 0 根无历史 → volume 信号为 0
        raw = self.enc.raw_signals([10.0], [11.0], [9.0], [10.0], [1000.0])
        self.assertEqual(raw["volume"], [0.0])

    def test_volume_constant_history(self):
        # 恒定成交量 → 每根变化都是 0
        n = 30
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        v = [500.0] * n
        raw = self.enc.raw_signals(o, h, l, c, v)
        self.assertTrue(all(x == 0.0 for x in raw["volume"]))

    def test_volume_known_ratio(self):
        # 前 20 根成交量均为 100，第 21 根为 200 → 变化 = +1.0
        n = VOLUME_BASELINE_WINDOW + 1
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        v = [100.0] * VOLUME_BASELINE_WINDOW + [200.0]
        raw = self.enc.raw_signals(o, h, l, c, v)
        self.assertTrue(all(x == 0.0 for x in raw["volume"][:n - 1]))
        self.assertTrue(approx(raw["volume"][-1], 1.0))

    def test_volume_negative_change_clipped(self):
        # 成交量下降 → 负变化被截断为 0
        o = [100.0, 100.0]
        h = [101.0, 101.0]
        l = [99.0, 99.0]
        c = [100.0, 100.0]
        v = [200.0, 100.0]
        raw = self.enc.raw_signals(o, h, l, c, v)
        self.assertTrue(approx(raw["volume"][1], 0.0))

    def test_volume_window_is_20(self):
        # 验证基线窗口 = 前 20 根（不含当前），且更早的离群值不参与基线:
        # v = [1e8] + [100]*20 + [200]，第 22 根 (i=21) 的历史 = volume[1:21] = 20 根 100
        # 若窗口只看前 20 根 → 基线 100 → 变化 = 200/100 - 1 = 1.0
        # 若把 i=0 的 1e8 离群值也算进来 → 基线巨大 → 变化 ≈ 0
        n = 22
        o = [100.0] * n
        h = [101.0] * n
        l = [99.0] * n
        c = [100.0] * n
        v = [1e8] + [100.0] * 20 + [200.0]
        raw = self.enc.raw_signals(o, h, l, c, v)
        self.assertTrue(approx(raw["volume"][21], 1.0))


class TestCausalPercentile(unittest.TestCase):
    """因果 percentile rank 归一化测试。"""

    def test_insufficient_history(self):
        # 19 根历史 → 第 20 根输出 0；20 根历史 → 开始输出非零
        vals = [0.5] * 19 + [0.6]
        out = MarketEncoder._causal_percentile(vals)
        self.assertTrue(all(v == 0.0 for v in out[:19]))
        # 第 20 根 (i=19): 历史恰 19 根 < 20 → 0
        self.assertEqual(out[19], 0.0)
        vals2 = [0.5] * 20 + [0.6]
        out2 = MarketEncoder._causal_percentile(vals2)
        # 第 21 根 (i=20): 历史恰 20 根, 0.6 > 全部 0.5 → 1.0
        self.assertTrue(approx(out2[20], 1.0))

    def test_zero_signal_outputs_zero(self):
        # 当前 raw = 0 → 即使历史充足也输出 0（而非高百分位）
        vals = [0.1] * 20 + [0.0]
        out = MarketEncoder._causal_percentile(vals)
        self.assertEqual(out[20], 0.0)

    def test_percentile_above_all(self):
        # 当前值 > 全部历史 → 1.0
        vals = [0.1] * 20 + [0.9]
        out = MarketEncoder._causal_percentile(vals)
        self.assertTrue(approx(out[20], 1.0))

    def test_percentile_below_all(self):
        # 0 < 当前值 < 全部历史 → (# <= 当前) = 0 → 0.0
        vals = [0.5] * 20 + [0.2]
        out = MarketEncoder._causal_percentile(vals)
        self.assertEqual(out[20], 0.0)

    def test_percentile_mid(self):
        # 历史 20 个值: 10 个 0.1, 10 个 0.5；当前 0.3 → (# <= 0.3) = 10 → 0.5
        vals = [0.1] * 10 + [0.5] * 10 + [0.3]
        out = MarketEncoder._causal_percentile(vals)
        self.assertTrue(approx(out[20], 0.5))

    def test_percentile_with_ties(self):
        # 当前值与部分历史相等: 历史 20 个 0.2, 当前 0.2 → 20/20 = 1.0 (含等于)
        vals = [0.2] * 20 + [0.2]
        out = MarketEncoder._causal_percentile(vals)
        self.assertTrue(approx(out[20], 1.0))

    def test_window_limited_to_100(self):
        # 120 个历史 0.5 + 当前 0.9: 若只用最近 100 根 → 100/100 = 1.0
        # （更早的历史不影响结果，此处用相同的 0.5 无法区分，改用结构验证: ）
        # 构造: [1.0]*20 + [0.5]*100 + [0.9]
        # i=120 的历史最近 100 根全为 0.5 → 1.0（最前面 20 根 1.0 不参与）
        vals = [1.0] * 20 + [0.5] * 100 + [0.9]
        out = MarketEncoder._causal_percentile(vals)
        self.assertTrue(approx(out[120], 1.0))
        # 反例: 若窗口无限, (# <= 0.9) = 100 (0.5们), 但 1.0 们 > 0.9 → 仍是 100/120
        # 区分性检查: 当前值 = 0.5 时, 有限窗口 → 100/100 = 1.0; 无限窗口 → 100/120
        vals2 = [1.0] * 20 + [0.5] * 100 + [0.5]
        out2 = MarketEncoder._causal_percentile(vals2)
        self.assertTrue(approx(out2[120], 1.0))  # 有限窗口取 100/100, 而非 100/120

    def test_causality_future_does_not_change_past(self):
        """核心性质: 修改未来 K 线不会改变过去已经生成的信号。"""
        n = 40
        base_o = [100.0] * n
        base_h = [101.0 + (i % 5) * 0.3 for i in range(n)]
        base_l = [99.0 - (i % 3) * 0.2 for i in range(n)]
        base_c = [100.0 + ((i * 7) % 11 - 5) * 0.8 for i in range(n)]
        base_v = [100.0 + ((i * 13) % 17) * 10.0 for i in range(n)]

        enc = MarketEncoder()
        out1 = enc.encode(base_o, base_h, base_l, base_c, base_v)

        # 修改第 20 根之后的所有 K 线（未来）
        k = 20
        mod_o = list(base_o)
        mod_h = list(base_h)
        mod_l = list(base_l)
        mod_c = list(base_c)
        mod_v = list(base_v)
        for i in range(k, n):
            mod_o[i] = 50.0 + i
            mod_h[i] = 90.0 + i
            mod_l[i] = 10.0 + i
            mod_c[i] = 70.0 + i * 2
            mod_v[i] = 12345.0 + i * 100

        out2 = enc.encode(mod_o, mod_h, mod_l, mod_c, mod_v)

        # 过去 (0..k-1) 的信号逐根逐路完全一致
        for i in range(k):
            self.assertEqual(out1[i], out2[i], msg=f"candle {i} changed by future edit")

        # 未来确实被改动了（sanity check, 避免恒等假阳性）
        changed = any(out1[i] != out2[i] for i in range(k, n))
        self.assertTrue(changed, "修改未来 K 线未产生任何信号变化, 测试无效")

    def test_encode_causal_values_range(self):
        # 25 根 K 线: 前 20 根涨跌交替（历史形成中），后 5 根造已知信号
        n = 25
        o = [100.0] * n
        c = [100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(n)]
        h = [max(o[i], c[i]) + 0.5 for i in range(n)]
        l = [min(o[i], c[i]) - 0.5 for i in range(n)]
        v = [100.0] * n
        # 第 20~24 根改为大幅上涨 + 放量 + 高波幅
        for i in range(20, n):
            c[i] = 130.0
            h[i] = 135.0
            l[i] = 95.0
            v[i] = 500.0

        enc = MarketEncoder()
        out = enc.encode(o, h, l, c, v)
        self.assertEqual(len(out), n)
        # 前 20 根: 历史不足 20 → 全部 0
        for i in range(20):
            for ch in CHANNELS:
                self.assertEqual(out[i][ch], 0.0, msg=f"candle {i} {ch}")
        # 第 21 根 (i=20): price_up raw=0.3 > 全部历史 (±0.01) → 1.0
        self.assertTrue(approx(out[20]["price_up"], 1.0))
        # volatility raw=0.4 > 全部历史 (0.01) → 1.0
        self.assertTrue(approx(out[20]["volatility"], 1.0))
        # price_down raw=0 → 0
        self.assertEqual(out[20]["price_down"], 0.0)
        # 所有输出在 [0,1]
        for candle in out:
            for ch in CHANNELS:
                self.assertGreaterEqual(candle[ch], 0.0)
                self.assertLessEqual(candle[ch], 1.0)

    def test_single_candle_all_zero(self):
        # 单根 K 线: 历史不足 20 → 全部输出 0
        enc = MarketEncoder()
        enc_out = enc.encode([100.0], [110.0], [95.0], [105.0], [500.0])
        self.assertEqual(len(enc_out), 1)
        for ch in CHANNELS:
            self.assertEqual(enc_out[0][ch], 0.0)


class TestDistribution(unittest.TestCase):
    def setUp(self):
        self.enc = MarketEncoder()

    def test_config_counts(self):
        # 与 market_input_map.json 生成时一致
        self.assertEqual(self.enc.counts["price_up"], 332)
        self.assertEqual(self.enc.counts["price_down"], 330)
        self.assertEqual(self.enc.counts["volume"], 481)
        self.assertEqual(self.enc.counts["volatility"], 481)
        self.assertEqual(sum(self.enc.counts.values()), 1624)

    def test_distribute_average_allocation(self):
        # 直接构造已知缩放信号, 独立于缩放逻辑测试分配
        encoded = [
            {"price_up": 1.0, "price_down": 0.0, "volume": 0.5, "volatility": 0.25},
            {"price_up": 0.0, "price_down": 0.8, "volume": 0.0, "volatility": 1.0},
        ]
        neurons = self.enc.distribute(encoded)

        self.assertEqual(len(neurons), 2)
        nmap = neurons[0]
        # 总神经元数 = 332 + 330 + 481 + 481 = 1624
        self.assertEqual(len(nmap), 1624)

        # 每路总信号守恒: sum(neuron values in group) == channel signal
        for ch in CHANNELS:
            total = sum(nmap[bid] for bid in self.enc.body_ids[ch])
            self.assertTrue(approx(total, encoded[0][ch]),
                            msg=f"{ch}: {total} != {encoded[0][ch]}")

        # 平均分配: price_up=1.0 → 每个 R7p 神经元 = 1/332
        expected = 1.0 / 332
        for bid in self.enc.body_ids["price_up"]:
            self.assertTrue(approx(nmap[bid], expected))
        # volume=0.5 → 每个 R7y 神经元 = 0.5/481
        expected = 0.5 / 481
        for bid in self.enc.body_ids["volume"]:
            self.assertTrue(approx(nmap[bid], expected))
        # price_down = 0.0 → 每个 R8p 神经元 = 0
        for bid in self.enc.body_ids["price_down"]:
            self.assertEqual(nmap[bid], 0.0)

        # 第 2 根: price_down=0.8 → 每个 R8p 神经元 = 0.8/330
        expected2 = 0.8 / 330
        for bid in self.enc.body_ids["price_down"]:
            self.assertTrue(approx(neurons[1][bid], expected2))

    def test_groups_disjoint_in_output(self):
        # 输出 dict 的 key 总数 = 四组神经元总数（无重叠才可能）
        o = [100.0, 90.0]
        h = [110.0, 95.0]
        l = [95.0, 85.0]
        c = [105.0, 92.0]
        v = [100.0, 250.0]
        neurons = self.enc.encode_ohlcv(o, h, l, c, v)
        for nmap in neurons:
            self.assertEqual(len(nmap), 1624)
            for val in nmap.values():
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.enc = MarketEncoder()

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            self.enc.raw_signals([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0])

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self.enc.raw_signals([], [], [], [], [])

    def test_zero_open(self):
        with self.assertRaises(ValueError):
            self.enc.raw_signals([0.0], [1.0], [0.0], [1.0], [1.0])

    def test_high_below_low(self):
        with self.assertRaises(ValueError):
            self.enc.raw_signals([100.0], [90.0], [95.0], [100.0], [10.0])

    def test_negative_volume(self):
        with self.assertRaises(ValueError):
            self.enc.raw_signals([100.0], [110.0], [90.0], [100.0], [-5.0])


class TestDataFrame(unittest.TestCase):
    def test_encode_df(self):
        import pandas as pd

        df = pd.DataFrame({
            "open": [100.0, 100.0],
            "high": [110.0, 102.0],
            "low": [99.0, 98.0],
            "close": [108.0, 99.0],
            "volume": [100.0, 300.0],
        })
        enc = MarketEncoder()
        out = enc.encode_df(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[0]), 1624)

    def test_encode_df_missing_column(self):
        import pandas as pd

        df = pd.DataFrame({"open": [1.0], "close": [1.0]})
        enc = MarketEncoder()
        with self.assertRaises(ValueError):
            enc.encode_df(df)


if __name__ == "__main__":
    unittest.main()
