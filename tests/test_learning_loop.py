# -*- coding: utf-8 -*-
"""learning_loop 内核不变量单元测试 (unittest 版)。

只用小型 CSR (4x4), 不加载 P_csr.npz, 保证测试秒级完成。

运行: python -m unittest tests.test_learning_loop -v

不变量:
  - 权重非负 (Hebbian 更新 + clip)
  - 资格痕一次性消耗 (apply_update 后 e == 0)
  - **每列 (pre j) 的出边权重和 = 1** (renorm 真的按 pre 归一化)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

# 让 `from src.learning_loop import ...` 可用
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.learning_loop import (  # noqa: E402
    _accumulate_e, _apply_update, _compute_col_sums, _renorm_columns,
)


def _small_csr():
    """4x4 P (column-normalized):
       col 0 -> rows 1,2   (0.4, 0.6)
       col 1 -> rows 0,3   (0.7, 0.3)
       col 2 -> rows 1,3   (0.5, 0.5)
       col 3 -> (empty)
    """
    rows = np.array([1, 2, 0, 3, 1, 3], dtype=np.int64)
    cols = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    data = np.array([0.4, 0.6, 0.7, 0.3, 0.5, 0.5])
    return csr_matrix((data, (rows, cols)), shape=(4, 4)).tocsr()


def _asymmetric_csr():
    """5x5 列归一化 CSR, 故意让 行和 ≠ 列和 (用来验证 renorm 按 pre 归一化,
    不按 post 归一化).

       col 0 -> row 0          (1.0)
       col 1 -> rows 0,1       (0.7, 0.3)
       col 2 -> row 1          (1.0)
       col 3 -> row 2          (1.0)
       col 4 -> rows 0,4       (0.6, 0.4)

    行和: row 0 = 2.3, row 1 = 1.3, row 2 = 1.0, row 3 = 0, row 4 = 0.4
    行和远非 1 (明显非对称); 列和每个非空列都是 1.
    """
    rows = np.array([0, 0, 1, 1, 2, 0, 4], dtype=np.int64)
    cols = np.array([0, 1, 1, 2, 3, 4, 4], dtype=np.int64)
    data = np.array([1.0, 0.7, 0.3, 1.0, 1.0, 0.6, 0.4])
    return csr_matrix((data, (rows, cols)), shape=(5, 5)).tocsr()


def _col_sums_of(data, indices, n_cols):
    """手算 col_sums: 对每个 j, sum_k data[k] where indices[k] == j."""
    s = np.zeros(n_cols, dtype=np.float64)
    for k in range(len(indices)):
        s[indices[k]] += data[k]
    return s


def _assert_column_normalized(test, data, indices, n_cols, msg=""):
    """断言每列 j (indices[k]=j) 的 data 加和 = 1."""
    cs = _col_sums_of(data, indices, n_cols)
    for j in range(n_cols):
        # 跳过空列 (没有任何 indices[k]=j)
        n_in_col = int((indices == j).sum())
        if n_in_col == 0:
            continue
        test.assertAlmostEqual(float(cs[j]), 1.0, places=9,
                               msg=f"{msg} col {j} sum = {cs[j]}")


def _get(P):
    return (P.indptr.astype(np.int64),
            P.indices.astype(np.int64),
            P.data.astype(np.float64).copy())


class TestApplyUpdate(unittest.TestCase):
    def test_clips_negative_and_clears_e(self):
        """eta_r < 0 且 e > 0 时, w 必须被 clip 到 >= 0; e 必须清零."""
        P = _small_csr()
        indptr, indices, data = _get(P)
        initial = data.copy()
        small_w_idx = np.argsort(initial)[:2]
        e = np.zeros_like(data)
        e[small_w_idx[0]] = initial[small_w_idx[0]] * 4 + 1
        e[small_w_idx[1]] = initial[small_w_idx[1]] * 4 + 1
        eta_r = -1.0
        _apply_update(data, e, eta_r)
        self.assertTrue((data >= 0).all(), f"weights not clipped: {data}")
        self.assertEqual(data[small_w_idx[0]], 0.0)
        self.assertEqual(data[small_w_idx[1]], 0.0)
        other = [i for i in range(len(data)) if i not in small_w_idx]
        np.testing.assert_allclose(data[other], initial[other], atol=1e-12)
        self.assertTrue((e == 0).all(), f"e not cleared: {e}")

    def test_increases_weights_for_positive_reward(self):
        P = _small_csr()
        indptr, indices, data = _get(P)
        initial = data.copy()
        e = np.full_like(data, 0.5)
        eta_r = 1.0
        _apply_update(data, e, eta_r)
        np.testing.assert_allclose(data, initial + 0.5, atol=1e-12)
        self.assertTrue((e == 0).all())


class TestRenormColumns(unittest.TestCase):
    def test_restores_unit_column_sum(self):
        """Hebbian 更新 + 按列 renorm 后, 每列出边权重和 = 1."""
        P = _small_csr()
        indptr, indices, data = _get(P)
        e = np.array([0.3, 0.1, -0.2, 0.4, 0.0, -0.5])
        eta_r = 0.5
        _apply_update(data, e, eta_r)
        col_sums = _compute_col_sums(data, indices, 4)
        _renorm_columns(data, indices, col_sums)
        _assert_column_normalized(self, data, indices, 4,
                                  msg="after apply_update + renorm")

    def test_asymmetric_row_vs_column_normalization(self):
        """故意行和 ≠ 列和: 验证 renorm 真的按 pre (列) 归一化, 而不是按 post (行).

        跑一整轮 accumulate -> apply_update -> compute_col_sums -> renorm
        之后断言:
          - 每列 (有出边的) 和 ≈ 1
          - 行和不一定 = 1 (故意构造不均匀)
          - 拓扑 (indptr / indices) 完全不变
          - 权重 ≥ 0
        """
        P0 = _asymmetric_csr()
        indptr, indices, data = _get(P0)
        e = np.zeros(P0.nnz, dtype=np.float64)
        ever = np.zeros(P0.nnz, dtype=np.uint8)

        rng_p = np.random.RandomState(7)
        rng_q = np.random.RandomState(8)
        rng_r = np.random.RandomState(9)
        n_post = P0.shape[0]
        n_pre = P0.shape[1]

        for _ in range(10):
            x_pre = rng_p.rand(n_pre)
            x_post = rng_q.rand(n_post)
            _accumulate_e(e, ever, indptr, indices, x_pre, x_post)
            r = float(rng_r.uniform(-1, 1))
            _apply_update(data, e, 1e-3 * r)
            col_sums = _compute_col_sums(data, indices, n_pre)
            _renorm_columns(data, indices, col_sums)

        # 1) 权重仍 ≥ 0
        self.assertTrue((data >= 0).all(),
                        f"negative weights leaked through: min={data.min()}")

        # 2) 每列 (有出边的) 和 = 1 (按 indices==j 求和)
        for j in range(n_pre):
            mask = (indices == j)
            n_in = int(mask.sum())
            if n_in == 0:
                continue
            s = float(data[mask].sum())
            self.assertAlmostEqual(s, 1.0, places=9,
                                   msg=f"col {j} sum = {s} (n_in={n_in})")

        # 3) 行和不一定 = 1 (故意构造). 至少确认行和是变化的、非平凡的:
        row_sums = np.zeros(n_post, dtype=np.float64)
        for post in range(n_post):
            for k in range(indptr[post], indptr[post + 1]):
                row_sums[post] += data[k]
        # row 0 至少有 col 0/1/4 三条入边, 应明显大于 row 3 的 0 (row 3 本来就没入边)
        self.assertGreater(row_sums[0], 1.0,
                           f"row 0 sum should be > 1 since it has multiple "
                           f"incoming cols; got {row_sums[0]}")
        self.assertAlmostEqual(row_sums[3], 0.0, places=9,
                               msg=f"row 3 should stay 0 (no incoming edges), "
                                   f"got {row_sums[3]}")

        # 4) 拓扑完全不变: indptr / indices 必须 == P0 的原始值
        np.testing.assert_array_equal(indptr, P0.indptr.astype(np.int64))
        np.testing.assert_array_equal(indices, P0.indices.astype(np.int64))

    def test_skips_dead_columns_without_error(self):
        """把 col 1 的两条边精确更新到 0, 验证归一化跳过 (不抛除零)."""
        P = _small_csr()
        indptr, indices, data = _get(P)
        e = np.zeros_like(data)
        # col 1 在 data 里是 indices==1 的两条 (data[2]=0.7, data[3]=0.3)
        col1_mask = (indices == 1)
        e[col1_mask] = -data[col1_mask]    # eta_r=1 -> w + 1*(-w) = 0
        eta_r = 1.0
        _apply_update(data, e, eta_r)
        # col 1 全 0
        col1_after = data[col1_mask]
        self.assertTrue((col1_after == 0).all(), f"col1 = {col1_after}")
        # 归一化应当不抛除零
        col_sums = _compute_col_sums(data, indices, 4)
        _renorm_columns(data, indices, col_sums)
        # 其他列仍归一化
        for j in [0, 2]:
            n_in = int((indices == j).sum())
            self.assertGreater(n_in, 0)
            s = float(data[indices == j].sum())
            self.assertAlmostEqual(s, 1.0, places=9,
                                   msg=f"col {j} sum = {s}")
        # col 1 仍 0
        self.assertTrue((data[col1_mask] == 0).all())


class TestAccumulateE(unittest.TestCase):
    def test_records_only_nonzero_products(self):
        P = _small_csr()
        indptr, indices, _ = _get(P)
        e = np.zeros(P.nnz, dtype=np.float64)
        ever = np.zeros(P.nnz, dtype=np.uint8)
        x_pre = np.array([0.0, 1.0, 0.0, 1.0])
        x_post = np.array([1.0, 0.0, 0.0, 1.0])
        _accumulate_e(e, ever, indptr, indices, x_pre, x_post)
        expected_e = np.zeros(P.nnz, dtype=np.float64)
        for pre in range(4):
            for k in range(indptr[pre], indptr[pre + 1]):
                post = indices[k]
                expected_e[k] = x_post[post] * x_pre[pre]
        np.testing.assert_allclose(e, expected_e)
        expected_ever = (expected_e != 0).astype(np.uint8)
        np.testing.assert_array_equal(ever, expected_ever)

    def test_sums_across_calls(self):
        P = _small_csr()
        indptr, indices, _ = _get(P)
        e = np.zeros(P.nnz, dtype=np.float64)
        ever = np.zeros(P.nnz, dtype=np.uint8)
        x_pre = np.array([0.0, 1.0, 0.0, 1.0])
        x_post = np.array([1.0, 0.0, 0.0, 1.0])
        _accumulate_e(e, ever, indptr, indices, x_pre, x_post)
        first = e.copy()
        _accumulate_e(e, ever, indptr, indices, x_pre, x_post)
        np.testing.assert_allclose(e, 2.0 * first)
        expected_ever = (first != 0).astype(np.uint8)
        np.testing.assert_array_equal(ever, expected_ever)


class TestFullPipeline(unittest.TestCase):
    def test_keeps_weights_nonneg_and_columns_normalized(self):
        """串联 20 轮 accumulate -> apply_update -> compute_col_sums -> renorm:
        末态权重全部 >= 0 且每列和 = 1."""
        P = _small_csr()
        indptr, indices, data = _get(P)
        e = np.zeros(P.nnz, dtype=np.float64)
        ever = np.zeros(P.nnz, dtype=np.uint8)
        rng_p = np.random.RandomState(0)
        rng_q = np.random.RandomState(1)
        rng_r = np.random.RandomState(2)

        for _ in range(20):
            x_pre = rng_p.rand(4)
            x_post = rng_q.rand(4)
            _accumulate_e(e, ever, indptr, indices, x_pre, x_post)
            r = float(rng_r.uniform(-1, 1))
            _apply_update(data, e, 1e-3 * r)
            col_sums = _compute_col_sums(data, indices, 4)
            _renorm_columns(data, indices, col_sums)

        self.assertTrue((data >= 0).all(), "negative weights leaked through")
        _assert_column_normalized(self, data, indices, 4,
                                  msg="after 20 rounds pipeline")


if __name__ == "__main__":
    unittest.main()
