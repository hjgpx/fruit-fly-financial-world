# -*- coding: utf-8 -*-
"""
benchmark_v1.py — Numba 并行 SpMV vs SciPy 单线程 SpMV 对比 (仅 150,000 神经元档)

条件与 benchmark_v0.py 的 150k 档完全一致:
  - 神经元数 150,000, 平均扇出 100, 传播 100 步, seed=42
  - 相同的 CSR 矩阵构建方式 (随机连接, 权重 [0,1), sum_duplicates)
  - 相同的传播语义: x <- relu(A x); 若 max>1 则除以 max

对比对象:
  - SciPy:   A @ x (scipy 内部单线程 SpMV)
  - Numba:   parallel=True 的行并行 SpMV 内核, 线程数 1/10/20/40

测量: 总耗时、conn/s、相对 SciPy 的加速倍数; 并验证计算结果一致。
仅依赖: numpy, scipy, numba (项目 .venv 已安装)
"""

import time
import numpy as np
import scipy.sparse as sp
from numba import njit, prange, set_num_threads

# ---------------- 参数 ----------------
N = 150_000          # 神经元数量
AVG_DEGREE = 100     # 平均扇出
STEPS = 100          # 传播步数
SEED = 42            # 随机种子
THREAD_LIST = [1, 10, 20, 40]   # 待测线程数


# ---------------- 矩阵与传播 (与 v0 相同的构建方式) ----------------
def build_csr(n: int, avg_degree: int) -> sp.csr_matrix:
    rng = np.random.default_rng(SEED)
    total_conn = n * avg_degree
    rows = rng.integers(0, n, size=total_conn, dtype=np.int64)
    cols = rng.integers(0, n, size=total_conn, dtype=np.int64)
    vals = rng.random(total_conn, dtype=np.float64)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    return A.tocsr()


def propagate_scipy(A: sp.csr_matrix, x: np.ndarray, steps: int) -> np.ndarray:
    """SciPy 版传播: 与 benchmark_v0.propagate 语义完全一致"""
    for _ in range(steps):
        y = A @ x
        np.maximum(y, 0.0, out=y)
        m = np.max(y) if y.size else 0.0
        if m > 1.0:
            y /= m
        x = y
    return x


# ---------------- Numba 并行 SpMV 内核 ----------------
@njit(parallel=True, cache=False)
def _spmv(indptr, indices, data, x, y, n):
    """行并行 CSR SpMV: y[i] = sum_j A[i,j] * x[j]"""
    for i in prange(n):
        s = 0.0
        for k in range(indptr[i], indptr[i + 1]):
            s += data[k] * x[indices[k]]
        y[i] = s


def propagate_numba(indptr, indices, data, x, steps: int, n: int) -> np.ndarray:
    """Numba 版传播: 每步调用并行 SpMV, 归一化语义与 SciPy 版一致"""
    x = x.copy()
    y = np.empty_like(x)
    for _ in range(steps):
        _spmv(indptr, indices, data, x, y, n)
        np.maximum(y, 0.0, out=y)
        m = np.max(y) if y.size else 0.0
        if m > 1.0:
            y /= m
        x, y = y, x   # 交换缓冲区, 避免 copy
    return x.copy()


def main() -> None:
    print("=" * 78)
    print(f"Numba 并行 SpMV vs SciPy 单线程  |  N={N:,}  扇出={AVG_DEGREE}  "
          f"步数={STEPS}  seed={SEED}")
    print("=" * 78)

    # 1. 构建矩阵 (只建一次, 所有方法共用)
    t0 = time.perf_counter()
    A = build_csr(N, AVG_DEGREE)
    build_s = time.perf_counter() - t0
    nnz = A.nnz
    print(f"CSR 矩阵构建完成: {build_s:.3f}s, 连接数 nnz={nnz:,}, "
          f"内存 {(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes)/1024/1024:.1f} MB")
    print()

    # 相同的初始状态向量
    rng = np.random.default_rng(SEED)
    x0 = rng.random(N, dtype=np.float64)

    indptr = A.indptr.astype(np.int64)
    indices = A.indices.astype(np.int64)
    data = A.data

    # 2. SciPy 基准 (跑一次, 结果作为参照)
    t0 = time.perf_counter()
    x_scipy = propagate_scipy(A, x0.copy(), STEPS)
    t_scipy = time.perf_counter() - t0
    conns = nnz * STEPS
    print(f"{'方案':<22} {'总耗时(s)':>10} {'conn/s':>16} {'加速倍数':>10}")
    print("-" * 78)
    print(f"{'SciPy 单线程 (基线)':<22} {t_scipy:>10.3f} {conns/t_scipy:>16,.0f} {1.0:>9.1f}x")

    # 3. Numba 并行, 不同线程数
    results = {}
    for nthreads in THREAD_LIST:
        set_num_threads(nthreads)
        # 预热: 先编译/触发 JIT (不计时)
        propagate_numba(indptr, indices, data, x0.copy(), 3, N)

        t0 = time.perf_counter()
        x_nb = propagate_numba(indptr, indices, data, x0.copy(), STEPS, N)
        t_nb = time.perf_counter() - t0
        speedup = t_scipy / t_nb
        results[nthreads] = (t_nb, x_nb)
        print(f"{f'Numba parallel x{nthreads}':<22} {t_nb:>10.3f} "
              f"{conns/t_nb:>16,.0f} {speedup:>9.1f}x")

    # 4. 结果一致性验证 (相对 SciPy 参照)
    print("-" * 78)
    print("结果一致性验证 (与 SciPy 参照比较):")
    ref = np.abs(x_scipy).max()
    print(f"  SciPy 参照: |x|_max = {ref:.6f} (有限值: {np.isfinite(x_scipy).all()})")
    for nthreads in THREAD_LIST:
        t_nb, x_nb = results[nthreads]
        max_diff = float(np.max(np.abs(x_nb - x_scipy)))
        rel_err = max_diff / max(ref, 1e-30)
        ok = np.allclose(x_nb, x_scipy, rtol=1e-10, atol=1e-12)
        print(f"  x{nthreads:<3} 线程: max|x_nb - x_scipy| = {max_diff:.3e} "
              f"(相对误差 {rel_err:.2e})  一致: {'是' if ok else '否'}")
    print()
    print("说明: conn/s = nnz x 步数 / 传播总耗时; 加速倍数 = t_scipy / t_numba")
    print("      浮点求和顺序不同会导致 ~1e-15 量级的微小差异, 属正常现象")


if __name__ == "__main__":
    main()
