# -*- coding: utf-8 -*-
"""
benchmark_v0.py — CSR 稀疏矩阵信号传播基准测试

模拟规模: 2000 / 6000 / 10000 / 150000 个神经元
测量指标: 构建耗时、传播耗时、内存占用、每秒处理连接数 (conn/s)

仅依赖: numpy, scipy, psutil (项目 .venv 已安装)
不做任何训练，只做一次前向传播基准。
"""

import time
import numpy as np
import psutil
import scipy.sparse as sp

# ---------------- 参数 ----------------
SIZES = [2000, 6000, 10000, 150000]   # 神经元数量
AVG_DEGREE = 100              # 每个神经元的平均扇出连接数
STEPS = 100                   # 传播步数
SEED = 42                     # 固定随机种子，保证可复现

rng = np.random.default_rng(SEED)
proc = psutil.Process()


def fmt_bytes(n: float) -> str:
    """字节数转可读字符串"""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def build_csr(n: int, avg_degree: int) -> sp.csr_matrix:
    """构建随机稀疏连接矩阵 (CSR), 权重在 [0, 1)"""
    total_conn = n * avg_degree
    rows = rng.integers(0, n, size=total_conn, dtype=np.int64)
    cols = rng.integers(0, n, size=total_conn, dtype=np.int64)
    vals = rng.random(total_conn, dtype=np.float64)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    return A.tocsr()


def propagate(A: sp.csr_matrix, x: np.ndarray, steps: int) -> np.ndarray:
    """信号传播: x <- relu(A @ x) / (1 + ||A x||_inf), 归一化防发散"""
    for _ in range(steps):
        y = A @ x
        np.maximum(y, 0.0, out=y)
        m = np.max(y) if y.size else 0.0
        if m > 1.0:
            y /= m
        x = y
    return x


def bench(n: int) -> dict:
    """对规模 n 执行一次完整基准"""
    mem0 = proc.memory_info().rss

    # 1. 构建矩阵
    t0 = time.perf_counter()
    A = build_csr(n, AVG_DEGREE)
    build_s = time.perf_counter() - t0
    nnz = A.nnz

    # 2. 传播
    x = rng.random(n, dtype=np.float64)
    mem1 = proc.memory_info().rss
    t0 = time.perf_counter()
    x = propagate(A, x, STEPS)
    sim_s = time.perf_counter() - t0
    mem2 = proc.memory_info().rss

    total_conn = nnz * STEPS
    return {
        "n": n,
        "nnz": nnz,
        "build_s": build_s,
        "sim_s": sim_s,
        "mem_matrix": A.data.nbytes + A.indices.nbytes + A.indptr.nbytes,
        "mem_peak": max(mem1, mem2) - mem0,
        "conns_per_s": total_conn / sim_s,
        "check": float(np.abs(x).max()),
    }


def main() -> None:
    print("=" * 74)
    print("CSR 稀疏矩阵信号传播基准  |  平均扇出=%d  传播步数=%d  seed=%d"
          % (AVG_DEGREE, STEPS, SEED))
    print("=" * 74)

    # CPU 概况
    cpu_name = ""
    try:
        import platform
        cpu_name = platform.processor() or platform.machine()
    except Exception:
        pass
    print(f"CPU: {cpu_name}  物理核: {psutil.cpu_count(logical=False)}"
          f"  逻辑核: {psutil.cpu_count(logical=True)}")
    print()

    hdr = (f"{'神经元':>8} {'总连接数':>12} {'建矩阵(s)':>10} {'传播(s)':>9} "
           f"{'矩阵内存':>10} {'峰值增量':>10} {'conn/s':>12}")
    print(hdr)
    print("-" * 74)

    for n in SIZES:
        r = bench(n)
        print(f"{r['n']:>8} {r['nnz']:>12,} {r['build_s']:>10.3f} {r['sim_s']:>9.3f} "
              f"{fmt_bytes(r['mem_matrix']):>10} {fmt_bytes(r['mem_peak']):>10} "
              f"{r['conns_per_s']:>12,.0f}")

    print("-" * 74)
    print("说明: conn/s = 总连接数 x 传播步数 / 传播耗时; 内存为进程 RSS 相对基准前增量")
    print("数值校验(最终状态最大值, 应为有限值且<=1): ", end="")
    last = bench(SIZES[-1])
    print(f"{last['check']:.6f}")


if __name__ == "__main__":
    main()
