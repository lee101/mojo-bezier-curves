"""Where the per-curve drop-in path spends its time.

The eleventh benchmark row -- 2048 `Curve.evaluate_multi` calls on a 256 point
grid -- is the one case at or below upstream parity, and upstream is a compiled
Cython object on the other side of it. This breaks that row into its parts so
the cost of each is visible: the ctypes call, the buffer allocation, the result
reshape and the Fortran-order transpose are timed on their own, and the kernel
alone is estimated from a single large call.

Run it under the same machine-wide lock `pixi run bench` takes, so a concurrent
job cannot distort the numbers:

    flock /tmp/mojo-bench.lock python bench/profile_curve_path.py
"""

import cProfile
import pstats
import time

import numpy as np

import mojo_bezier_curves as mbc
from mojo_bezier_curves import _lib
from mojo_bezier_curves.curve_helpers import _column

REPEATS = 7


def _time(function):
    for _ in range(2):
        function()
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def _family(count, dim, degree, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((count, dim, degree + 1)))


def main():
    count, degree, num_s = 2048, 6, 256
    nodes = _family(count, 2, degree, seed=11)
    values = np.linspace(0.0, 1.0, num_s)
    curves = [mbc.Curve(nodes[i], degree) for i in range(count)]
    batches = [curve._batch for curve in curves]
    addr = _lib._addr(values)
    library = _lib.lib

    # A single call, one curve, this grid: the kernel with no Python around it.
    one = batches[0]
    size = one.num_curves * one.dim * num_s
    scratch = np.empty(size + one.degree + 1, dtype=np.float64)
    raw_call = lambda: library.bc_evaluate_multi_s(
        one.address, addr, 1, 2, degree, num_s, scratch.ctypes.data
    )
    raw = _time(raw_call)

    # The same kernel over a grid 512 times longer, to get its per-output cost
    # without the fixed per-call overhead in the way.
    long_values = np.linspace(0.0, 1.0, num_s * 512)
    long_addr = _lib._addr(long_values)
    long_size = 1 * 2 * long_values.size
    long_scratch = np.empty(long_size + degree + 1, dtype=np.float64)
    long_call = lambda: library.bc_evaluate_multi_s(
        one.address, long_addr, 1, 2, degree, long_values.size,
        long_scratch.ctypes.data,
    )
    long_time = _time(long_call)
    kernel = long_time / 512.0

    allocate = _time(lambda: np.empty(size + one.degree + 1, dtype=np.float64))
    result = scratch[:size].reshape((1, 2, num_s))
    column = result[0]
    transpose = _time(lambda: _column(result))
    reshape = _time(
        lambda: scratch[:size].reshape((1, 2, num_s))
    )
    grid_check = _time(lambda: _lib._grid(values))
    full = lambda: [curve.evaluate_multi(values) for curve in curves]
    via_lib = lambda: [
        _lib.evaluate_multi_s(batch, values) for batch in batches
    ]
    whole = _time(full)
    lib_only = _time(via_lib)

    print(f"curve path, {count} x {num_s} s, degree {degree}\n")
    print(f"{'whole Curve.evaluate_multi loop':<40} {whole * 1e3:8.3f} ms")
    print(f"{'  per call':<40} {whole / count * 1e6:8.3f} us")
    print(f"{'_lib.evaluate_multi_s loop':<40} {lib_only * 1e3:8.3f} ms")
    print(f"{'  per call':<40} {lib_only / count * 1e6:8.3f} us")
    print()
    print(f"{'ctypes call + kernel (1 curve)':<40} {raw * 1e6:8.3f} us")
    print(f"{'kernel only (extrapolated)':<40} {kernel * 1e6:8.3f} us")
    print(f"{'np.empty(output + spare)':<40} {allocate * 1e6:8.3f} us")
    print(f"{'prefix slice + reshape':<40} {reshape * 1e6:8.3f} us")
    print(f"{'_column (Fortran transpose copy)':<40} {transpose * 1e6:8.3f} us")
    print(f"{'_grid (already a grid)':<40} {grid_check * 1e6:8.3f} us")
    print()

    profiler = cProfile.Profile()
    profiler.enable()
    full()
    profiler.disable()
    print("cProfile, whole loop (profiler overhead inflates every line)")
    pstats.Stats(profiler).sort_stats("tottime").print_stats(18)


if __name__ == "__main__":
    main()
