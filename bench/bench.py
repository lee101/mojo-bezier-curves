"""Correctness-gated benchmark for mojo-bezier-curves.

Every case checks agreement with an independent NumPy reference before timing,
so a regression in the Mojo kernels shows up as a correctness failure rather
than a suspiciously good number. The upstream `bezier` package is not installed
in this environment, so the baselines are the fastest reasonable NumPy
formulations of the same mathematics -- vectorised over the parameter grid, not
Python loops that NumPy would never be asked to run.
"""

from __future__ import annotations

import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_bezier_curves as mbc  # noqa: E402


def _time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bernstein_weights(degree: int, s_vals: np.ndarray) -> np.ndarray:
    """The Bernstein basis matrix, built with exact integer binomials."""
    grid = np.asarray(s_vals, dtype=np.float64)
    return np.array(
        [
            [math.comb(degree, j) * (1.0 - t) ** (degree - j) * t**j for t in grid]
            for j in range(degree + 1)
        ]
    )


def numpy_evaluate(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return nodes @ weights


def numpy_subdivide(nodes: np.ndarray):
    degree = nodes.shape[1] - 1
    triangle = np.array(nodes, dtype=np.float64)
    left = [triangle[:, 0]]
    right = [triangle[:, degree]]
    for k in range(1, degree + 1):
        triangle = 0.5 * (triangle[:, :-1] + triangle[:, 1:])
        left.append(triangle[:, 0])
        right.append(triangle[:, -1])
    return np.array(left).T, np.array(right[::-1]).T


def monotone_family(count, degree, seed=0):
    """Curves whose first coordinate is the parameter: Newton converges on these.

    A randomly shaped cubic loops back on itself, and then `B(s) = p` has several
    roots or none, which is a property of the curve rather than of the kernel.
    """
    rng = np.random.default_rng(seed)
    x = np.tile(np.linspace(0.0, 1.0, degree + 1), (count, 1))
    j = np.arange(degree + 1)[np.newaxis, :]
    phase = 0.3 + 1.4 * rng.random((count, 1))
    y = ((j + phase) / (degree + phase)) ** 2
    return np.stack([x, y], axis=1)


def family(count, dim, degree, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((count, dim, degree + 1))


def bench_evaluate(count=256, dim=2, degree=8, num_s=2048):
    grid = np.linspace(0.0, 1.0, num_s)
    weights = bernstein_weights(degree, grid)
    nodes = family(count, dim, degree, seed=1)
    curves = mbc.Curves(nodes)

    expected = np.stack([numpy_evaluate(curve, weights) for curve in nodes])
    got = curves.evaluate_multi(grid)
    assert np.allclose(got, expected, rtol=1e-11, atol=1e-12), "evaluate mismatch"

    def reference():
        return np.stack([numpy_evaluate(curve, weights) for curve in nodes])

    label = f"evaluate {count} curves x {num_s} s (deg {degree})"
    return label, _time(reference), _time(lambda: curves.evaluate_multi(grid))


def bench_evaluate_single(num_s=65536, degree=12):
    grid = np.linspace(0.0, 1.0, num_s)
    weights = bernstein_weights(degree, grid)
    nodes = family(1, 2, degree, seed=2)[0]
    curve = mbc.Curve(nodes)
    got = curve.evaluate_multi(grid)
    assert np.allclose(got, numpy_evaluate(nodes, weights), rtol=1e-11, atol=1e-12)

    label = f"evaluate one curve x {num_s} s (deg {degree})"
    return label, _time(lambda: numpy_evaluate(nodes, weights)), _time(
        lambda: curve.evaluate_multi(grid)
    )


def bench_subdivide(count=512, dim=2, degree=10):
    nodes = family(count, dim, degree, seed=3)
    curves = mbc.Curves(nodes)
    left, right = curves.subdivide()
    for index in (0, count // 2, count - 1):
        expect_left, expect_right = numpy_subdivide(nodes[index])
        assert np.allclose(left[index].nodes, expect_left, rtol=1e-14, atol=0.0)
        assert np.allclose(right[index].nodes, expect_right, rtol=1e-14, atol=0.0)

    def reference():
        return [numpy_subdivide(curve) for curve in nodes]

    label = f"subdivide {count} curves (deg {degree})"
    return label, _time(reference), _time(lambda: curves.subdivide())


def bench_elevate(count=4096, dim=2, degree=8):
    nodes = family(count, dim, degree, seed=4)
    curves = mbc.Curves(nodes)
    got = curves.elevate()
    for index in (0, count // 3, count - 1):
        width = degree + 1
        expected = np.empty((dim, width + 1))
        expected[:, 0] = nodes[index][:, 0]
        expected[:, -1] = nodes[index][:, -1]
        for j in range(1, width):
            expected[:, j] = (
                j * nodes[index][:, j - 1] + (width - j) * nodes[index][:, j]
            ) / width
        assert np.allclose(got[index].nodes, expected, rtol=1e-13, atol=0.0)

    def reference():
        out = np.empty((count, dim, degree + 2))
        out[:, :, 0] = nodes[:, :, 0]
        out[:, :, -1] = nodes[:, :, -1]
        for j in range(1, degree + 1):
            out[:, :, j] = (
                j * nodes[:, :, j - 1] + (degree + 1 - j) * nodes[:, :, j]
            ) / (degree + 1)
        return out

    label = f"elevate {count} curves (deg {degree})"
    return label, _time(reference), _time(lambda: curves.elevate())


def bench_newton(count=64, degree=6, num_points=4096):
    nodes = monotone_family(count, degree, seed=5)
    curves = mbc.Curves(nodes)
    grid = np.linspace(0.0, 1.0, num_points)
    points = np.ascontiguousarray(curves.evaluate_multi(grid).transpose(0, 2, 1))
    # Start near the answer, the way Curve.locate does after bisection.
    starts = np.broadcast_to(np.clip(grid - 0.05, 0.0, 1.0), (count, num_points)).copy()
    refined = mbc._lib.newton_refine(curves.nodes, points, starts, 20)
    residual = max(
        float(np.abs(mbc.bernstein(nodes[b], refined[b]).T - points[b]).max())
        for b in range(count)
    )
    assert residual < 1e-9, f"newton residual {residual}"

    def reference():
        s = starts.copy()
        for _ in range(20):
            delta = np.empty((count, num_points))
            norm = np.empty((count, num_points))
            for b in range(count):
                degree_ = degree
                first = degree_ * (nodes[b][:, 1:] - nodes[b][:, :-1])
                second = degree_ * (nodes[b][:, 2:] - 2 * nodes[b][:, 1:-1]
                                    + nodes[b][:, :-2])
                batch = mbc.bernstein(nodes[b], s[b]).T
                deriv = mbc.bernstein(first, s[b]).T
                delta[b] = ((points[b] - batch) * deriv).sum(axis=1)
                norm[b] = (deriv * deriv).sum(axis=1)
            s = s + delta / norm
        return s

    label = f"newton {count} curves x {num_points} points"
    return label, _time(reference, 2), _time(
        lambda: mbc._lib.newton_refine(curves.nodes, points, starts, 20)
    )


def bench_length(count=64, degree=6, panels=4096):
    nodes = family(count, 2, degree, seed=6)
    curves = mbc.Curves(nodes)
    got = curves.length(panels=panels)
    degree_ = degree
    forward = degree_ * (nodes[:, :, 1:] - nodes[:, :, :-1])
    weights = bernstein_weights(degree_ - 1, np.linspace(0.0, 1.0, panels + 1))
    speed = np.linalg.norm(forward @ weights, axis=1)
    weights_simpson = np.ones(panels + 1)
    weights_simpson[1:-1:2] = 4.0
    weights_simpson[2:-1:2] = 2.0
    expected = (speed @ weights_simpson) / (3.0 * panels)
    assert np.allclose(got, expected, rtol=1e-12, atol=0.0), "length mismatch"

    def reference():
        return (speed @ weights_simpson) / (3.0 * panels)

    label = f"length {count} curves, {panels} panels (deg {degree})"
    return label, _time(reference), _time(lambda: curves.length(panels=panels))


def main():
    print(f"{'case':<42}{'numpy':>12}{'mojo-bezier-curves':>20}{'ratio':>9}")
    print("-" * 83)
    for fn in (bench_evaluate, bench_evaluate_single, bench_subdivide, bench_elevate,
               bench_newton, bench_length):
        label, reference, ours = fn()
        ratio = reference / ours if ours else float("nan")
        print(f"{label:<42}{reference * 1e3:>10.2f}ms{ours * 1e3:>18.2f}ms{ratio:>8.2f}x")


if __name__ == "__main__":
    main()
