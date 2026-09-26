"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64`` for addresses; ``c_int``
truncates them and segfaults.

Curves are laid out ``(num_curves, dim, degree + 1)`` in C order, which is the
transpose of the ``(dim, degree + 1)`` Fortran-order array the `bezier`
package uses. Keeping the batch dimension outermost is what lets one call
evaluate a whole family of curves.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-bezier-curves.so"

_I64 = ctypes.c_int64
_F64 = ctypes.c_double


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    signatures = {
        "bc_evaluate_multi": [_I64] * 8,
        "bc_evaluate_de_casteljau": [_I64] * 8,
        "bc_split_at": [_I64] * 5 + [_F64] * 4 + [_I64, _I64],
        "bc_elevate": [_I64] * 5,
        "bc_hodograph": [_I64] * 8,
        "bc_curvature": [_I64] * 8,
        "bc_newton_refine": [_I64] * 10,
        "bc_length": [_I64, _I64, _I64, _I64, _I64, _I64, _I64],
    }
    for name, argtypes in signatures.items():
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = argtypes
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def as_batch(nodes, dim: int | None = None, degree: int | None = None):
    """Normalise input to a ``(num_curves, dim, degree + 1)`` float64 array.

    Accepts the ``(dim, degree + 1)`` single-curve layout of the `bezier`
    package, a ``(num_curves, dim, degree + 1)`` family, or an already-batched
    array; anything with a different rank is rejected rather than guessed at.
    """
    array = np.ascontiguousarray(nodes, dtype=np.float64)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(
            f"nodes must be (dim, degree + 1) or (curves, dim, degree + 1), got rank {nodes.ndim}"
        )
    if degree is not None and array.shape[2] - 1 != degree:
        raise ValueError(f"expected degree {degree}, got {array.shape[2] - 1}")
    if dim is not None and array.shape[1] != dim:
        raise ValueError(f"expected dimension {dim}, got {array.shape[1]}")
    if array.shape[2] < 1:
        raise ValueError("a curve needs at least one node")
    if not np.isfinite(array).all():
        raise ValueError("nodes must be finite")
    return array


def _s(s_vals) -> np.ndarray:
    return np.ascontiguousarray(np.atleast_1d(s_vals), dtype=np.float64)


def evaluate_multi(nodes, s_vals, de_casteljau: bool = False) -> np.ndarray:
    """Evaluate the curves on a parameter grid, result ``(curves, dim, num_s)``.

    The default is the VS / modified Horner recurrence; `de_casteljau` selects
    the classical triangle, which agrees to rounding and is the safer choice on
    a badly conditioned control polygon.
    """
    curves = as_batch(nodes)
    grid = _s(s_vals)
    num_curves, dim, width = curves.shape
    degree = width - 1
    dst = np.empty((num_curves, dim, grid.size), dtype=np.float64)
    work = np.empty(dim * width, dtype=np.float64)
    entry = lib.bc_evaluate_de_casteljau if de_casteljau else lib.bc_evaluate_multi
    entry(
        _addr(curves), _addr(grid), _addr(work), num_curves, dim, degree,
        grid.size, _addr(dst),
    )
    return dst


def subdivide(nodes) -> tuple[np.ndarray, np.ndarray]:
    """Split every curve in half at ``s = 1/2``; returns ``(left, right)``.

    Both halves have the same degree and node count as the input, and they meet
    at the midpoint: ``left[..., -1] == right[..., 0]``.
    """
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    left = np.empty_like(curves)
    right = np.empty_like(curves)
    work = np.empty(dim * width, dtype=np.float64)
    lib.bc_split_at(
        _addr(curves), _addr(work), num_curves, dim, degree,
        ctypes.c_double(0.5), ctypes.c_double(0.5),
        ctypes.c_double(0.5), ctypes.c_double(0.5),
        _addr(left), _addr(right),
    )
    return left, right


def _edges(nodes, t: float) -> tuple[np.ndarray, np.ndarray]:
    """The two restrictions of the curves at ``t``, as one kernel call.

    The left edge of the de Casteljau triangle at ``(1 - t, t)`` is the node
    set of the curve on ``[0, t]``; the right edge of the same triangle is the
    node set of the curve on ``[t, 1]``. Both keep the degree, so both have the
    input's node count.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    first = np.empty_like(curves)
    second = np.empty_like(curves)
    work = np.empty(dim * width, dtype=np.float64)
    lib.bc_split_at(
        _addr(curves), _addr(work), num_curves, dim, degree,
        ctypes.c_double(1.0 - t), ctypes.c_double(t),
        ctypes.c_double(1.0 - t), ctypes.c_double(t),
        _addr(first), _addr(second),
    )
    return first, second


def restrict(nodes, t: float, from_end: bool = False) -> np.ndarray:
    """Nodes of the curve restricted to ``[0, t]``, or to ``[t, 1]``.

    The restriction keeps the degree, so the result has the same node count as
    the input; it is a restriction, not a re-parameterisation of anything.
    """
    first, second = _edges(nodes, t)
    return second if from_end else first


def subdivide(nodes) -> tuple[np.ndarray, np.ndarray]:
    """Split every curve in half at ``s = 1/2``; returns ``(left, right)``.

    Both halves have the same degree and node count as the input, and they meet
    at the midpoint: ``left[..., -1] == right[..., 0]``.
    """
    return _edges(nodes, 0.5)


def specialize(nodes, start: float, end: float) -> np.ndarray:
    """Re-parameterise every curve from ``[start, end]`` onto ``[0, 1]``.

    Composed from two de Casteljau edge extractions. Restricting to ``[t, 1]``
    re-parameterises with ``s = t + (1 - t) u``, so restricting that to
    ``[0, (end - start) / (1 - start)]`` lands on ``[start, end]``. The
    degenerate ``start == 1`` case reduces to the constant ``B(1)``. The result
    has the same degree and node count as the input.
    """
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("require 0 <= start <= end <= 1")
    curves = as_batch(nodes)
    tail = restrict(curves, start, from_end=True)
    if start >= 1.0:
        beta = 0.0
    else:
        beta = (end - start) / (1.0 - start)
    return restrict(tail, beta)


def elevate(nodes) -> np.ndarray:
    """Degree-elevated curves, one node wider than the input."""
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    dst = np.empty((num_curves, dim, width + 1), dtype=np.float64)
    lib.bc_elevate(_addr(curves), num_curves, dim, width - 1, _addr(dst))
    return dst


def hodograph(nodes, s_vals) -> np.ndarray:
    """Tangent vectors ``B'(s)``, result ``(curves, dim, num_s)``."""
    curves = as_batch(nodes)
    grid = _s(s_vals)
    num_curves, dim, width = curves.shape
    degree = width - 1
    if degree < 1:
        raise ValueError("a curve of degree 0 has no tangent")
    dst = np.empty((num_curves, dim, grid.size), dtype=np.float64)
    work = np.empty(dim * width, dtype=np.float64)
    lib.bc_hodograph(
        _addr(curves), _addr(grid), _addr(work), num_curves, dim, degree,
        grid.size, _addr(dst),
    )
    return dst


def curvature(nodes, s_vals) -> np.ndarray:
    """Signed curvature of planar curves, result ``(curves, num_s)``."""
    curves = as_batch(nodes, dim=2)
    grid = _s(s_vals)
    num_curves, dim, width = curves.shape
    degree = width - 1
    dst = np.empty((num_curves, grid.size), dtype=np.float64)
    work = np.empty(2 * dim * width + 4, dtype=np.float64)
    lib.bc_curvature(
        _addr(curves), _addr(grid), _addr(work), num_curves, dim, degree,
        grid.size, _addr(dst),
    )
    return dst


def newton_refine(nodes, points, s0, iterations: int = 8) -> np.ndarray:
    """Refine ``s`` towards ``B(s) = point`` for many (curve, point) pairs.

    `points` is ``(curves, num_points, dim)`` and `s0`` is
    ``(curves, num_points)``; the result has the shape of `s0`.
    """
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    pts = np.ascontiguousarray(points, dtype=np.float64)
    starts = np.ascontiguousarray(s0, dtype=np.float64)
    if pts.ndim == 2:
        pts = pts[:, np.newaxis, :]
    if pts.ndim != 3 or pts.shape[0] != num_curves or pts.shape[2] != dim:
        raise ValueError("points must be (curves, num_points, dim)")
    if starts.shape != pts.shape[:2]:
        raise ValueError("s0 must be (curves, num_points)")
    num_points = pts.shape[1]
    dst = np.empty((num_curves, num_points), dtype=np.float64)
    work = np.empty(dim * (degree + 2), dtype=np.float64)
    lib.bc_newton_refine(
        _addr(curves), _addr(pts), _addr(starts), _addr(work), num_curves,
        num_points, dim, degree, int(iterations), _addr(dst),
    )
    return dst


def length(nodes, panels: int = 1024) -> np.ndarray:
    """Arc length by composite Simpson, result ``(curves,)``.

    Lines are exact. For higher degrees the answer is a quadrature with
    `panels` subintervals, so it converges as the panel count grows rather than
    being exact; the tests pin the convergence.
    """
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    dst = np.empty(num_curves, dtype=np.float64)
    if width == 1:
        dst[:] = 0.0
        return dst
    work = np.empty(2 * dim * width, dtype=np.float64)
    lib.bc_length(
        _addr(curves), _addr(work), num_curves, dim, width - 1, int(panels), _addr(dst)
    )
    return dst
