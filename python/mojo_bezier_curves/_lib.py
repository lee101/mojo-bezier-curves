"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64`` for addresses; ``c_int``
truncates them and segfaults.

Everything here is batched and C-ordered: a family of curves is
``(num_curves, dim, num_nodes)`` and a result over a parameter grid is
``(num_curves, dim, num_vals)``. The upstream-shaped, Fortran-ordered,
one-curve-at-a-time API lives in :mod:`mojo_bezier_curves.curve_helpers`, which
is a thin reshape over these calls.
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

_SIGNATURES = {
    "bc_make_subdivision_matrices": [_I64, _I64, _I64],
    "bc_subdivide_nodes": [_I64] * 8,
    "bc_evaluate_multi_vs": [_I64] * 8,
    "bc_evaluate_multi_s": ([_I64] * 7, _I64),
    "bc_evaluate_multi_de_casteljau": [_I64] * 9,
    "bc_vec_size": [_I64] * 9,
    "bc_compute_length": [_I64] * 7,
    "bc_elevate_nodes": [_I64] * 5,
    "bc_de_casteljau_one_round": [_I64, _I64, _I64, _I64, _F64, _F64, _I64],
    "bc_specialize_curve": [_I64, _I64, _I64, _I64, _I64, _F64, _F64, _I64],
    "bc_evaluate_hodograph": [_I64] * 8,
    "bc_get_curvature": [_I64] * 8,
    "bc_newton_refine": [_I64] * 9,
}


#: Ceiling on the Simpson panel count `compute_length` accepts. The kernel
#: evaluates the hodograph once per panel, so this is the loop trip count and
#: an unbounded one would be an unbounded loop inside an FFI call.
MAX_LENGTH_PANELS = 1 << 20


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(f"{_LIB_PATH} not found; run `pixi run build` first")
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name, entry in _SIGNATURES.items():
        fn = getattr(lib, name)
        if isinstance(entry, tuple):
            argtypes, restype = entry
        else:
            argtypes, restype = entry, None
        fn.restype = restype

        fn.argtypes = argtypes
    return lib


lib = _load()


def _addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def as_batch(nodes, dim: int | None = None) -> np.ndarray:
    """Normalise input to a ``(num_curves, dim, num_nodes)`` float64 array.

    Accepts a single curve in upstream's ``(dim, num_nodes)`` Fortran order, a
    family already batched, or anything convertible to either. The batch index
    is the slow one, which is what makes one call cover a whole family.
    """
    array = np.asarray(nodes, dtype=np.float64)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(
            f"nodes must be (dimension, degree + 1) or "
            f"(curves, dimension, degree + 1), got rank {array.ndim}"
        )
    if array.shape[2] < 1:
        raise ValueError("a curve needs at least one node")
    if dim is not None and array.shape[1] != dim:
        raise ValueError(f"expected dimension {dim}, got {array.shape[1]}")
    if not np.isfinite(array).all():
        raise ValueError("nodes must be finite")
    return np.ascontiguousarray(array)


def _grid(values) -> np.ndarray:
    if (
        isinstance(values, np.ndarray)
        and values.ndim == 1
        and values.dtype == np.float64
        and values.flags.c_contiguous
    ):
        return values
    return np.ascontiguousarray(np.atleast_1d(values), dtype=np.float64)


class Batch:
    """A validated node buffer, with the address the kernels take.

    `int(array.ctypes.data)` builds a fresh ctypes object on every call, which
    on the per-evaluation path costs more than the kernel it feeds, and
    `as_batch` re-checks the finiteness of every node every call. Both are
    per-array facts, and the pointer of a live array never moves, so a caller
    that keeps its nodes -- `Curve`, `Curves` -- resolves them once, here.
    """

    __slots__ = ("_address", "_array", "degree", "dim", "num_curves")

    def __init__(self, nodes, dim: int | None = None):
        array = as_batch(nodes, dim)
        self._array = array
        self._address = _addr(array)
        self.num_curves, self.dim, width = array.shape
        self.degree = width - 1

    @property
    def address(self) -> int:
        return self._address

    @property
    def array(self) -> np.ndarray:
        return self._array

    @classmethod
    def trusted(cls, array: np.ndarray) -> "Batch":
        """A `Batch` for an array this package has just produced.

        `as_batch` checks that every node is finite, which is the right check
        on a caller's array and a wasted pass over a whole family on the
        output of a kernel that has just written it, every time a caller
        subdivides, elevates or specializes and makes a family of the result.
        The shape is still checked: a kernel writes three dimensions and a
        caller can still hand this something else.
        """
        array = np.asarray(array, dtype=np.float64)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        if array.ndim != 3:
            raise ValueError(
                f"nodes must be (dimension, degree + 1) or "
                f"(curves, dimension, degree + 1), got rank {array.ndim}"
            )
        array = np.ascontiguousarray(array)
        batch = cls.__new__(cls)
        batch._array = array
        batch._address = _addr(array)
        batch.num_curves, batch.dim, width = array.shape
        batch.degree = width - 1
        return batch


_MATRIX_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def make_subdivision_matrices(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """The two subdivision matrices for `degree`, built once and reused."""
    degree = int(degree)
    cached = _MATRIX_CACHE.get(degree)
    if cached is not None:
        return cached
    width = degree + 1
    left = np.zeros(width * width, dtype=np.float64)
    right = np.zeros(width * width, dtype=np.float64)
    lib.bc_make_subdivision_matrices(degree, _addr(left), _addr(right))
    left = left.reshape(width, width)
    right = right.reshape(width, width)
    _MATRIX_CACHE[degree] = (left, right)
    return left, right


def subdivide_nodes(nodes) -> tuple[np.ndarray, np.ndarray]:
    """Split every curve in half; both results keep the input's node count."""
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    left_mat, right_mat = make_subdivision_matrices(degree)
    left = np.empty_like(curves)
    right = np.empty_like(curves)
    lib.bc_subdivide_nodes(
        _addr(curves), _addr(left_mat), _addr(right_mat), num_curves, dim,
        degree, _addr(left), _addr(right),
    )
    return left, right


def evaluate_multi_vs(batch: Batch, lambda1, lambda2) -> np.ndarray:
    """VS (modified Horner) evaluation, result ``(curves, dim, num_vals)``."""
    l1 = _grid(lambda1)
    l2 = _grid(lambda2)
    if l1.shape != l2.shape:
        raise ValueError("lambda1 and lambda2 must have the same shape")
    size = batch.num_curves * batch.dim * l1.size
    # The kernel writes its binomial row into the spare doubles after the
    # output, so the result is a prefix view of a slightly larger allocation.
    buffer = np.empty(size + batch.degree + 1, dtype=np.float64)
    lib.bc_evaluate_multi_vs(
        batch.address, _addr(l1), _addr(l2), batch.num_curves, batch.dim,
        batch.degree, l1.size, _addr(buffer),
    )
    return buffer[:size].reshape((batch.num_curves, batch.dim, l1.size))


def evaluate_multi_s(batch: Batch, s_vals) -> np.ndarray:
    """`B(s)` on a parameter grid, result ``(curves, dim, num_vals)``.

    The first barycentric weight is `1 - s`, which the kernel forms in
    registers. The alternative is a NumPy temporary, a pass over it and a
    second address through the FFI on every call, for a subtraction the
    vector unit does anyway. A batch big enough to be worth a launch runs on
    the GPU instead, and `used_gpu` says whether it did.
    """
    grid = _grid(s_vals)
    size = batch.num_curves * batch.dim * grid.size
    buffer = np.empty(size + batch.degree + 1, dtype=np.float64)
    global used_gpu
    used_gpu = bool(
        lib.bc_evaluate_multi_s(
            batch.address, _addr(grid), batch.num_curves, batch.dim,
            batch.degree, grid.size, _addr(buffer),
        )
    )
    return buffer[:size].reshape((batch.num_curves, batch.dim, grid.size))


#: Whether the last :func:`evaluate_multi_s` ran on the GPU rather than on the
#: host. The two paths are the same arithmetic, so this is a report of where
#: the work went, not of what was computed.
used_gpu = False


def evaluate_multi_de_casteljau(nodes, lambda1, lambda2) -> np.ndarray:
    """de Casteljau evaluation, result ``(curves, dim, num_vals)``."""
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    l1 = _grid(lambda1)
    l2 = _grid(lambda2)
    if l1.shape != l2.shape:
        raise ValueError("lambda1 and lambda2 must have the same shape")
    work = np.empty(dim * max(degree, 1), dtype=np.float64)
    dst = np.empty((num_curves, dim, l1.size), dtype=np.float64)
    lib.bc_evaluate_multi_de_casteljau(
        _addr(curves), _addr(l1), _addr(l2), _addr(work), num_curves, dim,
        degree, l1.size, _addr(dst),
    )
    return dst


def vec_size(nodes, lambda1, lambda2) -> np.ndarray:
    """``||B(s)||_2`` per (curve, parameter), result ``(curves, num_vals)``."""
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    l1 = _grid(lambda1)
    l2 = _grid(lambda2)
    work = np.empty(dim * width, dtype=np.float64)
    dst = np.empty((num_curves, l1.size), dtype=np.float64)
    lib.bc_vec_size(
        _addr(curves), _addr(l1), _addr(l2), _addr(work), num_curves, dim,
        width - 1, l1.size, _addr(dst),
    )
    return dst


def compute_length(nodes, panels: int = 1024) -> np.ndarray:
    """Arc length per curve, result ``(curves,)``.

    `panels` is the Simpson panel count, which is also the kernel's loop trip
    count, so it is bounded here: a caller that passes a float or a
    nonsense-sized value gets an error rather than an unbounded loop inside
    a call with no way to interrupt it.
    """
    panels = int(panels)
    if panels < 1 or panels > MAX_LENGTH_PANELS:
        raise ValueError(
            f"panels must be in [1, {MAX_LENGTH_PANELS}], got {panels}"
        )
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    dst = np.empty(num_curves, dtype=np.float64)
    work = np.empty((degree + dim) * width, dtype=np.float64)
    lib.bc_compute_length(
        _addr(curves), _addr(work), num_curves, dim, degree, panels, _addr(dst)
    )
    return dst


def elevate_nodes(nodes) -> np.ndarray:
    """Degree-elevated curves, one node wider than the input."""
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    dst = np.empty((num_curves, dim, width + 1), dtype=np.float64)
    lib.bc_elevate_nodes(
        _addr(curves), num_curves, dim, width - 1, _addr(dst)
    )
    return dst


def de_casteljau_one_round(nodes, lambda1, lambda2) -> np.ndarray:
    """One de Casteljau round, result ``(curves, dim, degree)``."""
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    if degree < 1:
        raise ValueError("a curve of degree 0 has no round to perform")
    dst = np.empty((num_curves, dim, degree), dtype=np.float64)
    lib.bc_de_casteljau_one_round(
        _addr(curves), num_curves, dim, degree, ctypes.c_double(lambda1),
        ctypes.c_double(lambda2), _addr(dst),
    )
    return dst


def specialize_curve(nodes, start: float, end: float) -> np.ndarray:
    """Re-parameterise ``[start, end]`` onto ``[0, 1]``; same node count."""
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    work = np.empty(2 * dim * width, dtype=np.float64)
    dst = np.empty_like(curves)
    lib.bc_specialize_curve(
        _addr(curves), _addr(work), num_curves, dim, width - 1,
        ctypes.c_double(start), ctypes.c_double(end), _addr(dst),
    )
    return dst


def evaluate_hodograph(nodes, s_vals) -> np.ndarray:
    """Tangent vectors ``B'(s)``, result ``(curves, dim, num_s)``."""
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    if degree < 1:
        raise ValueError("a curve of degree 0 has no tangent")
    grid = _grid(s_vals)
    # `dim * degree` unscaled forward differences, `dim` accumulator doubles
    # for the scalar tail, `degree` for the binomial row.
    work = np.empty(dim * degree + dim + degree, dtype=np.float64)
    dst = np.empty((num_curves, dim, grid.size), dtype=np.float64)
    lib.bc_evaluate_hodograph(
        _addr(curves), _addr(grid), _addr(work), num_curves, dim, degree,
        grid.size, _addr(dst),
    )
    return dst


def get_curvature(nodes, tangent, s_vals) -> np.ndarray:
    """Signed curvature from precomputed tangents, result ``(curves, num_s)``.

    `tangent` is ``(curves, dim, num_s)``; the curve must be planar, as the
    cross product it is built from only exists in two dimensions.
    """
    curves = as_batch(nodes, dim=2)
    num_curves, dim, width = curves.shape
    degree = width - 1
    if degree < 1:
        raise ValueError("a curve of degree 0 has no tangent to curve along")
    tangents = np.ascontiguousarray(tangent, dtype=np.float64)
    grid = _grid(s_vals)
    if tangents.shape != (num_curves, dim, grid.size):
        raise ValueError("tangent must be (curves, 2, num_s)")
    work = np.empty(2 * dim * max(degree, 1) + dim + degree, dtype=np.float64)
    dst = np.empty((num_curves, grid.size), dtype=np.float64)
    lib.bc_get_curvature(
        _addr(curves), _addr(tangents), _addr(grid), _addr(work), num_curves,
        degree, grid.size, _addr(dst),
    )
    return dst


def newton_refine(nodes, points, s_vals) -> np.ndarray:
    """One Newton step towards ``B(s) = point`` for each (curve, point) pair.

    `points` is ``(curves, num_points, dim)`` and `s_vals`` is
    ``(curves, num_points)``; the result has the shape of `s_vals`.
    """
    curves = as_batch(nodes)
    num_curves, dim, width = curves.shape
    degree = width - 1
    if degree < 1:
        raise ValueError("a curve of degree 0 has no tangent to refine along")
    pts = np.ascontiguousarray(points, dtype=np.float64)
    starts = np.ascontiguousarray(s_vals, dtype=np.float64)
    if pts.ndim == 2:
        pts = pts[:, np.newaxis, :]
    if pts.ndim != 3 or pts.shape[0] != num_curves or pts.shape[2] != dim:
        raise ValueError("points must be (curves, num_points, dimension)")
    if starts.shape != pts.shape[:2]:
        raise ValueError("s_vals must be (curves, num_points)")
    num_points = pts.shape[1]
    work = np.empty(dim * degree + dim * width + dim, dtype=np.float64)
    dst = np.empty((num_curves, num_points), dtype=np.float64)
    lib.bc_newton_refine(
        _addr(curves), _addr(pts), _addr(starts), _addr(work), num_curves,
        num_points, dim, degree, _addr(dst),
    )
    return dst
