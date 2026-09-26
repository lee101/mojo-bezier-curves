"""mojo-bezier-curves: Bezier curve evaluation, splitting and derivatives in Mojo.

A port of the compute core of the `bezier` package
(https://pypi.org/project/bezier/): Bernstein evaluation by two independent
algorithms, de Casteljau subdivision and interval restriction, degree
elevation, the hodograph, signed curvature, Newton's method for point location,
and arc length by composite Simpson.

The Python layer keeps the familiar ``(dimension, degree + 1)`` Fortran-order
node layout that ``bezier.Curve`` uses, and adds a batched ``Curves`` family
whose members share one parameter grid, which is where the compiled kernels pay
off. Every array is owned here and handed to the kernel as a 64-bit address.

The upstream package requires NumPy 2 and is not present in the parity test
environment, so the tests here check the port against an independent NumPy
transcription of the Bernstein definition and against analytic identities
(endpoints, subdivision closure, degree-elevation invariance, circle curvature,
straight-line length). See the README.
"""

from __future__ import annotations

import math

import numpy as np

from . import _lib

__all__ = ["Curve", "Curves", "bernstein"]


_MAX_LOCATE_SUBDIVISIONS = 20


def _for_nodes(nodes: np.ndarray) -> np.ndarray:
    """A ``(dim, degree + 1)`` float64 array, as the `bezier` package stores."""
    array = np.asfortranarray(np.asarray(nodes, dtype=np.float64))
    if array.ndim != 2:
        raise ValueError(
            f"nodes must be (dimension, degree + 1), got rank {np.asarray(nodes).ndim}"
        )
    if array.shape[1] < 1:
        raise ValueError("a curve needs at least one node")
    if not np.isfinite(array).all():
        raise ValueError("nodes must be finite")
    return array


class Curve:
    """A single Bezier curve, shaped like ``bezier.Curve`` for the covered subset."""

    __slots__ = ("_nodes",)

    def __init__(self, nodes, degree: int | None = None, copy: bool = True):
        array = np.asarray(nodes)
        if degree is not None and array.shape[-1] - 1 != degree:
            raise ValueError(
                f"expected degree {degree} for {array.shape[-1]} nodes"
            )
        self._nodes = _for_nodes(np.array(array, order="F", copy=copy) if copy
                                 else np.asfortranarray(array))

    @classmethod
    def from_nodes(cls, nodes, copy: bool = True) -> "Curve":
        """Build a curve from its nodes, taking the degree from their count."""
        return cls(nodes, copy=copy)

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes.copy(order="F")

    @property
    def degree(self) -> int:
        return self._nodes.shape[1] - 1

    @property
    def dimension(self) -> int:
        return self._nodes.shape[0]

    def __repr__(self) -> str:
        return f"Curve(dim={self.dimension}, degree={self.degree})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Curve):
            return NotImplemented
        return self._nodes.shape == other._nodes.shape and bool(
            np.array_equal(self._nodes, other._nodes)
        )

    # -- evaluation ------------------------------------------------------

    def evaluate(self, s: float) -> np.ndarray:
        """``B(s)`` as a ``(dimension, 1)`` array, as `bezier.Curve.evaluate`."""
        return self.evaluate_multi(np.asfortranarray([s]))

    def evaluate_multi(self, s_vals, de_casteljau: bool = False) -> np.ndarray:
        """``B(s)`` for many ``s``, as a ``(dimension, num_s)`` array."""
        grid = np.asfortranarray(np.atleast_1d(np.asarray(s_vals, dtype=np.float64)))
        return _lib.evaluate_multi(self._nodes, grid, de_casteljau)[0].copy(order="F")

    def hodograph(self, s_vals) -> np.ndarray:
        """Tangent vectors ``B'(s)`` as a ``(dimension, num_s)`` array."""
        if self.degree < 1:
            raise ValueError("a curve of degree 0 has no tangent")
        grid = np.asfortranarray(np.atleast_1d(np.asarray(s_vals, dtype=np.float64)))
        return _lib.hodograph(self._nodes, grid)[0].copy(order="F")

    # -- shape -----------------------------------------------------------

    def subdivide(self) -> tuple["Curve", "Curve"]:
        """Split into the ``s in [0, 1/2]`` and ``s in [1/2, 1]`` halves."""
        left, right = _lib.subdivide(self._nodes)
        return Curve(left[0], copy=False), Curve(right[0], copy=False)

    def restrict(self, t: float, from_end: bool = False) -> "Curve":
        """The curve restricted to ``[0, t]`` (or ``[1 - t, 1]``)."""
        return Curve(_lib.restrict(self._nodes, t, from_end)[0], copy=False)

    def specialize(self, start: float, end: float) -> "Curve":
        """Re-parameterise ``[start, end]`` onto ``[0, 1]``."""
        return Curve(_lib.specialize(self._nodes, start, end)[0], copy=False)

    def elevate(self) -> "Curve":
        """A degree-elevated curve representing the same geometry."""
        return Curve(_lib.elevate(self._nodes)[0], copy=False)

    # -- measurements ----------------------------------------------------

    def length(self, panels: int = 1024) -> float:
        """Arc length; exact for lines, a composite Simpson sum otherwise."""
        return float(_lib.length(self._nodes, panels)[0])

    def curvature(self, s_vals) -> np.ndarray:
        """Signed curvature of a planar curve, shape ``(num_s,)``."""
        if self.dimension != 2:
            raise ValueError("curvature is defined for planar curves only")
        grid = np.asfortranarray(np.atleast_1d(np.asarray(s_vals, dtype=np.float64)))
        return _lib.curvature(self._nodes, grid)[0]

    def locate(self, point, max_subdivisions: int = _MAX_LOCATE_SUBDIVISIONS,
               newton_iterations: int = 8):
        """Find ``s`` with ``B(s) == point``, or ``None`` if off the curve.

        Bisection over the control polygon's bounding box followed by Newton,
        which is the strategy of `bezier.Curve.locate`: repeatedly halve and
        keep the halves whose bounding box still contains the point, then
        polish with the compiled Newton kernel.
        """
        target = np.asarray(point, dtype=np.float64).ravel()
        if target.size != self.dimension:
            raise ValueError("point has the wrong dimension")
        candidates = [(0.0, 1.0, self._nodes)]
        for _ in range(max_subdivisions + 1):
            nxt = []
            for start, end, candidate in candidates:
                if not _contains(candidate, target):
                    continue
                midpoint = 0.5 * (start + end)
                left, right = _lib.subdivide(candidate)
                nxt.append((start, midpoint, left[0]))
                nxt.append((midpoint, end, right[0]))
            candidates = nxt
        if not candidates:
            return None
        starts = np.array([0.5 * (lo + hi) for lo, hi, _ in candidates])
        if starts.std() > 0.5**20:
            raise ValueError("parameters not close enough to one another")
        repeats = np.repeat(target[np.newaxis, np.newaxis, :], starts.size, axis=1)
        refined = _lib.newton_refine(
            self._nodes, repeats, starts[np.newaxis, :], newton_iterations
        )[0]
        # The mean of the candidate parameters must be in [0, 1], so the
        # refined value can be pushed back into the unit interval safely.
        return float(min(max(refined[0], 0.0), 1.0))


def _contains(nodes: np.ndarray, point: np.ndarray) -> bool:
    """Whether `point` lies inside the axis-aligned box of `nodes`."""
    return bool(np.all(nodes.min(axis=1) <= point) and np.all(point <= nodes.max(axis=1)))


class Curves:
    """A family of curves sharing a dimension, a degree and a parameter grid.

    The compiled kernels are batched over this family, which is the reason they
    exist: upstream evaluates one curve per Python call.
    """

    __slots__ = ("_nodes",)

    def __init__(self, nodes, copy: bool = True):
        array = np.asarray(nodes, dtype=np.float64)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        if array.ndim != 3:
            raise ValueError("nodes must be (curves, dimension, degree + 1)")
        self._nodes = np.ascontiguousarray(array) if copy else array
        _lib.as_batch(self._nodes)

    @classmethod
    def from_nodes(cls, nodes) -> "Curves":
        return cls(nodes)

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes.copy()



    @property
    def dimension(self) -> int:
        return self._nodes.shape[1]

    @property
    def degree(self) -> int:
        return self._nodes.shape[2] - 1

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Curve:
        return Curve(self._nodes[index], copy=False)

    def __repr__(self) -> str:
        return (
            f"Curves(count={self.count}, dim={self.dimension}, degree={self.degree})"
        )

    def evaluate_multi(self, s_vals, de_casteljau: bool = False) -> np.ndarray:
        return _lib.evaluate_multi(self._nodes, s_vals, de_casteljau)

    def hodograph(self, s_vals) -> np.ndarray:
        return _lib.hodograph(self._nodes, s_vals)

    def subdivide(self) -> tuple["Curves", "Curves"]:
        left, right = _lib.subdivide(self._nodes)
        return Curves(left, copy=False), Curves(right, copy=False)

    def restrict(self, t: float, from_end: bool = False) -> "Curves":
        return Curves(_lib.restrict(self._nodes, t, from_end), copy=False)

    def specialize(self, start: float, end: float) -> "Curves":
        return Curves(_lib.specialize(self._nodes, start, end), copy=False)

    def elevate(self) -> "Curves":
        return Curves(_lib.elevate(self._nodes), copy=False)

    def curvature(self, s_vals) -> np.ndarray:
        return _lib.curvature(self._nodes, s_vals)

    def length(self, panels: int = 1024) -> np.ndarray:
        return _lib.length(self._nodes, panels)


def bernstein(nodes, s) -> np.ndarray:
    """The Bernstein definition itself, written out longhand.

    ``B(s) = sum_j C(n, j) (1 - s)^(n - j) s^j v_j``. This is the reference the
    compiled recurrences are checked against: it shares no code with them, so a
    mistake in the VS recurrence or in the de Casteljau triangle cannot hide
    behind a matching mistake in the reference.
    """
    array = np.asarray(nodes, dtype=np.float64)
    if array.ndim == 3:
        return np.stack([bernstein(curve, s) for curve in array])
    degree = array.shape[1] - 1
    grid = np.atleast_1d(np.asarray(s, dtype=np.float64))
    result = np.zeros((array.shape[0], grid.size), dtype=np.float64)
    for j in range(degree + 1):
        coefficient = np.array(
            [math.comb(degree, j) * (1.0 - t) ** (degree - j) * t**j for t in grid]
        )
        result += array[:, j][:, np.newaxis] * coefficient[np.newaxis, :]
    return result
