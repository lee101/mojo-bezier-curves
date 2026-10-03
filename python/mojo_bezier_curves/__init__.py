"""mojo-bezier-curves: Bezier curve evaluation in Mojo.

A port of the compute core of the `bezier` package
(https://github.com/dhermes/bezier, release 2024.6.20). :class:`Curve` has the
names, argument order, defaults and return shapes of ``bezier.Curve`` for the
covered subset, so it is a drop-in for those methods, and
:mod:`mojo_bezier_curves.curve_helpers` has the names and signatures of
``bezier.hazmat.curve_helpers``.

:class:`Curves` is the addition: a family of curves that share a dimension, a
degree and a parameter grid, which the compiled kernels are batched over.
Upstream evaluates one curve per Python call, so a quadrature or a Newton
refinement over a thousand curves costs a thousand calls here and one.
"""

from __future__ import annotations

import numpy as np

from . import _lib, curve_helpers
from .curve_helpers import (  # noqa: F401  (re-exported for the drop-in shape)
    compute_length,
    de_casteljau_one_round,
    elevate_nodes,
    evaluate_hodograph,
    evaluate_multi,
    evaluate_multi_barycentric,
    evaluate_multi_de_casteljau,
    evaluate_multi_vs,
    get_curvature,
    locate_point,
    make_subdivision_matrices,
    newton_refine,
    specialize_curve,
    subdivide_nodes,
    vec_size,
)

__version__ = "0.1.0"

__all__ = [
    "Curve",
    "Curves",
    "curve_helpers",
    "make_subdivision_matrices",
    "subdivide_nodes",
    "evaluate_multi",
    "evaluate_multi_vs",
    "evaluate_multi_de_casteljau",
    "evaluate_multi_barycentric",
    "vec_size",
    "compute_length",
    "elevate_nodes",
    "de_casteljau_one_round",
    "specialize_curve",
    "evaluate_hodograph",
    "get_curvature",
    "newton_refine",
    "locate_point",
]

#: Panel count used by :attr:`Curve.length`, which upstream takes from QUADPACK.
DEFAULT_LENGTH_PANELS = 1024


def _sequence_to_array(nodes) -> np.ndarray:
    """A ``(dimension, num_nodes)`` float64 array, as ``bezier`` stores nodes."""
    array = np.asarray(nodes, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"nodes must be 2D, got rank {array.ndim}")
    if array.shape[0] < 1:
        raise ValueError("a curve must live in at least one dimension")
    if not np.isfinite(array).all():
        raise ValueError("nodes must be finite")
    return array


class Curve:
    """A single Bezier curve.

    Args:
        nodes (Sequence[Sequence[numbers.Number]]): The nodes of the curve, as
            a ``(dimension, degree + 1)`` array.
        degree (int): The degree of the curve.
        copy (bool): Whether to copy ``nodes`` before storing it.
        verify (bool): Whether to check that the node count matches `degree`.

    Raises:
        ValueError: If ``nodes`` is not 2D, if the node count does not match
            `degree`, or if the nodes are not finite.
    """

    __slots__ = ("_batch", "_degree", "_dimension", "_nodes")

    def __init__(self, nodes, degree: int, *, copy: bool = True, verify: bool = True):
        array = _sequence_to_array(nodes)
        self._dimension = array.shape[0]
        self._nodes = array.copy(order="F") if copy else array
        self._degree = int(degree)
        if verify:
            self._verify_degree()
        # The kernels read a C-contiguous batch by address. Resolving it here,
        # once, is what keeps a call down to the kernel: `as_batch` and
        # `int(array.ctypes.data)` are per-call work on an array that never
        # changes, and on this path they cost more than the evaluation.
        self._batch = _lib.Batch(self._nodes[np.newaxis])

    @classmethod
    def from_nodes(cls, nodes, copy: bool = True) -> "Curve":
        """Create a :class:`.Curve` from nodes, taking the degree from them."""
        array = _sequence_to_array(nodes)
        _, num_nodes = array.shape
        return cls(array, num_nodes - 1, copy=copy, verify=False)

    def _verify_degree(self) -> None:
        num_nodes = self._nodes.shape[1]
        expected_nodes = self._degree + 1
        if num_nodes != expected_nodes:
            raise ValueError(
                f"A degree {self._degree} curve should have "
                f"{expected_nodes} nodes, not {num_nodes}."
            )

    @property
    def nodes(self) -> np.ndarray:
        """numpy.ndarray: The nodes of the curve."""
        return self._nodes.copy(order="F")

    @property
    def degree(self) -> int:
        """int: The degree of the curve."""
        return self._degree

    @property
    def dimension(self) -> int:
        """int: The dimension the curve lives in."""
        return self._dimension

    @property
    def length(self) -> float:
        """float: The length of the curve."""
        return curve_helpers.compute_length(
            self._nodes, panels=DEFAULT_LENGTH_PANELS
        )

    def copy(self) -> "Curve":
        """Copy of the current curve."""
        return Curve(self._nodes, self._degree, copy=True, verify=False)

    def __repr__(self) -> str:
        return f"<Curve (degree={self._degree}, dimension={self._dimension})>"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Curve):
            return NotImplemented
        return (
            self._degree == other._degree
            and self._dimension == other._dimension
            and bool(np.array_equal(self._nodes, other._nodes))
        )

    def _evaluate_grid(self, grid):
        """`B(s)` for a continuous parameter grid, on the nodes held here.

        The batch and its address come from construction, and `1 - grid` is
        formed in the kernel, so an evaluation is the grid, the output buffer
        and the call.
        """
        if self._degree + 1 > curve_helpers._VS_MAX_NODES:
            return curve_helpers._column(
                _lib.evaluate_multi_de_casteljau(self._nodes, 1.0 - grid, grid)
            )
        return curve_helpers._column(_lib.evaluate_multi_s(self._batch, grid))

    def evaluate(self, s):
        r"""Evaluate :math:`B(s)` along the curve.

        Args:
            s (float): Parameter along the curve.

        Returns:
            numpy.ndarray: The point on the curve, as a ``(dimension, 1)``
            array.
        """
        return self._evaluate_grid(np.ascontiguousarray([s], dtype=np.float64))

    def evaluate_multi(self, s_vals):
        r"""Evaluate :math:`B(s)` for multiple points along the curve.

        Args:
            s_vals (numpy.ndarray): Parameters along the curve, as a 1D array.

        Returns:
            numpy.ndarray: The points on the curve, columns for each ``s``
            value and rows for the dimension.
        """
        return self._evaluate_grid(_lib._grid(s_vals))

    def evaluate_hodograph(self, s):
        r"""Evaluate the tangent vector :math:`B'(s)` along the curve.

        Args:
            s (float): Parameter along the curve.

        Returns:
            numpy.ndarray: The tangent vector, as a ``(dimension, 1)`` array.
        """
        return curve_helpers.evaluate_hodograph(s, self._nodes)

    def subdivide(self):
        r"""Split the curve into a left and a right half.

        Returns:
            Tuple[Curve, Curve]: The sub-curves on :math:`[0, 1/2]` and
            :math:`[1/2, 1]`.
        """
        left_nodes, right_nodes = curve_helpers.subdivide_nodes(self._nodes)
        left = Curve(left_nodes, self._degree, copy=False, verify=False)
        right = Curve(right_nodes, self._degree, copy=False, verify=False)
        return left, right

    def elevate(self) -> "Curve":
        """Return a degree-elevated version of the current curve."""
        new_nodes = curve_helpers.elevate_nodes(self._nodes)
        return Curve(new_nodes, self._degree + 1, copy=False, verify=False)

    def specialize(self, start, end) -> "Curve":
        """Specialize the curve to a sub-interval, re-parameterized to ``[0, 1]``.

        Args:
            start (float): The start point of the interval we specialize to.
            end (float): The end point of the interval we specialize to.

        Returns:
            Curve: The newly specialized curve.
        """
        new_nodes = curve_helpers.specialize_curve(self._nodes, start, end)
        return Curve(new_nodes, self._degree, copy=False, verify=False)

    def locate(self, point):
        r"""Find a point on the current curve, i.e. solve for :math:`B(s) = p`.

        Args:
            point (numpy.ndarray): The point to locate.

        Returns:
            Optional[float]: The parameter value, or :data:`None` if the point
            is not on the curve.
        """
        return curve_helpers.locate_point(self._nodes, point)


class Curves:
    """A family of curves sharing a dimension, a degree and a parameter grid.

    The compiled kernels are batched over this family, which is the reason they
    exist. Nodes are ``(curves, dimension, degree + 1)``; every method returns
    the same result for each curve, in the same layout, from a single call.
    """

    __slots__ = ("_batch", "_nodes")

    def __init__(self, nodes, copy: bool = True):
        array = np.asarray(nodes, dtype=np.float64)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        if array.ndim != 3:
            raise ValueError(
                "nodes must be (curves, dimension, degree + 1), got rank "
                f"{np.asarray(nodes).ndim}"
            )
        # Without the copy, an already contiguous array is adopted as it is;
        # anything else still has to be laid out the way the kernel reads it.
        # Either way the nodes came from a kernel or from this constructor,
        # so the finiteness pass `as_batch` would make is redundant here.
        self._batch = _lib.Batch(array) if copy else _lib.Batch.trusted(array)
        self._nodes = self._batch.array

    @classmethod
    def from_nodes(cls, nodes) -> "Curves":
        """Create a :class:`.Curves` family from a stacked node array."""
        return cls(nodes)

    @property
    def nodes(self) -> np.ndarray:
        """numpy.ndarray: The nodes of every curve in the family."""
        return self._nodes.copy()

    @property
    def count(self) -> int:
        """int: The number of curves in the family."""
        return self._nodes.shape[0]

    @property
    def dimension(self) -> int:
        """int: The dimension the curves live in."""
        return self._nodes.shape[1]

    @property
    def degree(self) -> int:
        """int: The degree shared by the curves."""
        return self._nodes.shape[2] - 1

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Curve:
        return Curve(self._nodes[index], self.degree, copy=False, verify=False)

    def __iter__(self):
        for index in range(self.count):
            yield self[index]

    def __repr__(self) -> str:
        return (
            f"Curves(count={self.count}, dimension={self.dimension}, "
            f"degree={self.degree})"
        )

    def _evaluate_barycentric(self, lambda1, lambda2):
        """`evaluate_multi_barycentric` over the whole family, in one call."""
        if self._nodes.shape[2] > curve_helpers._VS_MAX_NODES:
            return _lib.evaluate_multi_de_casteljau(self._nodes, lambda1, lambda2)
        return _lib.evaluate_multi_vs(self._batch, lambda1, lambda2)

    def _evaluate_grid(self, grid):
        """`B(s)` over the whole family, in one call."""
        if self._nodes.shape[2] > curve_helpers._VS_MAX_NODES:
            return _lib.evaluate_multi_de_casteljau(self._nodes, 1.0 - grid, grid)
        return _lib.evaluate_multi_s(self._batch, grid)

    def evaluate(self, s):
        """Evaluate every curve at one parameter, result ``(curves, dimension)``."""
        return self._evaluate_grid(np.ascontiguousarray([s], dtype=np.float64))[
            :, :, 0
        ]

    def evaluate_multi(self, s_vals):
        """Evaluate every curve on a grid, result ``(curves, dimension, num_s)``."""
        return self._evaluate_grid(
            np.ascontiguousarray(s_vals, dtype=np.float64).ravel()
        )

    def evaluate_hodograph(self, s_vals):
        """Tangent vectors for every curve, result ``(curves, dimension, num_s)``."""
        return _lib.evaluate_hodograph(self._nodes, s_vals)

    def subdivide(self):
        """Split every curve in half, as two new families."""
        left, right = _lib.subdivide_nodes(self._nodes)
        return Curves(left, copy=False), Curves(right, copy=False)

    def elevate(self) -> "Curves":
        """Degree-elevate every curve."""
        return Curves(_lib.elevate_nodes(self._nodes), copy=False)

    def specialize(self, start, end) -> "Curves":
        """Re-parameterize every curve from ``[start, end]`` onto ``[0, 1]``."""
        return Curves(_lib.specialize_curve(self._nodes, start, end), copy=False)

    def get_curvature(self, s_vals):
        """Signed curvature of every planar curve, result ``(curves, num_s)``."""
        grid = np.ascontiguousarray(s_vals, dtype=np.float64).ravel()
        tangents = _lib.evaluate_hodograph(self._nodes, grid)
        return _lib.get_curvature(self._nodes, tangents, grid)

    def length(self, panels: int = DEFAULT_LENGTH_PANELS):
        """Arc length of every curve, result ``(curves,)``."""
        return _lib.compute_length(self._nodes, panels)

    def newton_refine(self, points, s_vals):
        """One Newton step per (curve, point) pair, result shaped like `s_vals`."""
        return _lib.newton_refine(self._nodes, points, s_vals)
