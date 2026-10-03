"""The Bezier curve helpers of the `bezier` package, backed by Mojo.

This module mirrors ``bezier.hazmat.curve_helpers`` from
https://github.com/dhermes/bezier (release 2024.6.20): the same function names,
the same argument order, the same defaults, the same return shapes, and the
same Fortran-ordered ``(dimension, num_nodes)`` arrays. Every function is the
compiled kernel in :mod:`mojo_bezier_curves._lib`; the bodies here are argument
reshaping only, so a caller can swap the import and get the same numbers.

Three departures, all of them forced by the FFI and none of them mathematical:

  * ``compute_length`` takes a ``panels`` argument. Upstream integrates with
    QUADPACK through SciPy; the kernel integrates with composite Simpson, so
    the panel count is what the caller chooses instead of a hidden tolerance.
  * ``locate_point`` subdivides all surviving candidates in one kernel call
    rather than one call per candidate. The candidate set, the bounding-box
    test, the iteration count and the Newton step are unchanged.
  * The kernels are batched over a leading curve index. Everything here is
    called with a single curve, so that index is 1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _lib

__all__ = [
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

_MAX_LOCATE_SUBDIVISIONS = 20
_LOCATE_STD_CAP = 0.5**20
#: Above this many nodes the VS recurrence cannot represent the binomial
#: coefficients exactly, so de Casteljau takes over. Upstream's threshold.
_VS_MAX_NODES = 55


def _nodes(nodes) -> np.ndarray:
    """A single curve's nodes as a ``(dimension, num_nodes)`` float64 array."""
    array = np.asfortranarray(np.asarray(nodes, dtype=np.float64))
    if array.ndim != 2:
        raise ValueError(f"nodes must be 2D, got rank {array.ndim}")
    if array.shape[1] < 1:
        raise ValueError("a curve needs at least one node")
    return array


def _column(array: np.ndarray) -> np.ndarray:
    """The batched ``(1, dimension, num_vals)`` result as Fortran order."""
    return np.asfortranarray(array[0])


def make_subdivision_matrices(degree: int):
    """The matrices used to convert nodes into left and right sub-curve nodes.

    Args:
        degree (int): The degree of the curve.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The left and right matrices, both
        ``(degree + 1, degree + 1)`` and Fortran ordered.
    """
    left, right = _lib.make_subdivision_matrices(degree)
    return np.asfortranarray(left), np.asfortranarray(right)


def subdivide_nodes(nodes):
    """Subdivide a curve into two sub-curves.

    Args:
        nodes (numpy.ndarray): The nodes defining a Bezier curve.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The nodes of the two sub-curves.
    """
    left, right = _lib.subdivide_nodes(_nodes(nodes))
    return _column(left), _column(right)


def evaluate_multi(nodes, s_vals):
    """Compute multiple points along a curve.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        s_vals (numpy.ndarray): Parameters along the curve, as a 1D array.

    Returns:
        numpy.ndarray: The evaluated points, rows for the dimension and columns
        for each ``s`` value.
    """
    grid = _lib._grid(s_vals)
    array = _nodes(nodes)
    if array.shape[1] > _VS_MAX_NODES:
        return _column(
            _lib.evaluate_multi_de_casteljau(
                _lib.as_batch(array), 1.0 - grid, grid
            )
        )
    return _column(_lib.evaluate_multi_s(_lib.Batch(array), grid))


def evaluate_multi_vs(nodes, lambda1, lambda2):
    """Evaluate a Bezier type-function by the VS (modified Horner) algorithm.

    Of the form :math:`B = \\sum_j \\binom{n}{j} \\lambda_1^{n - j}
    \\lambda_2^j v_j`, for each pair of values in ``lambda1`` and ``lambda2``.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        lambda1 (numpy.ndarray): First barycentric weights, as a 1D array.
        lambda2 (numpy.ndarray): Second barycentric weights, as a 1D array.
            Typically ``lambda1 + lambda2 == 1``.

    Returns:
        numpy.ndarray: The evaluated points.
    """
    return _column(
        _lib.evaluate_multi_vs(_lib.Batch(_nodes(nodes)), lambda1, lambda2)
    )


def evaluate_multi_de_casteljau(nodes, lambda1, lambda2):
    """Evaluate a Bezier type-function through the de Casteljau triangle.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        lambda1 (numpy.ndarray): First barycentric weights, as a 1D array.
        lambda2 (numpy.ndarray): Second barycentric weights, as a 1D array.

    Returns:
        numpy.ndarray: The evaluated points.

    Raises:
        ValueError: If ``nodes`` has fewer than two nodes. Upstream raises
            ``IndexError`` on that shape; a degree 0 curve has no triangle to
            contract.
    """
    if _nodes(nodes).shape[1] < 2:
        raise ValueError(
            "evaluate_multi_de_casteljau assumes a curve of degree 1 or higher"
        )
    return _column(
        _lib.evaluate_multi_de_casteljau(_nodes(nodes), lambda1, lambda2)
    )


def evaluate_multi_barycentric(nodes, lambda1, lambda2):
    """Evaluate a Bezier type-function, choosing the algorithm by degree.

    Uses :func:`.evaluate_multi_vs` up to 55 nodes, where the binomial
    coefficients are still exact, and :func:`.evaluate_multi_de_casteljau`
    above that.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        lambda1 (numpy.ndarray): First barycentric weights, as a 1D array.
        lambda2 (numpy.ndarray): Second barycentric weights, as a 1D array.

    Returns:
        numpy.ndarray: The evaluated points.
    """
    array = _nodes(nodes)
    if array.shape[1] > _VS_MAX_NODES:
        return evaluate_multi_de_casteljau(nodes, lambda1, lambda2)
    return evaluate_multi_vs(nodes, lambda1, lambda2)


def vec_size(nodes, s_val) -> float:
    r"""Compute :math:`\|B(s)\|_2`.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        s_val (float): The parameter at which to evaluate.

    Returns:
        float: The norm of the evaluated point.
    """
    grid = np.asfortranarray([s_val], dtype=np.float64)
    return float(_lib.vec_size(_nodes(nodes), 1.0 - grid, grid)[0, 0])


def compute_length(nodes, panels: int = 1024) -> float:
    r"""Approximately compute the length of a curve.

    Uses the hodograph: :math:`\int_0^1 \|B'(s)\|_2 \, ds`, which upstream hands
    to QUADPACK and this kernel integrates with composite Simpson over `panels`
    subintervals. A single node gives 0.0 and a line gives its exact length.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        panels (int): Subintervals for the Simpson rule, rounded up to an even
            number. Defaults to 1024.

    Returns:
        float: The length of the curve.
    """
    array = np.asarray(nodes, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"nodes must be 2D, got rank {array.ndim}")
    if array.shape[1] == 0:
        raise ValueError("Curve should have at least one node.")
    return float(_lib.compute_length(array, panels)[0])


def elevate_nodes(nodes):
    """Degree-elevate a Bezier curve.

    Converts nodes :math:`v_0, \\ldots, v_n` into :math:`w_0, \\ldots, w_{n+1}`
    by :math:`w_0 = v_0`,
    :math:`w_j = (j v_{j - 1} + (n + 1 - j) v_j) / (n + 1)` and
    :math:`w_{n + 1} = v_n`.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.

    Returns:
        numpy.ndarray: The nodes of the degree-elevated curve.
    """
    return _column(_lib.elevate_nodes(_nodes(nodes)))


def de_casteljau_one_round(nodes, lambda1, lambda2):
    """Perform one round of de Casteljau's algorithm.

    The weights are assumed to sum to one.

    Args:
        nodes (numpy.ndarray): Control points for a curve.
        lambda1 (float): First barycentric weight on the interval.
        lambda2 (float): Second barycentric weight on the interval.

    Returns:
        numpy.ndarray: The nodes for a "blended" curve one degree lower.
    """
    return _column(_lib.de_casteljau_one_round(_nodes(nodes), lambda1, lambda2))


def specialize_curve(nodes, start, end):
    """Specialize a curve to a re-parameterization.

    The re-parameterization maps ``start`` to 0 and ``end`` to 1; neither has to
    lie inside ``[0, 1]``.

    Args:
        nodes (numpy.ndarray): Control points for a curve, degree 1 or higher.
        start (float): The start point of the interval we specialize to.
        end (float): The end point of the interval we specialize to.

    Returns:
        numpy.ndarray: The control points for the specialized curve.
    """
    if _nodes(nodes).shape[1] < 2:
        raise ValueError("specialize_curve assumes a curve of degree 1 or higher")
    return _column(_lib.specialize_curve(_nodes(nodes), start, end))


def evaluate_hodograph(s, nodes):
    r"""Evaluate the hodograph curve at a point :math:`s`.

    :math:`B'(s) = n \sum_j \binom{d}{j} s^j (1 - s)^{d - j} \Delta v_j` for
    forward differences :math:`\Delta v_j = v_{j + 1} - v_j`.

    Args:
        s (float): A parameter along the curve.
        nodes (numpy.ndarray): The nodes of a curve.

    Returns:
        numpy.ndarray: The tangent vector, as a ``(dimension, 1)`` array.
    """
    grid = np.asfortranarray([s], dtype=np.float64)
    return _column(_lib.evaluate_hodograph(_nodes(nodes), grid))


def get_curvature(nodes, tangent_vec, s) -> float:
    r"""Compute the signed curvature of a curve at :math:`s`.

    Computed via :math:`(B'(s) \times B''(s)) / \|B'(s)\|_2^3`. Lines have no
    curvature and give 0.0.

    Args:
        nodes (numpy.ndarray): The nodes defining a curve.
        tangent_vec (numpy.ndarray): The already computed value of :math:`B'(s)`.
        s (float): The parameter value along the curve.

    Returns:
        float: The signed curvature.
    """
    grid = np.asfortranarray([s], dtype=np.float64)
    tangent = np.asfortranarray(tangent_vec, dtype=np.float64)
    batch = _lib.get_curvature(
        _nodes(nodes), tangent.reshape(1, *tangent.shape), grid
    )
    return float(batch[0, 0])


def newton_refine(nodes, point, s) -> float:
    r"""Refine a solution to :math:`B(s) = p` using Newton's method.

    Computes :math:`\Delta s = (p - B(s)) \cdot B'(s) / B'(s) \cdot B'(s)` and
    returns :math:`s + \Delta s`, so this is one step, not a solve.

    Args:
        nodes (numpy.ndarray): The nodes defining a Bezier curve.
        point (numpy.ndarray): A point on the curve.
        s (float): An "almost" solution to :math:`B(s) = p`.

    Returns:
        float: The updated value :math:`s + \Delta s`.
    """
    array = _nodes(nodes)
    if array.shape[1] < 2:
        raise ValueError("newton_refine assumes a curve of degree 1 or higher")
    target = np.asfortranarray(point, dtype=np.float64).reshape(-1, 1)
    batch = _lib.newton_refine(
        array, target.reshape(1, 1, -1), np.array([[float(s)]], dtype=np.float64)
    )
    return float(batch[0, 0])


def _contains_nd(nodes, point) -> bool:
    """Whether `point` is inside the axis-aligned bounding box of `nodes`."""
    min_vals = np.min(nodes, axis=1)
    if not np.all(min_vals <= point):
        return False
    max_vals = np.max(nodes, axis=1)
    return bool(np.all(point <= max_vals))


def locate_point(nodes, point) -> Optional[float]:
    r"""Locate a point on a curve.

    Recursively subdivides the curve, rejecting sub-curves whose bounding boxes
    do not contain the point, then takes one Newton step at the mean of the
    surviving parameters. Returns :data:`None` if the point is not on the curve.

    Args:
        nodes (numpy.ndarray): The nodes defining a Bezier curve.
        point (numpy.ndarray): The point to locate.

    Returns:
        Optional[float]: The parameter value corresponding to ``point``.

    Raises:
        ValueError: If the remaining start / end parameters among the subdivided
            intervals are further apart than :math:`2^{-20}`.
    """
    array = _nodes(nodes)
    target = np.asarray(point, dtype=np.float64).ravel(order="F")
    candidates = [(0.0, 1.0, array)]
    for _ in range(_MAX_LOCATE_SUBDIVISIONS + 1):
        survivors = [
            (start, end, candidate)
            for start, end, candidate in candidates
            if _contains_nd(candidate, target)
        ]
        if not survivors:
            return None
        # Every survivor is the same degree, so they subdivide as one batch.
        batch = _lib.as_batch(
            np.stack([candidate for _, _, candidate in survivors])
        )
        left, right = _lib.subdivide_nodes(batch)
        candidates = []
        for index, (start, end, _) in enumerate(survivors):
            midpoint = 0.5 * (start + end)
            candidates.append((start, midpoint, left[index]))
            candidates.append((midpoint, end, right[index]))
    params = [(start, end) for start, end, _ in candidates]
    if np.std(params) > _LOCATE_STD_CAP:
        raise ValueError("Parameters not close enough to one another", params)

    s_approx = np.mean(params)
    s_approx = newton_refine(array, target.reshape(-1, 1), s_approx)
    # NOTE: Since ``np.mean(params)`` must be in ``[0, 1]`` it is "safe" to push
    #       the Newton-refined value back into the unit interval.
    if s_approx < 0.0:
        return 0.0
    if s_approx > 1.0:
        return 1.0
    return s_approx
