"""Compiled inner loops for Bezier curve geometry.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The batched layout is the point of this port. Upstream evaluates one curve at a
time from Python, which pays a call per curve; here a family of curves sharing
one parameter grid is laid out as

    nodes  (num_curves, dim, degree + 1)   C-contiguous float64
    s_vals (num_s,)                        the shared parameter grid
    result (num_curves, dim, num_s)        evaluate / hodograph output

so one call evaluates the whole family. Single-curve use is the same call with
`num_curves == 1`. Because the arrays are C-contiguous with the coordinate
dimension *before* the node dimension, the element for coordinate `d` and node
`j` of curve `b` lives at `b * dim * width + d * width + j`, and the result for
parameter `m` at `b * dim * num_s + d * num_s + m`. Getting those two strides
the wrong way round is the single easiest mistake to make here, and it is
transposed output rather than garbage, so the tests check coordinates
individually as well as in bulk.

Scratch regions are passed in as one buffer and addressed by explicit index
offsets, because `Pointer.__add__` is deprecated in this dialect.

The recurrences follow the published algorithms of the `bezier` package
(dhermes/bezier, `src/python/bezier/hazmat/curve_helpers.py`): the VS / modified
Horner evaluation, the classical de Casteljau triangle, the degree-elevation
formula, and Newton's method on `B(s) = p`.
"""

from std.math import sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def vs_eval(
    nodes: FPtr, node_offset: Int, width: Int, s: Float64, dst: FPtr,
    dst_offset: Int, dst_stride: Int, dim: Int, degree: Int
):
    """Bezier type function by the VS (modified Horner) algorithm.

    ``result = lambda1 * v_0``,
    ``result = lambda1 * (result + C(degree, j) lambda2^j v_j)`` for
    ``j = 1 .. degree - 1``, then ``result + lambda2^degree v_degree``, with
    ``lambda1 = 1 - s`` and ``lambda2 = s``. Coordinate ``d`` lands at
    ``dst[dst_offset + d * dst_stride]``.
    """
    var lambda1 = 1.0 - s
    var lambda2 = s
    for d in range(dim):
        dst[unsafe_offset=dst_offset + d * dst_stride] = (
            lambda1 * nodes[unsafe_offset=node_offset + d * width]
        )
    var binom = 1.0
    var lambda2_pow = 1.0
    for j in range(1, degree):
        lambda2_pow *= lambda2
        binom = binom * Float64(degree - j + 1) / Float64(j)
        for d in range(dim):
            var acc = dst[unsafe_offset=dst_offset + d * dst_stride]
            acc = acc + binom * lambda2_pow * nodes[unsafe_offset=node_offset + d * width + j]
            dst[unsafe_offset=dst_offset + d * dst_stride] = lambda1 * acc
    for d in range(dim):
        var acc = dst[unsafe_offset=dst_offset + d * dst_stride]
        dst[unsafe_offset=dst_offset + d * dst_stride] = (
            acc + lambda2 * lambda2_pow * nodes[unsafe_offset=node_offset + d * width + degree]
        )


def de_casteljau(
    nodes: FPtr, node_offset: Int, width: Int, lambda1: Float64,
    lambda2: Float64, work: FPtr, work_offset: Int, dst: FPtr, dst_offset: Int,
    dst_stride: Int, dim: Int, degree: Int
):
    """Bezier type function through one de Casteljau triangle.

    The triangle lives in `work[work_offset : work_offset + dim * (degree + 1)]`
    and is clobbered.
    """
    for i in range(degree):
        for d in range(dim):
            work[unsafe_offset=work_offset + d * width + i] = (
                lambda1 * nodes[unsafe_offset=node_offset + d * width + i]
                + lambda2 * nodes[unsafe_offset=node_offset + d * width + i + 1]
            )
    # Every round shortens the valid part of each coordinate's row by one, but
    # the stride stays `width` so a round's writes never overtake another
    # coordinate's unread entries.
    var row = degree
    while row > 1:
        for d in range(dim):
            for i in range(row - 1):
                work[unsafe_offset=work_offset + d * width + i] = (
                    lambda1 * work[unsafe_offset=work_offset + d * width + i]
                    + lambda2 * work[unsafe_offset=work_offset + d * width + i + 1]
                )
        row -= 1
    for d in range(dim):
        dst[unsafe_offset=dst_offset + d * dst_stride] = work[
            unsafe_offset=work_offset + d * width
        ]


@export("bc_evaluate_multi")
def bc_evaluate_multi(
    nodes_addr: Int, s_addr: Int, work_addr: Int, num_curves: Int, dim: Int,
    degree: Int, num_s: Int, dst_addr: Int
) abi("C"):
    """Evaluate every curve on the shared grid with the VS algorithm."""
    var nodes = fp(nodes_addr)
    var s_vals = fp(s_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        for m in range(num_s):
            vs_eval(
                nodes, base, width, s_vals[unsafe_offset=m], dst,
                b * dim * num_s + m, num_s, dim, degree
            )


@export("bc_evaluate_de_casteljau")
def bc_evaluate_de_casteljau(
    nodes_addr: Int, s_addr: Int, work_addr: Int, num_curves: Int, dim: Int,
    degree: Int, num_s: Int, dst_addr: Int
) abi("C"):
    """Evaluate every curve on the shared grid with de Casteljau instead.

    The same answer as `bc_evaluate_multi` up to rounding. Two independent
    algorithms are kept so a test can tell them apart, and so a caller can take
    the numerically gentler one on a badly conditioned control polygon.
    `work` needs ``dim * (degree + 1)`` doubles.
    """
    var nodes = fp(nodes_addr)
    var s_vals = fp(s_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        for m in range(num_s):
            de_casteljau(
                nodes, base, width, 1.0 - s_vals[unsafe_offset=m],
                s_vals[unsafe_offset=m], work, 0, dst,
                b * dim * num_s + m, num_s, dim, degree
            )


# ---------------------------------------------------------------------------
# Splitting, re-parameterisation, degree elevation
# ---------------------------------------------------------------------------


@export("bc_split_at")
def bc_split_at(
    nodes_addr: Int, work_addr: Int, num_curves: Int, dim: Int, degree: Int,
    left1: Float64, left2: Float64, right1: Float64, right2: Float64,
    left_addr: Int, right_addr: Int
) abi("C"):
    """Read the two edges of the de Casteljau triangle for two weight pairs.

    The left edge evaluated at ``(1 - t, t)`` are the nodes of the curve
    restricted to ``[0, t]``; the right edge at the same weights are the nodes
    of the curve restricted to ``[t, 1]``. Restricting to an interior interval
    ``[a, b]`` is therefore the composition restrict-to-``[0, a]`` then
    restrict-to-``[b, 1]``, which is what the Python layer does.

    With ``(1/2, 1/2)`` on both sides this is de Casteljau's subdivision: the
    left edge are the first half's nodes, the right edge the second half's, and
    they meet at the midpoint.

    `work` needs ``dim * (degree + 1)`` doubles.
    """
    var nodes = fp(nodes_addr)
    var work = fp(work_addr)
    var left = fp(left_addr)
    var right = fp(right_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        for d in range(dim):
            for i in range(width):
                work[unsafe_offset=d * width + i] = nodes[unsafe_offset=base + d * width + i]
        for k in range(degree):
            for d in range(dim):
                left[unsafe_offset=base + d * width + k] = work[unsafe_offset=d * width]
            for d in range(dim):
                for i in range(width - k - 1):
                    work[unsafe_offset=d * width + i] = (
                        left1 * work[unsafe_offset=d * width + i]
                        + left2 * work[unsafe_offset=d * width + i + 1]
                    )
        for d in range(dim):
            left[unsafe_offset=base + d * width + degree] = work[unsafe_offset=d * width]

        for d in range(dim):
            for i in range(width):
                work[unsafe_offset=d * width + i] = nodes[unsafe_offset=base + d * width + i]
        for k in range(degree + 1):
            for d in range(dim):
                right[unsafe_offset=base + d * width + (degree - k)] = work[
                    unsafe_offset=d * width + (degree - k)
                ]
            for d in range(dim):
                for i in range(width - k - 1):
                    work[unsafe_offset=d * width + i] = (
                        right1 * work[unsafe_offset=d * width + i]
                        + right2 * work[unsafe_offset=d * width + i + 1]
                    )


@export("bc_elevate")
def bc_elevate(
    nodes_addr: Int, num_curves: Int, dim: Int, degree: Int, dst_addr: Int
) abi("C"):
    """Degree elevation: w_0 = v_0, w_n+1 = v_n, w_j = (j v_{j-1} + (n+1-j) v_j)/(n+1).

    `dst` is ``(num_curves, dim, degree + 2)``.
    """
    var nodes = fp(nodes_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    var new_width = width + 1
    var denominator = Float64(width)
    for b in range(num_curves):
        var base = b * dim * width
        var obase = b * dim * new_width
        for d in range(dim):
            dst[unsafe_offset=obase + d * new_width] = nodes[unsafe_offset=base + d * width]
        for j in range(1, width):
            var jf = Float64(j)
            for d in range(dim):
                dst[unsafe_offset=obase + d * new_width + j] = (
                    jf * nodes[unsafe_offset=base + d * width + j - 1]
                    + (denominator - jf) * nodes[unsafe_offset=base + d * width + j]
                ) / denominator
        for d in range(dim):
            dst[unsafe_offset=obase + d * new_width + width] = nodes[
                unsafe_offset=base + d * width + degree
            ]


# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------


def forward_diff(
    nodes: FPtr, node_offset: Int, width: Int, work: FPtr, work_offset: Int,
    dim: Int, degree: Int
):
    """Write ``degree * (v_{j+1} - v_j)``: the nodes of the hodograph."""
    for d in range(dim):
        for j in range(degree):
            work[unsafe_offset=work_offset + d * degree + j] = Float64(degree) * (
                nodes[unsafe_offset=node_offset + d * width + j + 1]
                - nodes[unsafe_offset=node_offset + d * width + j]
            )


@export("bc_hodograph")
def bc_hodograph(
    nodes_addr: Int, s_addr: Int, work_addr: Int, num_curves: Int, dim: Int,
    degree: Int, num_s: Int, dst_addr: Int
) abi("C"):
    """Evaluate the tangent vector ``B'(s)`` of every curve on the grid."""
    var nodes = fp(nodes_addr)
    var s_vals = fp(s_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        forward_diff(nodes, base, width, work, 0, dim, degree)
        for m in range(num_s):
            vs_eval(
                work, 0, degree, s_vals[unsafe_offset=m], dst,
                b * dim * num_s + m, num_s, dim, degree - 1
            )


@export("bc_curvature")
def bc_curvature(
    nodes_addr: Int, s_addr: Int, work_addr: Int, num_curves: Int, dim: Int,
    degree: Int, num_s: Int, dst_addr: Int
) abi("C"):
    """Signed curvature ``(x' y'' - y' x'') / |B'|^3`` of planar curves.

    `dst` is ``(num_curves, num_s)``. `work` needs
    ``dim * degree + dim * (degree - 1) + 4`` doubles: the hodograph nodes, the
    second-derivative nodes, and four slots for the two evaluated vectors.
    """
    var nodes = fp(nodes_addr)
    var s_vals = fp(s_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    var first = dim * degree
    var second = first + dim * (degree - 1)
    var slots = second
    for b in range(num_curves):
        var base = b * dim * width
        if degree < 2:
            for m in range(num_s):
                dst[unsafe_offset=b * num_s + m] = 0.0
            continue
        forward_diff(nodes, base, width, work, 0, dim, degree)
        forward_diff(work, 0, degree, work, first, dim, degree - 1)
        for m in range(num_s):
            var s = s_vals[unsafe_offset=m]
            vs_eval(work, 0, degree, s, work, slots, 1, 2, degree - 1)
            vs_eval(work, first, degree - 1, s, work, slots + 2, 1, 2, degree - 2)
            var x1 = work[unsafe_offset=slots]
            var y1 = work[unsafe_offset=slots + 1]
            var x2 = work[unsafe_offset=slots + 2]
            var y2 = work[unsafe_offset=slots + 3]
            var speed = sqrt(x1 * x1 + y1 * y1)
            if speed == 0.0:
                dst[unsafe_offset=b * num_s + m] = 0.0
            else:
                dst[unsafe_offset=b * num_s + m] = (
                    x1 * y2 - y1 * x2
                ) / (speed * speed * speed)


# ---------------------------------------------------------------------------
# Point location and length
# ---------------------------------------------------------------------------


@export("bc_newton_refine")
def bc_newton_refine(
    nodes_addr: Int, points_addr: Int, s0_addr: Int, work_addr: Int,
    num_curves: Int, num_points: Int, dim: Int, degree: Int, iterations: Int,
    dst_addr: Int
) abi("C"):
    """Newton iterations on ``B(s) = p`` for many (curve, point) pairs.

    ``s += ((p - B(s)) . B'(s)) / (B'(s) . B'(s))``, applied `iterations` times
    from ``s0``. Residual and derivative are both re-evaluated each step, as
    upstream's `newton_refine` does; this kernel only keeps the loop out of
    Python, so locating many points on one curve costs one call.

    `dst` is ``(num_curves, num_points)``. `work` needs ``dim * (degree + 2)``
    doubles laid out as three regions that must not overlap: the hodograph nodes
    (``dim * degree``), the residual vector ``B(s)`` (``dim``), and the
    evaluated derivative (``dim``).
    """
    var nodes = fp(nodes_addr)
    var points = fp(points_addr)
    var s0 = fp(s0_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        if degree >= 1:
            forward_diff(nodes, base, width, work, 0, dim, degree)
        for k in range(num_points):
            var pbase = (b * num_points + k) * dim
            var s = s0[unsafe_offset=b * num_points + k]
            for _ in range(iterations):
                var residual_slot = dim * degree
                var deriv_slot = residual_slot + dim
                vs_eval(nodes, base, width, s, work, residual_slot, 1, dim, degree)
                var residual = Float64(0.0)
                var norm = Float64(0.0)
                if degree >= 1:
                    vs_eval(work, 0, degree, s, work, deriv_slot, 1, dim, degree - 1)
                    for d in range(dim):
                        var delta = (
                            points[unsafe_offset=pbase + d]
                            - work[unsafe_offset=residual_slot + d]
                        )
                        var deriv = work[unsafe_offset=deriv_slot + d]
                        residual = residual + delta * deriv
                        norm = norm + deriv * deriv
                if norm == 0.0:
                    break
                s = s + residual / norm
            dst[unsafe_offset=b * num_points + k] = s


@export("bc_length")
def bc_length(
    nodes_addr: Int, work_addr: Int, num_curves: Int, dim: Int, degree: Int,
    panels: Int, dst_addr: Int
) abi("C"):
    """Arc length by composite Simpson on the hodograph norm.

    ``length = integral over [0, 1] of ||B'(s)||_2 ds``. `panels` subintervals
    are used, rounded up to an even number; halving the step halves the error,
    so the accuracy is a parameter rather than a claim. Lines (degree 1) come
    out exactly, as upstream returns them.

    `dst` is ``(num_curves,)``. `work` needs ``dim * (degree + 1)`` doubles:
    the hodograph nodes, then the evaluated speed vector.
    """
    var nodes = fp(nodes_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    var second = dim * degree
    for b in range(num_curves):
        var base = b * dim * width
        if degree == 0:
            dst[unsafe_offset=b] = 0.0
            continue
        if degree == 1:
            var acc = Float64(0.0)
            for d in range(dim):
                var delta = nodes[unsafe_offset=base + d * width + 1] - nodes[
                    unsafe_offset=base + d * width
                ]
                acc = acc + delta * delta
            dst[unsafe_offset=b] = sqrt(acc)
            continue
        forward_diff(nodes, base, width, work, 0, dim, degree)
        var intervals = panels
        if intervals < 2:
            intervals = 2
        intervals = (intervals + 1) // 2 * 2
        var h = 1.0 / Float64(intervals)
        var total = Float64(0.0)
        for i in range(intervals + 1):
            var s = Float64(i) * h
            vs_eval(work, 0, degree, s, work, second, 1, dim, degree - 1)
            var acc = Float64(0.0)
            for d in range(dim):
                acc = acc + work[unsafe_offset=second + d] * work[unsafe_offset=second + d]
            var speed = sqrt(acc)
            if i == 0 or i == intervals:
                total += speed
            elif i % 2 == 0:
                total += 2.0 * speed
            else:
                total += 4.0 * speed
        dst[unsafe_offset=b] = total * h / 3.0
