"""The vectorised kernels at the edges of their block.

`W = simd_width_of[float64]()` is 4 on this machine, so a grid of `n` values is
`n // W` vector blocks and `n % W` parameters through the scalar tail. These
tests sweep `n` across that boundary -- 0, 1, 2, 3, 4, 5, up to a full block
more than one -- for every kernel that has a vector body, against upstream's
own answers. A tail that read past the end of the grid, skipped a parameter, or
was only right for the sizes the other tests happen to use would fail here.

The last test sweeps the same grid lengths for the degrees above 55, where
`evaluate_multi` dispatches to de Casteljau instead and there is no vector body
left to get wrong.
"""

import numpy as np
import pytest

from conftest import ATOL, RTOL, curve_family
from mojo_bezier_curves import Curves, _lib
from mojo_bezier_curves import curve_helpers as mine

#: 0 is an empty grid, 4 is exactly one block, and 13 is three blocks plus one.
_SIZES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13]


@pytest.mark.parametrize("num_vals", _SIZES)
@pytest.mark.parametrize("degree", [0, 1, 2, 5, 8])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_evaluate_multi_at_every_tail_size(num_vals, degree, dim, up):
    values = np.linspace(0.0, 1.0, num_vals)
    for nodes in curve_family(2, dim, degree, seed=num_vals + degree + dim):
        np.testing.assert_allclose(
            mine.evaluate_multi(nodes, values),
            up.evaluate_multi(nodes, values),
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize("num_vals", _SIZES)
@pytest.mark.parametrize("degree", [1, 2, 4, 9])
def test_evaluate_multi_vs_at_every_tail_size(num_vals, degree, up):
    values = np.linspace(0.0, 1.0, num_vals)
    lambda1 = 1.0 - values
    for nodes in curve_family(2, 2, degree, seed=num_vals + degree):
        np.testing.assert_allclose(
            mine.evaluate_multi_vs(nodes, lambda1, values),
            up.evaluate_multi_vs(nodes, lambda1, values),
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize("num_vals", _SIZES)
@pytest.mark.parametrize("degree", [1, 2, 4, 9])
def test_evaluate_hodograph_at_every_tail_size(num_vals, degree, up):
    """Upstream's hodograph is scalar, so the grid is its own reference."""
    values = np.linspace(0.0, 1.0, num_vals)
    for nodes in curve_family(2, 2, degree, seed=num_vals + degree):
        result = _lib.evaluate_hodograph(nodes, values)[0]
        for column, s in enumerate(values):
            np.testing.assert_allclose(
                result[:, column],
                up.evaluate_hodograph(s, nodes).ravel(),
                rtol=RTOL,
                atol=ATOL,
            )


@pytest.mark.parametrize("num_vals", _SIZES)
@pytest.mark.parametrize("degree", [1, 2, 4, 9])
def test_get_curvature_at_every_tail_size(num_vals, degree, up):
    values = np.linspace(0.0, 1.0, num_vals)
    for nodes in curve_family(2, 2, degree, seed=num_vals + degree):
        tangents = _lib.evaluate_hodograph(nodes, values)
        result = _lib.get_curvature(nodes, tangents, values)[0]
        for column, s in enumerate(values):
            np.testing.assert_allclose(
                result[column],
                up.get_curvature(nodes, tangents[0][:, [column]], s),
                rtol=RTOL,
                atol=ATOL,
            )


@pytest.mark.parametrize("num_vals", _SIZES)
def test_batched_evaluation_keeps_the_tail_of_a_family(num_vals):
    """The same tail, through the batched API rather than the per-curve one."""
    values = np.linspace(0.0, 1.0, num_vals)
    nodes = np.ascontiguousarray(np.stack(curve_family(5, 2, 6, seed=num_vals)))
    batched = Curves(nodes).evaluate_multi(values)
    assert batched.shape == (5, 2, num_vals)
    for index, curve_nodes in enumerate(nodes):
        np.testing.assert_allclose(
            batched[index],
            mine.evaluate_multi(curve_nodes, values),
            rtol=RTOL,
            atol=ATOL,
        )


#: Degrees either side of the `evaluate_multi` dispatch, plus a couple above
#: it, swept over every grid length that leaves a SIMD tail.
_HIGH_DEGREES = [54, 55, 56, 57, 70]


@pytest.mark.parametrize("degree", _HIGH_DEGREES)
@pytest.mark.parametrize("num_vals", _SIZES)
def test_above_the_vs_threshold_the_tail_is_the_de_casteljau_one(
    degree, num_vals, up
):
    """Past 55 nodes the dispatch runs the de Casteljau triangle instead.

    That path has no SIMD body at all -- one triangle per (curve, parameter) --
    so this covers the leftover parameters on the far side of the threshold,
    against upstream's own de Casteljau rather than its dispatcher. It is the
    high-degree half of the same tail question this module asks of the
    low-degree kernels, at the degrees `evaluate_multi` actually sends there.

    Every scratch buffer is allocated at exactly the size the kernel is
    documented to need, so a write past the end is a heap corruption rather
    than a number that happens to come out right.
    """
    nodes = np.ascontiguousarray(
        np.stack(curve_family(2, 2, degree, seed=degree + num_vals))
    )
    values = np.linspace(0.0, 1.0, num_vals)
    work = np.zeros(2 * max(degree, 1), dtype=np.float64)
    dst = np.zeros(2 * 2 * num_vals, dtype=np.float64)
    _lib.lib.bc_evaluate_multi_de_casteljau(
        _lib._addr(nodes), _lib._addr(1.0 - values), _lib._addr(values),
        int(work.ctypes.data), 2, 2, degree, num_vals, _lib._addr(dst),
    )
    result = dst.reshape((2, 2, num_vals))
    expected = np.stack(
        [up.evaluate_multi_de_casteljau(n, 1.0 - values, values) for n in nodes]
    )
    np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)
