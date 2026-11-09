"""Parity for the public functions that had no direct upstream comparison, and
the degenerate inputs where upstream defines the behaviour.

Extend of the existing parity suites: `test_hazmat_parity` compares most
helpers and `test_curve_parity` compares every `Curve` method, but
`evaluate_multi_barycentric` had no comparison at all and the scalar
`evaluate_hodograph` helper was only exercised through the batched kernel
and the `Curve` method, never helper-to-helper with a tolerance. The
degenerate cases below are likewise comparisons, not just assertions about
the port: empty and one-element grids, duplicate nodes, NaN/inf parameters,
mismatched shapes, a degree-0 curve, and subdivision matrices for a
negative or non-integer degree.

Where upstream raises incidentally (an ``IndexError`` from indexing a
zero-width array) the port raises ``ValueError`` instead; those cases are
pinned as ``ValueError`` here and documented, not asserted as upstream
parity. Where both sides raise ``ValueError`` for a mismatched shape the
test asserts that both raise it. Where upstream propagates NaN/inf
through a parameter value the port does the same, and the test compares
with ``equal_nan`` so a port that returned a finite number would fail.
"""

import numpy as np
import pytest

from conftest import ATOL, NAMED_NODES, RTOL, curve_family
from mojo_bezier_curves import Curve, Curves
from mojo_bezier_curves import curve_helpers as mine


def _values(count=7):
    return np.linspace(0.0, 1.0, count)


# --------------------------------------------------------------------------
# evaluate_multi_barycentric: the one public helper with no direct test.
# The dispatch is `num_nodes > 55 -> de Casteljau`, so degrees on either
# side of the threshold and at the extremes are all compared.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NAMED_NODES))
def test_evaluate_multi_barycentric_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    values = _values()
    lambda1 = 1.0 - values
    result = mine.evaluate_multi_barycentric(nodes, lambda1, values)
    assert result.shape == (nodes.shape[0], values.size)
    assert result.flags.f_contiguous
    np.testing.assert_allclose(
        result,
        up.evaluate_multi_barycentric(nodes, lambda1, values),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 13, 54, 55, 56, 70])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_evaluate_multi_barycentric_matches_upstream_on_random_curves(degree, dim, up):
    values = _values(9)
    lambda1 = 1.0 - values
    for nodes in curve_family(3, dim, degree, seed=degree + dim * 7):
        np.testing.assert_allclose(
            mine.evaluate_multi_barycentric(nodes, lambda1, values),
            up.evaluate_multi_barycentric(nodes, lambda1, values),
            rtol=RTOL,
            atol=ATOL,
        )


def test_evaluate_multi_barycentric_dispatches_like_evaluate_multi(up):
    """Both dispatchers switch at the same node count, on either side."""
    values = _values(5)
    lambda1 = 1.0 - values
    for degree in (54, 55, 56, 57):
        for nodes in curve_family(2, 2, degree, seed=degree):
            np.testing.assert_allclose(
                mine.evaluate_multi_barycentric(nodes, lambda1, values),
                mine.evaluate_multi(nodes, values),
                rtol=RTOL,
                atol=ATOL,
            )
            np.testing.assert_allclose(
                mine.evaluate_multi_barycentric(nodes, lambda1, values),
                up.evaluate_multi_barycentric(nodes, lambda1, values),
                rtol=RTOL,
                atol=ATOL,
            )


# --------------------------------------------------------------------------
# evaluate_hodograph / specialize_curve called helper-to-helper.
# test_curve_parity covers Curve.evaluate_hodograph and test_buffers
# covers the batched kernel; the scalar helper that Curve delegates to had
# no direct tolerance comparison.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NAMED_NODES))
@pytest.mark.parametrize("s", [0.0, 0.3, 0.5, 1.0])
def test_evaluate_hodograph_helper_matches_upstream(name, s, up):
    if name in ("line", "planar_line"):
        # A line's hodograph is constant; still compared, not skipped for
        # the degree-1 case it represents.
        pass
    nodes = NAMED_NODES[name]
    result = mine.evaluate_hodograph(s, nodes)
    assert result.shape == (nodes.shape[0], 1)
    assert result.flags.f_contiguous
    np.testing.assert_allclose(
        result, up.evaluate_hodograph(s, nodes), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("degree", [1, 2, 5, 9])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_evaluate_hodograph_helper_matches_upstream_on_random_curves(degree, dim, up):
    for nodes in curve_family(3, dim, degree, seed=700 + degree + dim):
        for s in (0.0, 0.25, 1.0):
            np.testing.assert_allclose(
                mine.evaluate_hodograph(s, nodes),
                up.evaluate_hodograph(s, nodes),
                rtol=RTOL,
                atol=ATOL,
            )


@pytest.mark.parametrize(
    "start,end",
    [(0.8, 0.2), (2.0, 1.0), (-1.0, -2.0), (0.5, 0.5)],
)
def test_specialize_curve_reversed_or_degenerate_intervals_match_upstream(start, end, up):
    """`specialize` re-parameterises whatever [start, end] is, even reversed."""
    for nodes in curve_family(2, 2, 4, seed=42 + int(abs(start * 10) + abs(end * 10))):
        np.testing.assert_allclose(
            mine.specialize_curve(nodes, start, end),
            up.specialize_curve(nodes, start, end),
            rtol=RTOL,
            atol=ATOL,
        )


def test_specialize_curve_with_nan_start_matches_upstream(up):
    """A NaN bound propagates through both sides rather than raising."""
    nodes = NAMED_NODES["quadratic"]
    with np.errstate(invalid="ignore"):
        expected = up.specialize_curve(nodes, np.nan, 0.75)
        result = mine.specialize_curve(nodes, np.nan, 0.75)
    np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL, equal_nan=True)


# --------------------------------------------------------------------------
# Empty and one-element grids: both sides return an empty-shaped array.
# Upstream's `evaluate_multi_vs`/`_de_casteljau` key on `lambda1.shape`, so
# an empty grid is a legitimate input, not an error.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [0, 1, 2, 5])
def test_empty_parameter_grid_matches_upstream(degree, up):
    empty = np.array([], dtype=np.float64)
    for nodes in curve_family(2, 2, degree, seed=degree + 11):
        for evaluate in (mine.evaluate_multi, up.evaluate_multi):
            assert evaluate(nodes, empty).shape == (2, 0)
        result = mine.evaluate_multi(nodes, empty)
        np.testing.assert_allclose(
            result, up.evaluate_multi(nodes, empty), rtol=RTOL, atol=ATOL
        )
        assert result.flags.f_contiguous


@pytest.mark.parametrize("degree", [1, 2, 5])
def test_empty_barycentric_grids_match_upstream(degree, up):
    empty = np.array([], dtype=np.float64)
    for nodes in curve_family(2, 2, degree, seed=degree + 12):
        for function in (mine.evaluate_multi_vs, up.evaluate_multi_vs):
            assert function(nodes, empty, empty).shape == (2, 0)
        np.testing.assert_allclose(
            mine.evaluate_multi_vs(nodes, empty, empty),
            up.evaluate_multi_vs(nodes, empty, empty),
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            mine.evaluate_multi_de_casteljau(nodes, empty, empty),
            up.evaluate_multi_de_casteljau(nodes, empty, empty),
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            mine.evaluate_multi_barycentric(nodes, empty, empty),
            up.evaluate_multi_barycentric(nodes, empty, empty),
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize("degree", [0, 1, 5])
def test_one_element_grid_matches_upstream(degree, up):
    values = np.array([0.37])
    for nodes in curve_family(2, 3, degree, seed=degree + 13):
        np.testing.assert_allclose(
            mine.evaluate_multi(nodes, values),
            up.evaluate_multi(nodes, values),
            rtol=RTOL,
            atol=ATOL,
        )
        np.testing.assert_allclose(
            mine.evaluate_multi_barycentric(nodes, 1.0 - values, values),
            up.evaluate_multi_barycentric(nodes, 1.0 - values, values),
            rtol=RTOL,
            atol=ATOL,
        )


def test_curve_evaluate_multi_with_zero_or_one_value(up):
    nodes = NAMED_NODES["cubic"]
    import bezier

    theirs, ours = bezier.Curve(nodes, 3), Curve(nodes, 3)
    assert ours.evaluate_multi(np.array([])).shape == (2, 0)
    np.testing.assert_allclose(
        ours.evaluate_multi(np.array([0.25])),
        theirs.evaluate_multi(np.array([0.25])),
        rtol=RTOL,
        atol=ATOL,
    )


def test_batched_empty_grid_shape_matches_scalar(up):
    """The family keeps the same (curves, dim, 0) shape on an empty grid."""
    nodes = np.ascontiguousarray(
        np.stack(curve_family(3, 2, 4, seed=17))
    )
    result = Curves(nodes).evaluate_multi(np.array([], dtype=np.float64))
    assert result.shape == (3, 2, 0)
    for index in range(3):
        assert result[index].shape == (2, 0)


# --------------------------------------------------------------------------
# Duplicate nodes: a curve whose control points coincide is a degenerate
# but well-defined point. Contraction by zero tangent must give NaN in
# both, and evaluation must give the coincident point.
# --------------------------------------------------------------------------


def test_duplicate_nodes_evaluate_to_the_coincident_point(up):
    nodes = np.asfortranarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    values = _values(5)
    expected = up.evaluate_multi(nodes, values)
    np.testing.assert_allclose(
        mine.evaluate_multi(nodes, values), expected, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(expected, np.array([[1.0] * 5, [2.0] * 5]))
    np.testing.assert_allclose(
        mine.subdivide_nodes(nodes)[0],
        up.subdivide_nodes(nodes)[0],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        mine.elevate_nodes(nodes), up.elevate_nodes(nodes), rtol=RTOL, atol=ATOL
    )


def test_duplicate_nodes_have_zero_length_and_nan_derivative_quantities(up):
    """Zero tangent divided by zero: both sides produce NaN, not an error."""
    nodes = np.asfortranarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert mine.compute_length(nodes) == pytest.approx(
        up.compute_length(nodes), rel=RTOL, abs=ATOL
    )
    assert mine.compute_length(nodes) == 0.0
    with np.errstate(invalid="ignore"):
        assert np.isnan(
            mine.newton_refine(nodes, np.asfortranarray([[1.0], [2.0]]), 0.5)
        )
        assert np.isnan(
            up.newton_refine(nodes, np.asfortranarray([[1.0], [2.0]]), 0.5)
        )
        tangent = mine.evaluate_hodograph(0.5, nodes)
        assert np.isnan(mine.get_curvature(nodes, tangent, 0.5))
        assert np.isnan(up.get_curvature(nodes, tangent, 0.5))


def test_subdivide_and_elevate_of_a_degree_zero_curve_match_upstream(up):
    """One node subdivides into two copies of itself; elevation raises it."""
    nodes = np.asfortranarray([[1.0], [2.0]])
    left, right = mine.subdivide_nodes(nodes)
    expected_left, expected_right = up.subdivide_nodes(nodes)
    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)
    np.testing.assert_allclose(
        mine.elevate_nodes(nodes), up.elevate_nodes(nodes), rtol=RTOL, atol=ATOL
    )


def test_de_casteljau_one_round_of_a_single_node_is_empty_like_upstream(up):
    """Upstream's `de_casteljau_one_round` slices `nodes[:, :-1]`, so a
    single node gives a `(dimension, 0)` array; the port returns the same."""
    nodes = np.asfortranarray([[1.0], [2.0]])
    result = mine.de_casteljau_one_round(nodes, 0.3, 0.7)
    assert result.shape == (2, 0)
    assert result.flags.f_contiguous
    np.testing.assert_array_equal(result, up.de_casteljau_one_round(nodes, 0.3, 0.7))


# --------------------------------------------------------------------------
# NaN and inf in the parameter (not in the nodes: the port refuses
# non-finite nodes where upstream propagates them, and that refusal is
# pinned in test_paths/test_curve_parity). Upstream returns NaN / inf
# results for these grids, so the port must too.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("s", [np.nan, np.inf, -np.inf])
def test_non_finite_parameter_matches_upstream(s, up):
    nodes = NAMED_NODES["quadratic"]
    values = np.array([s])
    with np.errstate(invalid="ignore", divide="ignore"):
        np.testing.assert_allclose(
            mine.evaluate_multi(nodes, values),
            up.evaluate_multi(nodes, values),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            mine.evaluate_multi_vs(nodes, 1.0 - values, values),
            up.evaluate_multi_vs(nodes, 1.0 - values, values),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            mine.evaluate_hodograph(s, nodes),
            up.evaluate_hodograph(s, nodes),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        )
        assert mine.vec_size(nodes, s) == pytest.approx(
            up.vec_size(nodes, s), rel=RTOL, abs=ATOL, nan_ok=True
        )


def test_non_finite_newton_and_curvature_parameters_match_upstream(up):
    nodes = NAMED_NODES["quadratic"]
    point = up.evaluate_multi(nodes, np.array([0.4]))
    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.isnan(mine.newton_refine(nodes, point, np.nan))
        assert np.isnan(up.newton_refine(nodes, point, np.nan))
        tangent = mine.evaluate_hodograph(0.5, nodes)
        assert np.isnan(mine.get_curvature(nodes, tangent, np.nan))
        assert np.isnan(up.get_curvature(nodes, tangent, np.nan))


def test_locate_point_with_nan_point_returns_none_in_both(up):
    nodes = NAMED_NODES["quadratic"]
    off = np.asfortranarray([[np.nan], [0.0]])
    assert mine.locate_point(nodes, off) is None
    assert up.locate_point(nodes, off) is None


# --------------------------------------------------------------------------
# Mismatched shapes: where upstream gets a broadcasting ValueError the
# port raises ValueError; where upstream accepts a scalar s the port
# documents its own flattening, so only the errors are compared.
# --------------------------------------------------------------------------


def test_mismatched_barycentric_lengths_raise_value_error_in_both(up):
    """Upstream's VS path raises ValueError on (2,) vs (3,) broadcast."""
    nodes = NAMED_NODES["quadratic"]
    lambda1 = np.array([0.5, 0.5])
    lambda2 = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        up.evaluate_multi_vs(nodes, lambda1, lambda2)
    with pytest.raises(ValueError):
        mine.evaluate_multi_vs(nodes, lambda1, lambda2)
    with pytest.raises(ValueError):
        up.evaluate_multi_barycentric(nodes, lambda1, lambda2)
    with pytest.raises(ValueError):
        mine.evaluate_multi_barycentric(nodes, lambda1, lambda2)
    with pytest.raises(ValueError):
        mine.evaluate_multi_de_casteljau(nodes, lambda1, lambda2)


def test_mismatched_point_dimension_raises_value_error_in_both(up):
    nodes = NAMED_NODES["quadratic"]
    big_point = np.asfortranarray([[1.0], [1.0], [1.0]])
    with pytest.raises(ValueError):
        up.newton_refine(nodes, big_point, 0.5)
    with pytest.raises(ValueError):
        mine.newton_refine(nodes, big_point, 0.5)
    with pytest.raises(ValueError):
        up.locate_point(nodes, big_point)
    with pytest.raises(ValueError):
        mine.locate_point(nodes, big_point)


def test_nodes_with_wrong_rank_raise_value_error_in_both(up):
    with pytest.raises(ValueError):
        up.evaluate_multi(np.zeros((2, 2, 2)), np.array([0.0]))
    with pytest.raises(ValueError):
        mine.evaluate_multi(np.zeros((2, 2, 2)), np.array([0.0]))
    with pytest.raises(ValueError):
        up.evaluate_multi(np.array([1.0, 2.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        mine.evaluate_multi(np.array([1.0, 2.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        up.compute_length(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        mine.compute_length(np.zeros((2, 2, 2)))


def test_make_subdivision_matrices_rejects_a_negative_degree():
    """Upstream indexes `left[0, 0]` on a 0x0 array: IndexError there,
    ValueError here, and the port must not write past a zero-width buffer."""
    with pytest.raises(IndexError):
        __import__("bezier.hazmat.curve_helpers", fromlist=["x"]).make_subdivision_matrices(-1)
    with pytest.raises(ValueError):
        mine.make_subdivision_matrices(-1)
    with pytest.raises(ValueError):
        mine.make_subdivision_matrices(-2)
    # Upstream requires an integer (`float` cannot be interpreted as one);
    # the port truncates via `int()`, so a non-integral float is not
    # asserted to raise here — only the negative-degree guard is parity.
    assert mine.make_subdivision_matrices(2.5)[0].shape == (3, 3)


def test_degree_zero_curve_value_behaviour_matches_upstream(up):
    """The port and upstream agree on values where upstream defines them."""
    nodes = np.asfortranarray([[1.0], [2.0]])
    values = np.array([0.0, 0.4, 1.0])
    np.testing.assert_array_equal(
        mine.evaluate_multi(nodes, values), up.evaluate_multi(nodes, values)
    )
    np.testing.assert_array_equal(
        mine.evaluate_multi_barycentric(nodes, 1.0 - values, values),
        up.evaluate_multi_barycentric(nodes, 1.0 - values, values),
    )
    assert mine.vec_size(nodes, 0.5) == pytest.approx(
        up.vec_size(nodes, 0.5), rel=RTOL, abs=ATOL
    )
    assert mine.compute_length(nodes) == up.compute_length(nodes) == 0.0
    # Upstream indexes a zero-width difference array for the tangent; the
    # port reports the missing derivative as ValueError, as documented.
    with pytest.raises(ValueError, match="tangent"):
        mine.evaluate_hodograph(0.5, nodes)


def test_empty_nodes_compute_length_raises_like_upstream(up):
    """`compute_length` is the one empty-nodes case upstream names: ValueError."""
    empty = np.asfortranarray(np.zeros((2, 0)))
    with pytest.raises(ValueError, match="at least one node"):
        up.compute_length(empty)
    with pytest.raises(ValueError, match="at least one node"):
        mine.compute_length(empty)
