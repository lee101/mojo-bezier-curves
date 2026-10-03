"""`Curves`: the batched family, and the claim that it is the same answer.

The kernels carry a leading curve index, so the batched results are not merely
close to the per-curve results, they are the same arithmetic on different
addresses. Most of these tests assert exact equality for that reason, and fall
back to upstream agreement where the comparison is against Python.
"""

import numpy as np
import pytest

import bezier

from conftest import ATOL, RTOL, curve_family
from mojo_bezier_curves import Curve, Curves
from mojo_bezier_curves import curve_helpers as mine


def _stacked(count, dim, degree, seed=0):
    return np.ascontiguousarray(
        np.stack(curve_family(count, dim, degree, seed=seed))
    )


def test_batched_evaluate_is_the_per_curve_answer():
    """The batched result against the per-curve one, on the same grid.

    The batched path runs `W` parameters at a time and the per-curve path one
    at a time, into the Fortran layout the drop-in API returns. Same
    arithmetic, so they agree to the tolerance the fused multiply-add
    difference needs rather than bit for bit.
    """
    nodes = _stacked(5, 2, 7, seed=1)
    family = Curves(nodes)
    values = np.linspace(0.0, 1.0, 13)
    batched = family.evaluate_multi(values)
    for index in range(len(family)):
        expected = mine.evaluate_multi(nodes[index], values)
        np.testing.assert_allclose(batched[index], expected, rtol=RTOL, atol=ATOL)


def test_batched_evaluate_is_exact_against_a_batched_per_curve_call():
    """Both sides on the same layout, so both sides are the same code path."""
    nodes = _stacked(5, 2, 7, seed=1)
    values = np.linspace(0.0, 1.0, 13)
    batched = Curves(nodes).evaluate_multi(values)
    for index in range(nodes.shape[0]):
        one = Curves(nodes[index:index + 1]).evaluate_multi(values)
        np.testing.assert_array_equal(batched[index], one[0])


def test_batched_evaluate_matches_upstream_curve_by_curve():
    nodes = _stacked(4, 2, 5, seed=2)
    family = Curves(nodes)
    values = np.linspace(0.0, 1.0, 9)
    batched = family.evaluate_multi(values)
    for index, curve_nodes in enumerate(nodes):
        theirs = bezier.Curve(curve_nodes, degree=curve_nodes.shape[1] - 1)
        np.testing.assert_allclose(
            batched[index], theirs.evaluate_multi(values), rtol=RTOL, atol=ATOL
        )


def test_batched_hodograph_matches_upstream_curve_by_curve():
    nodes = _stacked(4, 3, 4, seed=3)
    family = Curves(nodes)
    values = np.linspace(0.0, 1.0, 7)
    batched = family.evaluate_hodograph(values)
    for index, curve_nodes in enumerate(nodes):
        theirs = bezier.Curve(curve_nodes, degree=curve_nodes.shape[1] - 1)
        expected = np.column_stack(
            [theirs.evaluate_hodograph(s).ravel() for s in values]
        )
        np.testing.assert_allclose(batched[index], expected, rtol=RTOL, atol=ATOL)


def test_batched_subdivide_matches_upstream_curve_by_curve():
    nodes = _stacked(3, 2, 6, seed=4)
    family = Curves(nodes)
    left, right = family.subdivide()
    assert left.count == right.count == 3
    for index, curve_nodes in enumerate(nodes):
        theirs = bezier.Curve(curve_nodes, degree=curve_nodes.shape[1] - 1)
        their_left, their_right = theirs.subdivide()
        np.testing.assert_allclose(left[index].nodes, their_left.nodes, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(right[index].nodes, their_right.nodes, rtol=RTOL, atol=ATOL)


def test_batched_elevate_and_specialize_match_upstream_curve_by_curve():
    nodes = _stacked(3, 2, 3, seed=5)
    family = Curves(nodes)
    elevated = family.elevate()
    specialized = family.specialize(0.2, 0.9)
    for index, curve_nodes in enumerate(nodes):
        theirs = bezier.Curve(curve_nodes, degree=curve_nodes.shape[1] - 1)
        np.testing.assert_allclose(
            elevated[index].nodes, theirs.elevate().nodes, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            specialized[index].nodes,
            theirs.specialize(0.2, 0.9).nodes,
            rtol=RTOL,
            atol=ATOL,
        )


def test_batched_curvature_matches_the_scalar_helper():
    nodes = _stacked(4, 2, 4, seed=6)
    family = Curves(nodes)
    values = np.linspace(0.1, 0.9, 6)
    batched = family.get_curvature(values)
    for b in range(4):
        for column, s in enumerate(values):
            tangent = mine.evaluate_hodograph(s, nodes[b])
            assert batched[b, column] == pytest.approx(
                mine.get_curvature(nodes[b], tangent, s), rel=RTOL, abs=ATOL
            )


def test_batched_length_matches_upstream_curve_by_curve():
    nodes = _stacked(3, 2, 5, seed=7)
    lengths = Curves(nodes).length(panels=8192)
    for index, curve_nodes in enumerate(nodes):
        theirs = bezier.Curve(curve_nodes, degree=curve_nodes.shape[1] - 1)
        assert lengths[index] == pytest.approx(theirs.length, rel=1e-8)


def test_batched_newton_matches_the_scalar_helper_bit_for_bit():
    nodes = _stacked(3, 2, 4, seed=8)
    family = Curves(nodes)
    values = np.linspace(0.1, 0.9, 5)
    points = np.ascontiguousarray(family.evaluate_multi(values).transpose(0, 2, 1))
    starts = np.full((3, values.size), 0.6)
    batched = family.newton_refine(points, starts)
    for b in range(3):
        for k in range(values.size):
            expected = mine.newton_refine(
                nodes[b], points[b, k, :], starts[b, k]
            )
            assert batched[b, k] == pytest.approx(expected, rel=RTOL, abs=ATOL)


def test_a_family_of_one_matches_the_single_curve_api():
    nodes = _stacked(1, 2, 4, seed=9)
    family = Curves(nodes)
    single = Curve(nodes[0], 4)
    values = np.linspace(0.0, 1.0, 5)
    np.testing.assert_array_equal(
        family.evaluate_multi(values)[0], single.evaluate_multi(values)
    )


def test_family_indexing_and_iteration_agree():
    nodes = _stacked(4, 2, 3, seed=10)
    family = Curves(nodes)
    assert len(family) == 4
    assert family.count == 4
    assert family.dimension == 2
    assert family.degree == 3
    for index, curve in enumerate(family):
        np.testing.assert_array_equal(curve.nodes, nodes[index])
    np.testing.assert_array_equal(family[2].nodes, family[2].nodes)


def test_family_accepts_a_single_curve():
    family = Curves(np.asfortranarray([[0.0, 1.0, 2.0], [0.0, 1.0, 0.0]]))
    assert family.count == 1
    assert family.degree == 2


def test_family_rejects_ragged_or_non_finite_nodes():
    with pytest.raises(ValueError, match="rank"):
        Curves(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="finite"):
        Curves(np.ascontiguousarray([[[0.0, np.inf], [0.0, 1.0]]]))


def test_a_high_degree_family_switches_to_de_casteljau():
    """The dispatch threshold is on the family, not per curve."""
    degree = 56
    angles = np.linspace(0.0, degree / 20.0, degree + 1)
    one = np.ascontiguousarray(
        np.stack([np.stack([np.cos(angles), np.sin(angles)])])
    )
    family = Curves(one)
    values = np.linspace(0.0, 1.0, 4)
    np.testing.assert_allclose(
        family.evaluate_multi(values)[0],
        mine.evaluate_multi(one[0], values),
        rtol=RTOL,
        atol=ATOL,
    )
