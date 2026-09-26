"""Parity tests for subdivision, restriction, degree elevation and location."""

import numpy as np
import pytest

import mojo_bezier_curves as mbc
from test_evaluate import CURVES

# The batched kernels need a family that shares a dimension and a degree, so
# these are all planar cubics. Agreement between the batched path and the
# per-curve path is the point: a kernel that indexed the family wrongly would
# still look right on any single curve.
FAMILY = [
    np.asfortranarray([[0.0, 0.25, 0.8, 1.0], [0.0, 1.0, -0.2, 0.5]]),
    np.asfortranarray([[0.0, 0.5, 0.5, 1.0], [0.0, 2.0, -1.0, 0.0]]),
    np.asfortranarray([[-1.0, 0.0, 1.0, 2.0], [3.0, 1.0, 1.0, -3.0]]),
    np.asfortranarray([[0.1, 0.9, 0.2, 0.8], [1.0, 0.0, 1.0, 0.0]]),
]

# A curve whose first coordinate is the parameter itself, so B is injective and
# Newton's method converges from anywhere. The looping cubic above is not like
# that, and upstream's Newton loop does not converge on it either.
MONOTONE = np.asfortranarray([[0.0, 1.0 / 3, 2.0 / 3, 1.0], [0.0, 0.25, 0.75, 1.0]])


def family(nodes_list=None):
    return mbc.Curves(np.stack(FAMILY if nodes_list is None else nodes_list))


@pytest.mark.parametrize("nodes", CURVES)
def test_subdivision_halves_agree_with_the_whole_curve(nodes):
    """The left half is B on [0, 1/2] pulled back, the right half B on [1/2, 1].

    Sharing a midpoint is not enough: each half must trace the same curve.
    """
    curve = mbc.Curve(nodes)
    left, right = curve.subdivide()
    grid = np.linspace(0.0, 1.0, 97)
    np.testing.assert_allclose(
        left.evaluate_multi(grid), curve.evaluate_multi(0.5 * grid), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        right.evaluate_multi(grid), curve.evaluate_multi(0.5 + 0.5 * grid),
        rtol=1e-11, atol=1e-12,
    )


@pytest.mark.parametrize("nodes", CURVES)
def test_subdivision_halves_meet_at_the_midpoint(nodes):
    left, right = mbc.Curve(nodes).subdivide()
    np.testing.assert_allclose(left.nodes[:, -1], right.nodes[:, 0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        left.nodes[:, -1], mbc.Curve(nodes).evaluate(0.5)[:, 0], rtol=1e-12, atol=1e-13
    )


def test_batched_subdivision_matches_per_curve_subdivision():
    curves = family()
    left, right = curves.subdivide()
    for index, nodes in enumerate(FAMILY):
        one_left, one_right = mbc.Curve(nodes).subdivide()
        np.testing.assert_allclose(left[index].nodes, one_left.nodes, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(right[index].nodes, one_right.nodes, rtol=0.0, atol=0.0)


def test_batched_subdivision_keeps_each_curve_on_its_own_path():
    curves = family()
    left, _ = curves.subdivide()
    grid = np.linspace(0.0, 1.0, 33)
    expected = np.stack([mbc.bernstein(n, 0.5 * grid) for n in FAMILY])
    np.testing.assert_allclose(left.evaluate_multi(grid), expected, rtol=1e-12, atol=1e-13)


def test_subdivision_can_be_repeated():
    curves = family()
    quarters = curves.subdivide()[0].subdivide()[0]
    grid = np.linspace(0.0, 1.0, 33)
    np.testing.assert_allclose(
        quarters.evaluate_multi(grid), curves.evaluate_multi(0.25 * grid),
        rtol=1e-11, atol=1e-12,
    )


@pytest.mark.parametrize("nodes", CURVES)
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_restrict_to_a_prefix_traces_the_same_curve(nodes, t):
    curve = mbc.Curve(nodes)
    restricted = curve.restrict(t)
    grid = np.linspace(0.0, 1.0, 65)
    np.testing.assert_allclose(
        restricted.evaluate_multi(grid), curve.evaluate_multi(t * grid),
        rtol=1e-11, atol=1e-12,
    )


@pytest.mark.parametrize("nodes", CURVES)
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_restrict_to_a_suffix_traces_the_same_curve(nodes, t):
    curve = mbc.Curve(nodes)
    restricted = curve.restrict(t, from_end=True)
    grid = np.linspace(0.0, 1.0, 65)
    np.testing.assert_allclose(
        restricted.evaluate_multi(grid), curve.evaluate_multi(t + (1.0 - t) * grid),
        rtol=1e-11, atol=1e-12,
    )


@pytest.mark.parametrize("nodes", CURVES)
def test_restricting_to_a_degenerate_interval_collapses_the_curve(nodes):
    """Restricting to [0, 0] is the constant B(0); to [1, 1] the constant B(1)."""
    curve = mbc.Curve(nodes)
    grid = np.linspace(0.0, 1.0, 17)
    for t, from_end in ((0.0, False), (1.0, True)):
        restricted = curve.restrict(t, from_end=from_end)
        expected = curve.evaluate(t)[:, 0]
        np.testing.assert_allclose(
            restricted.evaluate_multi(grid),
            np.repeat(expected[:, np.newaxis], grid.size, axis=1),
            rtol=1e-12, atol=1e-13,
        )


@pytest.mark.parametrize("nodes", CURVES)
def test_a_full_restriction_is_the_identity(nodes):
    curve = mbc.Curve(nodes)
    np.testing.assert_allclose(curve.restrict(1.0).nodes, curve.nodes, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(curve.restrict(0.0, from_end=True).nodes, curve.nodes,
                               rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("nodes", CURVES)
@pytest.mark.parametrize("start,end", [(0.0, 1.0), (0.25, 0.75), (0.0, 0.5), (0.5, 1.0),
                                       (0.3, 0.3)])
def test_specialize_reparameterises_onto_the_unit_interval(nodes, start, end):
    curve = mbc.Curve(nodes)
    specialized = curve.specialize(start, end)
    grid = np.linspace(0.0, 1.0, 65)
    np.testing.assert_allclose(
        specialized.evaluate_multi(grid),
        curve.evaluate_multi(start + (end - start) * grid),
        rtol=1e-10, atol=1e-12,
    )
    np.testing.assert_allclose(specialized.nodes[:, 0], curve.evaluate(start)[:, 0],
                               rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(specialized.nodes[:, -1], curve.evaluate(end)[:, 0],
                               rtol=1e-12, atol=1e-13)


def test_specialize_to_the_whole_interval_is_the_identity():
    curve = mbc.Curve(CURVES[0])
    np.testing.assert_allclose(curve.specialize(0.0, 1.0).nodes, curve.nodes,
                               rtol=1e-14, atol=1e-15)


def test_specialize_rejects_an_inverted_interval():
    with pytest.raises(ValueError):
        mbc.Curve(CURVES[0]).specialize(0.75, 0.25)
    with pytest.raises(ValueError):
        mbc.Curve(CURVES[0]).specialize(-0.1, 0.5)
    with pytest.raises(ValueError):
        mbc.Curve(CURVES[0]).restrict(1.5)


@pytest.mark.parametrize("nodes", CURVES)
def test_degree_elevation_preserves_the_curve(nodes):
    curve = mbc.Curve(nodes)
    elevated = curve.elevate()
    assert elevated.degree == curve.degree + 1
    grid = np.linspace(0.0, 1.0, 129)
    np.testing.assert_allclose(elevated.evaluate_multi(grid), curve.evaluate_multi(grid),
                               rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(elevated.nodes[:, 0], nodes[:, 0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(elevated.nodes[:, -1], nodes[:, -1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(elevated.length(panels=1 << 14),
                               curve.length(panels=1 << 14), rtol=1e-9)


def test_degree_elevation_repeatedly_still_agrees():
    curve = mbc.Curve(CURVES[0])
    lifted = curve
    for _ in range(4):
        lifted = lifted.elevate()
    assert lifted.degree == curve.degree + 4
    grid = np.linspace(0.0, 1.0, 65)
    np.testing.assert_allclose(lifted.evaluate_multi(grid), curve.evaluate_multi(grid),
                               rtol=1e-11, atol=1e-12)


def test_batched_elevate_matches_per_curve_elevate():
    curves = family()
    lifted = curves.elevate()
    for index, nodes in enumerate(FAMILY):
        np.testing.assert_allclose(lifted[index].nodes, mbc.Curve(nodes).elevate().nodes,
                                   rtol=0.0, atol=0.0)


def test_batched_restrict_and_specialize_match_single_curve():
    curves = family()
    for t in (0.0, 0.3, 1.0):
        batched = curves.restrict(t)
        for index, nodes in enumerate(FAMILY):
            np.testing.assert_allclose(batched[index].nodes,
                                       mbc.Curve(nodes).restrict(t).nodes,
                                       rtol=0.0, atol=0.0)
    batched = curves.specialize(0.2, 0.8)
    for index, nodes in enumerate(FAMILY):
        np.testing.assert_allclose(batched[index].nodes,
                                   mbc.Curve(nodes).specialize(0.2, 0.8).nodes,
                                   rtol=0.0, atol=0.0)


def test_locate_finds_an_interior_parameter():
    curve = mbc.Curve(MONOTONE)
    for s in (0.11, 0.37, 0.5, 0.86):
        assert curve.locate(curve.evaluate(s)[:, 0]) == pytest.approx(s, abs=1e-9)


def test_locate_finds_the_endpoints():
    curve = mbc.Curve(MONOTONE)
    for s in (0.0, 1.0):
        assert curve.locate(curve.evaluate(s)[:, 0]) == pytest.approx(s, abs=1e-9)


def test_locate_returns_none_for_a_point_off_the_curve():
    curve = mbc.Curve(np.asfortranarray([[0.0, 0.5, 1.0], [0.0, 1.0, 0.0]]))
    assert curve.locate(np.array([5.0, 5.0])) is None


def test_locate_rejects_a_point_of_the_wrong_dimension():
    curve = mbc.Curve(CURVES[0])
    with pytest.raises(ValueError):
        curve.locate(np.array([1.0, 2.0, 3.0]))


def test_locate_handles_many_points_in_one_call():
    """The compiled Newton loop takes every point of every curve at once."""
    curves = mbc.Curves(np.stack([MONOTONE, MONOTONE + 0.5]))
    grid = np.linspace(0.0, 1.0, 12)
    points = np.ascontiguousarray(curves.evaluate_multi(grid).transpose(0, 2, 1))
    starts = np.full(points.shape[:2], 0.5)
    refined = mbc._lib.newton_refine(curves.nodes, points, starts, 24)
    for index in range(points.shape[0]):
        for column, s in enumerate(grid):
            assert refined[index, column] == pytest.approx(s, abs=1e-12)


def test_newton_refine_validates_its_shapes():
    curves = family()
    n = len(FAMILY)
    with pytest.raises(ValueError):
        mbc._lib.newton_refine(curves.nodes, np.zeros((n, 3, 7)), np.zeros((n, 3)), 2)
    with pytest.raises(ValueError):
        mbc._lib.newton_refine(curves.nodes, np.zeros((n, 3, 2)), np.zeros((n, 4)), 2)
