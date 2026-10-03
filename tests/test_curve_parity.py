"""`mojo_bezier_curves.Curve` against `bezier.Curve`, method by method.

The point of these tests is drop-in-ness: same names, same argument order, same
return shapes, same numbers. Anything the port does differently has to be
visible here rather than discovered by a caller.
"""

import numpy as np
import pytest

import bezier

from conftest import ATOL, NAMED_NODES, RTOL, curve_family
from mojo_bezier_curves import Curve

_METHOD_CURVES = [name for name in sorted(NAMED_NODES) if name != "planar_line"]


def _pair(nodes):
    return bezier.Curve(nodes, degree=nodes.shape[1] - 1), Curve(
        nodes, nodes.shape[1] - 1
    )


def test_repr_matches_upstream():
    nodes = NAMED_NODES["quadratic"]
    theirs, ours = _pair(nodes)
    assert repr(ours) == repr(theirs)


@pytest.mark.parametrize("name", _METHOD_CURVES)
def test_from_nodes_and_properties(name):
    nodes = NAMED_NODES[name]
    curve = Curve.from_nodes(nodes)
    assert curve.degree == nodes.shape[1] - 1
    assert curve.dimension == nodes.shape[0]
    np.testing.assert_array_equal(curve.nodes, nodes)


@pytest.mark.parametrize("name", _METHOD_CURVES)
def test_evaluate_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    theirs, ours = _pair(nodes)
    for s in (0.0, 0.25, 0.75, 1.0):
        result = ours.evaluate(s)
        assert result.shape == (nodes.shape[0], 1)
        np.testing.assert_allclose(
            result, theirs.evaluate(s), rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("name", _METHOD_CURVES)
def test_evaluate_multi_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    theirs, ours = _pair(nodes)
    values = np.linspace(0.0, 1.0, 11)
    result = ours.evaluate_multi(values)
    assert result.shape == (nodes.shape[0], values.size)
    assert result.flags.f_contiguous
    np.testing.assert_allclose(
        result, theirs.evaluate_multi(values), rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("name", _METHOD_CURVES)
def test_evaluate_hodograph_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    theirs, ours = _pair(nodes)
    for s in (0.0, 0.3, 1.0):
        result = ours.evaluate_hodograph(s)
        assert result.shape == (nodes.shape[0], 1)
        np.testing.assert_allclose(
            result, theirs.evaluate_hodograph(s), rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("degree", [1, 2, 3, 6, 12])
def test_subdivide_matches_upstream(degree, up):
    for nodes in curve_family(2, 2, degree, seed=degree):
        theirs, ours = _pair(nodes)
        our_left, our_right = ours.subdivide()
        their_left, their_right = theirs.subdivide()
        assert our_left.degree == our_right.degree == degree
        np.testing.assert_allclose(our_left.nodes, their_left.nodes, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(our_right.nodes, their_right.nodes, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("degree", [0, 1, 2, 4, 9])
def test_elevate_matches_upstream(degree, up):
    for nodes in curve_family(2, 3, degree, seed=500 + degree):
        theirs, ours = _pair(nodes)
        assert ours.elevate().degree == theirs.elevate().degree
        np.testing.assert_allclose(
            ours.elevate().nodes, theirs.elevate().nodes, rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("start,end", [(0.0, 0.5), (0.3, 0.8), (-0.5, 0.25), (1.0, 2.0)])
@pytest.mark.parametrize("degree", [1, 2, 5])
def test_specialize_matches_upstream(start, end, degree, up):
    for nodes in curve_family(2, 2, degree, seed=600 + degree):
        theirs, ours = _pair(nodes)
        np.testing.assert_allclose(
            ours.specialize(start, end).nodes,
            theirs.specialize(start, end).nodes,
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize("name", ["planar_line", "quadratic", "cubic", "high_degree"])
def test_locate_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    theirs, ours = _pair(nodes)
    for s in (0.0, 0.2, 0.55, 1.0):
        point = theirs.evaluate(s)
        assert ours.locate(point) == pytest.approx(
            theirs.locate(point), rel=1e-9, abs=1e-9
        )


def test_locate_returns_none_off_the_curve(up):
    nodes = NAMED_NODES["quadratic"]
    theirs, ours = _pair(nodes)
    far_away = np.asfortranarray([[10.0], [-10.0]])
    assert ours.locate(far_away) is None
    assert theirs.locate(far_away) is None


@pytest.mark.parametrize("name", ["planar_line", "cubic", "high_degree"])
def test_length_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    theirs, ours = _pair(nodes)
    assert ours.length == pytest.approx(theirs.length, rel=1e-6)


def test_length_is_a_property_like_upstream():
    """`bezier.Curve.length` is a property; a caller must not need to call it."""
    nodes = NAMED_NODES["planar_line"]
    assert isinstance(type(Curve(nodes, 1)).length, property)


def test_copy_is_independent(up):
    nodes = NAMED_NODES["quadratic"]
    curve = Curve(nodes, 2)
    duplicate = curve.copy()
    assert duplicate == curve
    mutated = duplicate.nodes
    mutated[0, 0] = 99.0
    assert curve.nodes[0, 0] == 0.0


def test_constructor_rejects_a_degree_that_does_not_match():
    with pytest.raises(ValueError, match="degree"):
        Curve(NAMED_NODES["quadratic"], 3)
    with pytest.raises(ValueError, match="2D"):
        Curve(np.zeros((2, 2, 2)), 1)
    with pytest.raises(ValueError, match="finite"):
        Curve(np.asfortranarray([[0.0, np.nan], [0.0, 1.0]]), 1)


def test_degree_zero_curve_is_evaluable(up):
    nodes = np.asfortranarray([[1.0], [2.0]])
    theirs, ours = _pair(nodes)
    np.testing.assert_array_equal(ours.evaluate(0.4), theirs.evaluate(0.4))
    assert ours.length == 0.0
    assert theirs.length == 0.0
