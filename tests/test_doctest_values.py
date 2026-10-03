"""The values printed in upstream's own docstrings.

These are the numbers `bezier` publishes as correct in its documentation: the
evaluate and hodograph examples, the curvature example, the specialize
example, and the Newton convergence rates. They are fixed points that neither
implementation is asked to produce, so they catch a port that is self
consistent and wrong.
"""

import math

import numpy as np
import pytest

from mojo_bezier_curves import Curve
from mojo_bezier_curves import curve_helpers as mine


def test_curve_evaluate_example():
    """`bezier.Curve` documents `evaluate(0.75)` on this quadratic."""
    nodes = np.asfortranarray([[0.0, 0.625, 1.0], [0.0, 0.5, 0.5]])
    np.testing.assert_array_equal(
        Curve(nodes, 2).evaluate(0.75), np.array([[0.796875], [0.46875]])
    )


def test_curve_evaluate_multi_example():
    """`bezier.Curve` documents this line evaluated on five parameters."""
    nodes = np.asfortranarray([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    np.testing.assert_array_equal(
        Curve(nodes, 1).evaluate_multi(np.linspace(0.0, 1.0, 5)),
        np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0],
                [0.0, 0.5, 1.0, 1.5, 2.0],
                [0.0, 0.75, 1.5, 2.25, 3.0],
            ]
        ),
    )


def test_curve_evaluate_hodograph_example():
    """`bezier.Curve` documents `evaluate_hodograph(0.75)` on this quadratic."""
    nodes = np.asfortranarray([[0.0, 0.625, 1.0], [0.0, 0.5, 0.5]])
    np.testing.assert_allclose(
        Curve(nodes, 2).evaluate_hodograph(0.75), np.array([[0.875], [0.25]])
    )


def test_curve_subdivide_example():
    """`bezier.Curve.subdivide` documents these exact node arrays."""
    nodes = np.asfortranarray([[0.0, 1.25, 2.0], [0.0, 3.0, 1.0]])
    left, right = Curve(nodes, 2).subdivide()
    np.testing.assert_allclose(
        left.nodes, np.array([[0.0, 0.625, 1.125], [0.0, 1.5, 1.75]])
    )
    np.testing.assert_allclose(
        right.nodes, np.array([[1.125, 1.625, 2.0], [1.75, 2.0, 1.0]])
    )


def test_curve_specialize_example():
    """`bezier.Curve.specialize(-0.25, 0.75)` documents these nodes."""
    nodes = np.asfortranarray([[0.0, 0.5, 1.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        Curve(nodes, 2).specialize(-0.25, 0.75).nodes,
        np.array([[-0.25, 0.25, 0.75], [-0.625, 0.875, 0.375]]),
    )


def test_get_curvature_example():
    """`get_curvature`'s doctest: tangent `(-1, 0)` and curvature `-12.0`."""
    nodes = np.asfortranarray(
        [[1.0, 0.75, 0.5, 0.25, 0.0], [0.0, 2.0, -2.0, 2.0, 0.0]]
    )
    tangent = mine.evaluate_hodograph(0.5, nodes)
    np.testing.assert_array_equal(tangent, np.array([[-1.0], [0.0]]))
    assert float(mine.get_curvature(nodes, tangent, 0.5)) == -12.0


def test_newton_refine_converges_as_documented():
    """`newton_refine`'s doctest: the error roughly squares each step.

    Upstream's curve is the cusp `[[6, -2, -2, 6], [-3, 3, -3, 3]]` and the
    documented log2 errors are -3, -3.983, -4.979, -5.978, -6.978, -7.978 from a
    start of 5/8. Those are the published values, so a port that iterates the
    wrong quantity or the wrong number of times cannot match them.
    """
    nodes = np.asfortranarray([[6.0, -2.0, -2.0, 6.0], [-3.0, 3.0, -3.0, 3.0]])
    point = mine.evaluate_multi(nodes, np.array([0.5]))
    expected = 0.5
    current = 0.625
    assert math.log2(abs(expected - current)) == -3.0
    documented = [-3.983, -4.979, -5.978, -6.978, -7.978]
    for index, target in enumerate(documented):
        current = mine.newton_refine(nodes, point, current)
        assert math.log2(abs(expected - current)) == pytest.approx(
            target, abs=0.001
        ), f"step {index + 1}"


def test_newton_refine_terminates_near_the_cusp():
    """The continuation doctest: near the cusp the floor is sqrt(epsilon).

    Upstream states the final error sits between 2**-31 and 2**-28, which is a
    property of the iteration near a stationary point and not of any tolerance
    the port might pick.
    """
    nodes = np.asfortranarray([[6.0, -2.0, -2.0, 6.0], [-3.0, 3.0, -3.0, 3.0]])
    point = mine.evaluate_multi(nodes, np.array([0.5]))
    current = 0.625
    seen = [current]
    for _ in range(200):
        following = mine.newton_refine(nodes, point, current)
        if following == current:
            break
        seen.append(following)
        current = following
    assert current == mine.newton_refine(nodes, point, seen[-1])
    assert 2.0**-31 <= abs(seen[-1] - 0.5) <= 2.0**-28
