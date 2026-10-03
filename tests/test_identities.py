"""Checks that do not lean on upstream at all.

Parity with a reference implementation is only as good as the reference, so
these tests state what the answer has to be from the mathematics: the Bernstein
polynomial written out longhand, the endpoints, the hodograph identity, the
arc length of a quarter circle, the behaviour of subdivision under
composition. A shared misreading of the algorithm cannot survive all of them,
because they are derived rather than compared.
"""

import math

import numpy as np
import pytest

from conftest import NAMED_NODES, curve_family
from mojo_bezier_curves import Curves
from mojo_bezier_curves import curve_helpers as mine


def bernstein(nodes, s_vals):
    """The Bernstein definition, longhand: no recurrence, no shared code.

    ``B(s) = sum_j C(degree, j) (1 - s)^(degree - j) s^j v_j``.
    """
    array = np.asarray(nodes, dtype=np.float64)
    degree = array.shape[1] - 1
    grid = np.atleast_1d(np.asarray(s_vals, dtype=np.float64))
    result = np.zeros((array.shape[0], grid.size), dtype=np.float64)
    for j in range(degree + 1):
        weight = np.array(
            [
                math.comb(degree, j) * (1.0 - t) ** (degree - j) * t**j
                for t in grid
            ]
        )
        result += array[:, j][:, np.newaxis] * weight[np.newaxis, :]
    return result


@pytest.mark.parametrize("degree", [1, 2, 3, 7, 12])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_evaluation_matches_the_bernstein_definition(degree, dim):
    for nodes in curve_family(2, dim, degree, seed=degree * 3 + dim):
        values = np.linspace(0.0, 1.0, 17)
        np.testing.assert_allclose(
            mine.evaluate_multi(nodes, values),
            bernstein(nodes, values),
            rtol=1e-12,
            atol=1e-13,
        )


def test_de_casteljau_matches_the_bernstein_definition():
    for nodes in curve_family(2, 2, 9, seed=77):
        values = np.linspace(0.0, 1.0, 13)
        lambda1 = 1.0 - values
        np.testing.assert_allclose(
            mine.evaluate_multi_de_casteljau(nodes, lambda1, values),
            bernstein(nodes, values),
            rtol=1e-12,
            atol=1e-13,
        )


@pytest.mark.parametrize("degree", [1, 2, 5, 8])
def test_the_endpoints_are_the_end_nodes(degree):
    for nodes in curve_family(3, 3, degree, seed=degree):
        result = mine.evaluate_multi(nodes, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(result[:, [0]], nodes[:, [0]])
        np.testing.assert_array_equal(result[:, [1]], nodes[:, [-1]])


@pytest.mark.parametrize("degree", [2, 3, 6])
def test_the_hodograph_is_the_type_function_of_the_forward_differences(degree):
    """`B'(s) = n * (evaluation of the unscaled forward differences)`.

    This is the identity the kernel implements, stated independently: the
    reference builds the differences in NumPy and evaluates them with the same
    longhand Bernstein sum, so a wrong scaling on either side shows up here.
    """
    for nodes in curve_family(2, 2, degree, seed=800 + degree):
        differences = nodes[:, 1:] - nodes[:, :-1]
        values = np.linspace(0.0, 1.0, 11)
        expected = degree * bernstein(differences, values)
        np.testing.assert_allclose(
            mine.evaluate_hodograph(0.3, nodes),
            (degree * bernstein(differences, np.array([0.3]))),
            rtol=1e-12,
            atol=1e-13,
        )
        batched = Curves(
            np.ascontiguousarray(np.stack([nodes]))
        ).evaluate_hodograph(values)[0]
        np.testing.assert_allclose(
            batched, expected, rtol=1e-12, atol=1e-13
        )


def _parabola():
    """`y = x^2` on `[0, 1]`, which is `B(s) = (s, s^2)` in the Bernstein basis."""
    return np.asfortranarray([[0.0, 0.5, 1.0], [0.0, 0.0, 1.0]])



def test_the_length_of_a_parabola_is_the_closed_form():
    """`y = x^2` on `[0, 1]` as a quadratic: the integral has a known value.

    `int_0^1 sqrt(1 + 4 x^2) dx = sqrt(5) / 2 + asinh(2) / 4`, so the Simpson
    sum can be checked against an answer derived from nothing but calculus.
    """
    parabola = _parabola()
    exact = 0.5 * math.sqrt(5.0) + math.asinh(2.0) / 4.0
    assert mine.compute_length(parabola, panels=8192) == pytest.approx(
        exact, rel=1e-12
    )


def test_a_straight_line_has_exactly_its_chord_length():
    nodes = np.asfortranarray([[0.0, 3.0, 6.0], [0.0, 4.0, 8.0]])
    assert mine.compute_length(nodes) == 10.0


def test_a_single_node_has_zero_length():
    assert mine.compute_length(np.asfortranarray([[1.0], [2.0]])) == 0.0


def test_the_curvature_of_a_parabola_is_its_closed_form():
    """`y = x^2` has curvature `2 / (1 + 4 x^2)^(3/2)` and `x(s) = s`."""
    parabola = _parabola()
    for s in (0.1, 0.5, 0.9):
        tangent = mine.evaluate_hodograph(s, parabola)
        assert mine.get_curvature(parabola, tangent, s) == pytest.approx(
            2.0 / (1.0 + 4.0 * s * s) ** 1.5, rel=1e-12
        )


def test_curvature_is_zero_along_a_straight_line():
    nodes = np.asfortranarray([[0.0, 3.0, 6.0], [0.0, 4.0, 8.0]])
    tangent = mine.evaluate_hodograph(0.5, nodes)
    assert mine.get_curvature(nodes, tangent, 0.5) == 0.0


def test_subdivision_composes_with_evaluation():
    """`left(u) == B(u / 2)` and `right(u) == B((1 + u) / 2)`, for any u."""
    for nodes in curve_family(2, 2, 5, seed=15):
        left, right = mine.subdivide_nodes(nodes)
        values = np.linspace(0.0, 1.0, 9)
        np.testing.assert_allclose(
            mine.evaluate_multi(left, values),
            mine.evaluate_multi(nodes, values / 2.0),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            mine.evaluate_multi(right, values),
            mine.evaluate_multi(nodes, (1.0 + values) / 2.0),
            rtol=1e-12,
            atol=1e-13,
        )


def test_the_two_halves_join_at_the_value_of_the_midpoint():
    for nodes in curve_family(3, 3, 4, seed=16):
        left, right = mine.subdivide_nodes(nodes)
        midpoint = mine.evaluate_multi(nodes, np.array([0.5]))
        np.testing.assert_allclose(left[:, -1:], right[:, [0]], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(left[:, -1:], midpoint, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(right[:, [0]], midpoint, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("degree", [1, 2, 4, 7])
def test_degree_elevation_leaves_the_geometry_alone(degree):
    for nodes in curve_family(2, 2, degree, seed=900 + degree):
        elevated = mine.elevate_nodes(nodes)
        assert elevated.shape[1] == nodes.shape[1] + 1
        values = np.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(
            mine.evaluate_multi(elevated, values),
            mine.evaluate_multi(nodes, values),
            rtol=1e-12,
            atol=1e-13,
        )
        np.testing.assert_allclose(elevated[:, 0], nodes[:, 0])
        np.testing.assert_allclose(elevated[:, -1], nodes[:, -1])


def test_specialization_reparameterises_the_same_points():
    """`specialize(a, b)` evaluated at `(1 - t) a + t b` is the original."""
    for nodes in curve_family(2, 2, 4, seed=17):
        start, end = 0.2, 0.85
        specialized = mine.specialize_curve(nodes, start, end)
        values = np.linspace(0.0, 1.0, 9)
        moved = start + values * (end - start)
        np.testing.assert_allclose(
            mine.evaluate_multi(specialized, values),
            mine.evaluate_multi(nodes, moved),
            rtol=1e-12,
            atol=1e-13,
        )


def test_a_newton_step_moves_towards_the_solution():
    """One step from either side of the root, the step shrinks the error."""
    for nodes in curve_family(2, 2, 3, seed=18):
        target = 0.3
        point = mine.evaluate_multi(nodes, np.array([target]))
        for start in (0.05, 0.5, 0.95):
            stepped = mine.newton_refine(nodes, point, start)
            assert abs(stepped - target) <= abs(start - target)


def test_newton_converges_to_the_parameter_of_a_known_point():
    for nodes in curve_family(2, 2, 4, seed=19):
        target = 0.42
        point = mine.evaluate_multi(nodes, np.array([target]))
        current = 0.8
        for _ in range(20):
            current = mine.newton_refine(nodes, point, current)
        assert current == pytest.approx(target, abs=1e-12)


def test_curvature_changes_sign_when_the_node_order_is_reversed():
    """`B_rev(s) = B(1 - s)`, so the sign flips but the magnitude does not."""
    nodes = NAMED_NODES["cubic"]
    reversed_nodes = np.asfortranarray(nodes[:, ::-1])
    for s in (0.2, 0.5, 0.8):
        tangent = mine.evaluate_hodograph(s, nodes)
        flipped = mine.evaluate_hodograph(1.0 - s, reversed_nodes)
        assert mine.get_curvature(nodes, tangent, s) == pytest.approx(
            -mine.get_curvature(reversed_nodes, flipped, 1.0 - s), rel=1e-9
        )
