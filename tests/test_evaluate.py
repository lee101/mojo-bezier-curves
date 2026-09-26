"""Parity tests for Bezier evaluation, derivatives and length.

The upstream `bezier` package is not installed in the parity environment (it
requires NumPy 2 and the shared test venv has NumPy 1.26), so these tests check
the compiled kernels two ways:

* against `mojo_bezier_curves.bernstein`, the Bernstein definition written out
  longhand with `math.comb` and per-parameter scalar powers. It shares no code
  with either compiled recurrence, so a mistake in the VS loop or in the
  de Casteljau triangle cannot hide behind the same mistake in the reference;
* against analytic identities that only the true curve satisfies -- endpoint
  values, the derivative of a known polynomial, the closed-form curvature of a
  quadratic at its apex, and the length of a straight segment.
"""

import numpy as np
import pytest

import mojo_bezier_curves as mbc

CURVES = [
    np.asfortranarray([[0.0, 0.25, 0.8, 1.0], [0.0, 1.0, -0.2, 0.5]]),
    np.asfortranarray([[-2.0, 0.0, 1.0], [1.0, -1.0, 2.0]]),
    np.asfortranarray([[0.0, 1.0], [2.0, -1.0], [1.0, 3.0]]),
    np.asfortranarray([[1.0, 2.0, 3.0, 4.0, 5.0]]),
    np.asfortranarray([[0.0, 0.5, 1.0], [0.0, 1.0, 0.0]]),
]


@pytest.mark.parametrize("nodes", CURVES)
def test_evaluate_multi_matches_the_bernstein_definition(nodes):
    curve = mbc.Curve(nodes)
    grid = np.linspace(0.0, 1.0, 257)
    np.testing.assert_allclose(curve.evaluate_multi(grid), mbc.bernstein(nodes, grid),
                               rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("nodes", CURVES)
def test_de_casteljau_and_vs_agree(nodes):
    curve = mbc.Curve(nodes)
    grid = np.linspace(0.0, 1.0, 129)
    vs = curve.evaluate_multi(grid)
    dc = curve.evaluate_multi(grid, de_casteljau=True)
    np.testing.assert_allclose(dc, vs, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(dc, mbc.bernstein(nodes, grid), rtol=1e-12, atol=1e-13)


def test_evaluate_hits_the_endpoints_exactly():
    for nodes in CURVES:
        curve = mbc.Curve(nodes)
        np.testing.assert_array_equal(curve.evaluate(0.0)[:, 0], nodes[:, 0])
        np.testing.assert_array_equal(curve.evaluate(1.0)[:, 0], nodes[:, -1])


def test_evaluate_multi_on_a_long_grid_catches_a_dropped_tail():
    curve = mbc.Curve(CURVES[0])
    grid = np.linspace(0.0, 1.0, 262_147)
    np.testing.assert_allclose(curve.evaluate_multi(grid), mbc.bernstein(CURVES[0], grid),
                               rtol=1e-11, atol=1e-12)


def test_de_casteljau_is_not_the_first_coordinate_repeated():
    """A stride mix-up in the triangle shows up as a plausible-looking answer."""
    nodes = np.asfortranarray([[0.0, 0.5, 1.0], [0.0, 1.0, 0.0]])
    curve = mbc.Curve(nodes)
    got = curve.evaluate_multi([0.0, 0.5, 1.0], de_casteljau=True)
    assert got[0].tolist() == [0.0, 0.5, 1.0]
    assert got[1].tolist() == [0.0, 0.5, 0.0]


def test_batched_evaluation_matches_per_curve_evaluation():
    family = [
        np.asfortranarray([[0.0, 0.25, 0.8, 1.0], [0.0, 1.0, -0.2, 0.5]]),
        np.asfortranarray([[0.0, 0.5, 0.5, 1.0], [0.0, 2.0, -1.0, 0.0]]),
        np.asfortranarray([[-1.0, 0.0, 1.0, 2.0], [3.0, 1.0, 1.0, -3.0]]),
    ]
    curves = mbc.Curves(np.stack(family))
    grid = np.linspace(0.0, 1.0, 61)
    batched = curves.evaluate_multi(grid)
    assert batched.shape == (len(family), 2, grid.size)
    expected = np.stack([mbc.bernstein(nodes, grid) for nodes in family])
    np.testing.assert_allclose(batched, expected, rtol=1e-12, atol=1e-13)
    for index, nodes in enumerate(family):
        np.testing.assert_allclose(
            batched[index], mbc.Curve(nodes).evaluate_multi(grid), rtol=1e-14, atol=0.0
        )


def test_batched_evaluation_keeps_each_curve_on_its_own_path():
    """Two curves that differ must not produce the same coordinates."""
    a = np.asfortranarray([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]])
    b = np.asfortranarray([[0.0, 1.0, 2.0, 3.0], [9.0, 9.0, 9.0, 9.0]])
    got = mbc.Curves(np.stack([a, b])).evaluate_multi([0.0, 0.5, 1.0])
    np.testing.assert_allclose(got[0][0], [0.0, 1.5, 3.0])
    np.testing.assert_allclose(got[0][1], np.zeros(3))
    np.testing.assert_allclose(got[1][0], [0.0, 1.5, 3.0])
    np.testing.assert_allclose(got[1][1], np.full(3, 9.0))


@pytest.mark.parametrize("nodes", CURVES)
def test_hodograph_is_the_derivative_of_the_curve(nodes):
    degree = nodes.shape[1] - 1
    curve = mbc.Curve(nodes)
    grid = np.linspace(0.0, 1.0, 101)
    forward = degree * (nodes[:, 1:] - nodes[:, :-1])
    expected = mbc.bernstein(forward, grid)
    np.testing.assert_allclose(curve.hodograph(grid), expected, rtol=1e-12, atol=1e-12)


def test_hodograph_of_a_line_is_its_direction():
    nodes = np.asfortranarray([[1.0, -2.0], [0.5, 4.0]])
    curve = mbc.Curve(nodes)
    direction = (nodes[:, 1] - nodes[:, 0]).reshape(2, 1)
    np.testing.assert_allclose(curve.hodograph([0.0, 0.25, 1.0]),
                               np.repeat(direction, 3, axis=1), rtol=0.0, atol=0.0)


def test_degree_zero_curve_is_constant():
    curve = mbc.Curve(np.asfortranarray([[3.0], [4.0]]))
    assert curve.degree == 0
    np.testing.assert_array_equal(curve.evaluate_multi([0.0, 0.5, 1.0]),
                                  np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]))
    assert curve.length() == 0.0
    with pytest.raises(ValueError):
        curve.hodograph([0.5])


def test_length_of_a_line_is_exact():
    nodes = np.asfortranarray([[0.0, 3.0], [0.0, 4.0]])
    assert mbc.Curve(nodes).length() == 5.0
    # A straight quadratic: the hodograph is constant, so Simpson is exact too.
    straight = np.asfortranarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    assert mbc.Curve(straight).length(panels=8) == pytest.approx(2 * np.sqrt(2), rel=1e-14)


def simpson_length(nodes, panels: int) -> float:
    """Composite Simpson on the hodograph norm, evaluated with the reference."""
    degree = nodes.shape[1] - 1
    forward = degree * (nodes[:, 1:] - nodes[:, :-1])
    s_vals = np.linspace(0.0, 1.0, panels + 1)
    speed = np.linalg.norm(mbc.bernstein(forward, s_vals), axis=0)
    h = 1.0 / panels
    total = speed[0] + speed[-1]
    for i in range(1, panels):
        total += (4.0 if i % 2 else 2.0) * speed[i]
    return float(total * h / 3.0)


@pytest.mark.parametrize("panels", [64, 512, 4096])
def test_length_matches_a_reference_quadrature(panels):
    nodes = np.asfortranarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    assert mbc.Curve(nodes).length(panels=panels) == pytest.approx(
        simpson_length(nodes, panels), rel=1e-12
    )


def test_length_quadrature_converges_as_panels_grow():
    nodes = np.asfortranarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    curve = mbc.Curve(nodes)
    exact = simpson_length(nodes, 1 << 16)
    coarse_error = abs(curve.length(panels=256) - exact)
    fine_error = abs(curve.length(panels=1024) - exact)
    assert fine_error < coarse_error
    assert fine_error < 1e-9


def test_curvature_matches_the_bernstein_derivatives():
    nodes = np.asfortranarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    curve = mbc.Curve(nodes)
    grid = np.linspace(0.0, 1.0, 65)
    degree = 2
    first = mbc.bernstein(degree * (nodes[:, 1:] - nodes[:, :-1]), grid)
    second = mbc.bernstein(degree * (degree - 1) * (nodes[:, 2:] - 2 * nodes[:, 1:-1]
                                                 + nodes[:, :-2]), grid)
    speed = np.linalg.norm(first, axis=0)
    expected = (first[0] * second[1] - first[1] * second[0]) / speed**3
    np.testing.assert_allclose(curve.curvature(grid), expected, rtol=1e-11, atol=1e-12)


def test_curvature_of_a_quadratic_at_its_apex_is_the_closed_form():
    """At s = 1/2 a quadratic has B' = v2 - v0 and B'' = 2 (v0 - 2 v1 + v2)."""
    nodes = np.asfortranarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    chord = nodes[:, 2] - nodes[:, 0]
    concavity = nodes[:, 0] - 2 * nodes[:, 1] + nodes[:, 2]
    cross = chord[0] * concavity[1] - chord[1] * concavity[0]
    expected = 2.0 * cross / np.linalg.norm(chord) ** 3
    assert mbc.Curve(nodes).curvature([0.5])[0] == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(np.sqrt(2.0), rel=1e-12)


def test_curvature_is_zero_for_straight_and_flat_curves():
    line = mbc.Curve(np.asfortranarray([[0.0, 1.0], [0.0, 0.0]]))
    np.testing.assert_array_equal(line.curvature([0.0, 0.5, 1.0]), np.zeros(3))
    straight = mbc.Curve(np.asfortranarray([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(straight.curvature([0.0, 0.5, 1.0]), np.zeros(3), atol=1e-15)
    # A curve that doubles back on itself has zero curvature at its cusp.
    cusp = mbc.Curve(np.asfortranarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(cusp.curvature([0.5]), np.zeros(1))


def test_curvature_changes_sign_when_the_node_order_is_reverses():
    nodes = np.asfortranarray([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    forward = mbc.Curve(nodes).curvature([0.5])[0]
    backward = mbc.Curve(np.asfortranarray(nodes[:, ::-1])).curvature([0.5])[0]
    assert forward == pytest.approx(-backward, rel=1e-12)


def test_curvature_needs_planar_curves():
    curve = mbc.Curve(np.asfortranarray([[0.0, 1.0, 2.0]]))
    with pytest.raises(ValueError):
        curve.curvature([0.5])


def test_input_validation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        mbc.Curve(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        mbc.Curve(np.asfortranarray([[0.0, np.nan]]))
    with pytest.raises(ValueError):
        mbc.Curve(np.asfortranarray([[0.0, 1.0]]), degree=7)
