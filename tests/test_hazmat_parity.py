"""Parity with the upstream `bezier` package, function by function.

Every test here compares against the same-named function in
``bezier.hazmat.curve_helpers`` (release 2024.6.20, installed in this
environment). Nothing is reimplemented on the reference side: if the two agree,
it is because the Mojo kernel computes the same thing as the Python upstream
does.
"""

import numpy as np
import pytest

from conftest import ATOL, NAMED_NODES, RTOL, curve_family
from mojo_bezier_curves import curve_helpers as mine

_PLANAR = ["planar_line", "quadratic", "cubic", "cusp"]


def _values(count=7):
    return np.linspace(0.0, 1.0, count)


@pytest.mark.parametrize("name", sorted(NAMED_NODES))
def test_evaluate_multi_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    values = _values()
    np.testing.assert_allclose(
        mine.evaluate_multi(nodes, values),
        up.evaluate_multi(nodes, values),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("degree", [1, 2, 3, 7, 15])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_evaluate_multi_matches_upstream_on_random_curves(degree, dim, up):
    values = _values(9)
    for nodes in curve_family(3, dim, degree, seed=degree + dim):
        np.testing.assert_allclose(
            mine.evaluate_multi(nodes, values),
            up.evaluate_multi(nodes, values),
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize("name", sorted(NAMED_NODES))
def test_evaluate_multi_vs_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    values = _values()
    lambda1 = 1.0 - values
    np.testing.assert_allclose(
        mine.evaluate_multi_vs(nodes, lambda1, values),
        up.evaluate_multi_vs(nodes, lambda1, values),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("name", sorted(NAMED_NODES))
def test_evaluate_multi_de_casteljau_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    values = _values()
    lambda1 = 1.0 - values
    np.testing.assert_allclose(
        mine.evaluate_multi_de_casteljau(nodes, lambda1, values),
        up.evaluate_multi_de_casteljau(nodes, lambda1, values),
        rtol=RTOL,
        atol=ATOL,
    )


def test_the_two_evaluation_algorithms_agree(up):
    """Upstream keeps both because they fail differently; so does the port."""
    for nodes in curve_family(2, 2, 12, seed=3):
        values = _values(11)
        lambda1 = 1.0 - values
        np.testing.assert_allclose(
            mine.evaluate_multi_vs(nodes, lambda1, values),
            mine.evaluate_multi_de_casteljau(nodes, lambda1, values),
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize("degree", [1, 2, 5, 10, 20])
def test_make_subdivision_matrices_matches_upstream(degree, up):
    """Every entry is a dyadic rational, so this is exact, not approximate."""
    left, right = mine.make_subdivision_matrices(degree)
    expected_left, expected_right = up.make_subdivision_matrices(degree)
    np.testing.assert_array_equal(left, expected_left)
    np.testing.assert_array_equal(right, expected_right)
    assert left.flags.f_contiguous and right.flags.f_contiguous


@pytest.mark.parametrize("degree", [1, 2, 3, 6, 11])
def test_subdivide_nodes_matches_upstream(degree, up):
    for nodes in curve_family(3, 2, degree, seed=degree):
        left, right = mine.subdivide_nodes(nodes)
        expected_left, expected_right = up.subdivide_nodes(nodes)
        np.testing.assert_allclose(left, expected_left, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(right, expected_right, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("degree", [0, 1, 2, 5, 9])
def test_elevate_nodes_matches_upstream(degree, up):
    for nodes in curve_family(3, 2, degree, seed=100 + degree):
        np.testing.assert_allclose(
            mine.elevate_nodes(nodes), up.elevate_nodes(nodes), rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("degree", [1, 2, 4, 7])
def test_de_casteljau_one_round_matches_upstream(degree, up):
    for nodes in curve_family(2, 3, degree, seed=200 + degree):
        np.testing.assert_allclose(
            mine.de_casteljau_one_round(nodes, 0.3, 0.7),
            up.de_casteljau_one_round(nodes, 0.3, 0.7),
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize(
    "start,end",
    [(0.0, 0.5), (0.5, 1.0), (0.1, 0.9), (0.25, 0.75), (-0.25, 0.75), (1.0, 3.0)],
)
@pytest.mark.parametrize("degree", [1, 2, 4, 7])
def test_specialize_curve_matches_upstream(start, end, degree, up):
    for nodes in curve_family(2, 2, degree, seed=300 + degree):
        np.testing.assert_allclose(
            mine.specialize_curve(nodes, start, end),
            up.specialize_curve(nodes, start, end),
            rtol=RTOL,
            atol=ATOL,
        )


def test_specialize_half_matches_subdivide(up):
    """Upstream's own doctest: `specialize(0, 1/2)` reproduces `subdivide()`."""
    nodes = NAMED_NODES["quadratic"]
    left, right = up.subdivide_nodes(nodes)
    np.testing.assert_array_equal(mine.specialize_curve(nodes, 0.0, 0.5), left)
    np.testing.assert_array_equal(mine.specialize_curve(nodes, 0.5, 1.0), right)


@pytest.mark.parametrize("degree", [2, 5])
def test_specialize_half_agrees_with_subdivide(degree):
    """The same identity on arbitrary nodes, where the two may differ by an ulp.

    Upstream subdivides with a matrix product and specializes with de Casteljau
    rounds; those sum the same dyadic products in a different order, so exact
    equality is a property of particular nodes, not of the identity.
    """
    for nodes in curve_family(3, 2, degree, seed=11 + degree):
        left, right = mine.subdivide_nodes(nodes)
        np.testing.assert_allclose(
            mine.specialize_curve(nodes, 0.0, 0.5), left, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            mine.specialize_curve(nodes, 0.5, 1.0), right, rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("name", ["planar_line", "quadratic", "cubic"])
@pytest.mark.parametrize("s", [0.2, 0.5, 1.0])
def test_get_curvature_matches_upstream(name, s, up):
    nodes = NAMED_NODES[name]
    tangent = mine.evaluate_hodograph(s, nodes)
    np.testing.assert_allclose(
        mine.get_curvature(nodes, tangent, s),
        up.get_curvature(nodes, tangent, s),
        rtol=RTOL,
        atol=ATOL,
    )


def test_curvature_at_a_zero_tangent_is_not_a_number_in_either(up):
    """The cusp's tangent vanishes at s = 1/2, so both sides divide 0 by 0."""
    nodes = NAMED_NODES["cusp"]
    tangent = mine.evaluate_hodograph(0.5, nodes)
    with np.errstate(invalid="ignore"):
        assert np.isnan(mine.get_curvature(nodes, tangent, 0.5))
        assert np.isnan(up.get_curvature(nodes, tangent, 0.5))


def test_vec_size_matches_upstream(up):
    for nodes in curve_family(3, 3, 6, seed=5):
        for s in (0.0, 0.35, 0.8):
            assert mine.vec_size(nodes, s) == pytest.approx(
                up.vec_size(nodes, s), rel=RTOL, abs=ATOL
            )


@pytest.mark.parametrize("degree", [1, 2, 3, 6, 9])
def test_compute_length_matches_upstream(degree, up):
    """The Simpson sum converges to QUADPACK's answer as panels grow."""
    for nodes in curve_family(2, 2, degree, seed=400 + degree):
        expected = up.compute_length(nodes)
        coarse = mine.compute_length(nodes, 512)
        fine = mine.compute_length(nodes, 16384)
        assert coarse == pytest.approx(expected, rel=1e-5)
        assert fine == pytest.approx(expected, rel=1e-9)


def test_compute_length_is_exact_for_lines(up):
    nodes = NAMED_NODES["line"]
    assert mine.compute_length(nodes) == pytest.approx(
        up.compute_length(nodes), rel=RTOL, abs=ATOL
    )


def test_newton_refine_matches_upstream(up):
    """One step, not a solve: the port must land where upstream lands."""
    for nodes in curve_family(2, 2, 4, seed=6):
        point = up.evaluate_multi(nodes, np.array([0.3]))
        for start in (0.05, 0.4, 0.8, 1.0):
            assert mine.newton_refine(nodes, point, start) == pytest.approx(
                up.newton_refine(nodes, point, start), rel=RTOL, abs=ATOL
            )


@pytest.mark.parametrize("name", ["planar_line", "quadratic", "cubic", "high_degree"])
def test_locate_point_matches_upstream(name, up):
    nodes = NAMED_NODES[name]
    for s in (0.0, 0.15, 0.5, 0.87, 1.0):
        point = up.evaluate_multi(nodes, np.array([s]))
        assert mine.locate_point(nodes, point) == pytest.approx(
            up.locate_point(nodes, point), rel=1e-9, abs=1e-9
        )


def test_locate_point_rejects_points_off_the_curve(up):
    nodes = NAMED_NODES["quadratic"]
    off_curve = np.asfortranarray([[5.0], [5.0]])
    assert mine.locate_point(nodes, off_curve) is None
    assert up.locate_point(nodes, off_curve) is None


def test_evaluate_multi_switches_algorithm_at_55_nodes(up):
    """Both sides change algorithm at the same degree, and agree either side."""
    values = _values(3)
    for degree in (54, 56):
        angles = np.linspace(0.0, degree / 20.0, degree + 1)
        nodes = np.asfortranarray(np.vstack([np.cos(angles), np.sin(angles)]))
        np.testing.assert_allclose(
            mine.evaluate_multi(nodes, values),
            up.evaluate_multi(nodes, values),
            rtol=RTOL,
            atol=ATOL,
        )


def test_a_degree_zero_curve_has_no_tangent():
    """The derivatives need a forward difference, and a single node has none."""
    nodes = np.asfortranarray([[1.0], [2.0]])
    with pytest.raises(ValueError, match="tangent"):
        mine.evaluate_hodograph(0.5, nodes)
    with pytest.raises(ValueError, match="tangent"):
        mine.get_curvature(nodes, np.asfortranarray([[1.0], [2.0]]), 0.5)
    with pytest.raises(ValueError, match="degree 1 or higher"):
        mine.newton_refine(nodes, np.asfortranarray([[1.0], [2.0]]), 0.5)


def test_a_curve_with_no_nodes_has_no_length(up):
    """Upstream's own error, raised before any kernel call."""
    empty = np.asfortranarray(np.zeros((2, 0)))
    with pytest.raises(ValueError, match="at least one node"):
        mine.compute_length(empty)
    with pytest.raises(ValueError, match="at least one node"):
        up.compute_length(empty)
