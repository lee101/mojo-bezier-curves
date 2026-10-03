"""The Python paths: the cached batch, the shapes it accepts, what it refuses.

`Curve` and `Curves` resolve their node buffer and its address once, at
construction, and every call after that is the kernel. These tests pin what
that caching may and may not change: the answers stay the same as the uncached
helpers, a second call must not read a stale address, a family built without a
copy still sees the array it was handed, and the finiteness check still fires
on the paths where the caller can get it wrong.
"""

import numpy as np
import pytest

from conftest import ATOL, RTOL, curve_family
from mojo_bezier_curves import Curve, Curves, _lib
from mojo_bezier_curves import curve_helpers as mine


def _nodes(dim=2, degree=5, seed=0):
    return np.ascontiguousarray(
        np.stack(curve_family(1, dim, degree, seed=seed))[0]
    )


def test_cached_batch_gives_the_helper_answer():
    nodes = _nodes()
    values = np.linspace(0.0, 1.0, 9)
    curve = Curve(nodes, nodes.shape[1] - 1)
    np.testing.assert_allclose(
        curve.evaluate_multi(values),
        mine.evaluate_multi(nodes, values),
        rtol=RTOL,
        atol=ATOL,
    )
    # Twice, because the second call is the one that would read a stale address.
    np.testing.assert_allclose(
        curve.evaluate_multi(values),
        curve.evaluate_multi(values),
        rtol=RTOL,
        atol=ATOL,
    )


def test_family_batch_gives_the_helper_answer():
    nodes = np.ascontiguousarray(np.stack(curve_family(4, 2, 5, seed=3)))
    values = np.linspace(0.0, 1.0, 11)
    result = Curves(nodes).evaluate_multi(values)
    for index, curve_nodes in enumerate(nodes):
        np.testing.assert_allclose(
            result[index],
            mine.evaluate_multi(curve_nodes, values),
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize(
    "make_values",
    [
        lambda v: list(v),
        lambda v: v,
        lambda v: np.asfortranarray(v),
        lambda v: np.ascontiguousarray(v[::-1])[::-1],
        lambda v: np.asfortranarray(v[np.newaxis])[0],
    ],
    ids=["list", "contiguous", "fortran", "reversed-view", "fortran-2d-row"],
)
def test_curve_evaluate_multi_takes_any_continuous_grid(make_values):
    nodes = _nodes()
    values = np.linspace(0.0, 1.0, 6)
    np.testing.assert_allclose(
        Curve(nodes, nodes.shape[1] - 1).evaluate_multi(make_values(values)),
        mine.evaluate_multi(nodes, values),
        rtol=RTOL,
        atol=ATOL,
    )


def test_curve_evaluate_still_returns_one_column():
    nodes = _nodes()
    curve = Curve(nodes, nodes.shape[1] - 1)
    result = curve.evaluate(0.25)
    assert result.shape == (2, 1)
    np.testing.assert_allclose(
        result, curve.evaluate_multi(np.array([0.25])), rtol=RTOL, atol=ATOL
    )


def test_a_family_without_a_copy_still_sees_its_array():
    """`copy=False` adopts the caller's array, as it always did."""
    nodes = np.ascontiguousarray(np.stack(curve_family(3, 2, 4, seed=1)))
    values = np.linspace(0.0, 1.0, 5)
    family = Curves(nodes, copy=False)
    before = family.evaluate_multi(values).copy()
    np.testing.assert_allclose(
        before[0], mine.evaluate_multi(nodes[0], values), rtol=RTOL, atol=ATOL
    )
    nodes[0] += 1.0
    after = family.evaluate_multi(values)
    assert not np.allclose(after[0], before[0])
    np.testing.assert_allclose(
        after[0], mine.evaluate_multi(nodes[0], values), rtol=RTOL, atol=ATOL
    )


def test_a_curve_without_a_copy_still_sees_its_array():
    nodes = _nodes(seed=2)
    values = np.linspace(0.0, 1.0, 5)
    curve = Curve(nodes, nodes.shape[1] - 1, copy=False)
    before = curve.evaluate_multi(values).copy()
    nodes += 1.0
    after = curve.evaluate_multi(values)
    assert not np.allclose(after, before)
    np.testing.assert_allclose(
        after, mine.evaluate_multi(nodes, values), rtol=RTOL, atol=ATOL
    )


def test_non_finite_nodes_are_still_refused_where_they_can_arrive():
    bad = _nodes(seed=6) * np.nan
    with pytest.raises(ValueError, match="finite"):
        Curve(bad, bad.shape[1] - 1)
    with pytest.raises(ValueError, match="finite"):
        Curves(bad[np.newaxis])
    with pytest.raises(ValueError, match="finite"):
        mine.evaluate_multi(bad, np.linspace(0.0, 1.0, 4))
    with pytest.raises(ValueError, match="finite"):
        _lib.Batch(bad[np.newaxis])


def test_batch_exposes_the_array_and_the_address_of_that_same_array():
    nodes = _nodes(seed=4)
    batch = _lib.Batch(nodes)
    assert batch.array.ctypes.data == batch.address
    assert batch.array.shape == nodes[np.newaxis].shape
    assert batch.degree == nodes.shape[1] - 1
    assert batch.num_curves == 1 and batch.dim == nodes.shape[0]


def test_trusted_batch_skips_the_finiteness_pass_not_the_shape():
    nodes = _nodes(seed=5)
    trusted = _lib.Batch.trusted(np.asfortranarray(nodes))
    assert trusted.array.flags.c_contiguous
    values = np.linspace(0.0, 1.0, 7)
    np.testing.assert_allclose(
        _lib.evaluate_multi_s(trusted, values)[0],
        mine.evaluate_multi(nodes, values),
        rtol=RTOL,
        atol=ATOL,
    )
    with pytest.raises(ValueError, match="rank"):
        _lib.Batch.trusted(np.arange(4.0))


def test_length_panels_is_bounded():
    """`panels` is the kernel's loop trip count, so it is checked, not trusted."""
    nodes = np.asfortranarray(_nodes(seed=7))
    for bad in (0, -1, _lib.MAX_LENGTH_PANELS + 1):
        with pytest.raises(ValueError, match="panels"):
            mine.compute_length(nodes, bad)
    # Simpson converges: 1024 panels against 65536, four orders of h apart.
    assert mine.compute_length(nodes, 1024) == pytest.approx(
        mine.compute_length(nodes, 65536), rel=1e-6
    )


def test_de_casteljau_refuses_a_curve_with_no_triangle():
    """A degree 0 curve has nothing to contract; upstream raises here too."""
    nodes = np.asfortranarray([[1.0], [2.0]])
    values = np.linspace(0.0, 1.0, 3)
    with pytest.raises(ValueError, match="degree 1 or higher"):
        mine.evaluate_multi_de_casteljau(nodes, 1.0 - values, values)
    # The kernel itself must not read `work` to find that out: it returns the
    # node, which is the whole of a degree 0 curve.
    np.testing.assert_array_equal(
        _lib.evaluate_multi_de_casteljau(nodes[np.newaxis], 1.0 - values, values),
        np.broadcast_to(nodes[np.newaxis], (1, 2, 3)),
    )


def test_elevate_never_divides_an_unwritten_node(up):
    """The boundary nodes are copies, not weighted sums.

    They are written before anything divides, so a destination poisoned with a
    sentinel cannot reach the answer. Before that was true the two boundary
    slots were divided on the way through, which read the caller's buffer.
    """
    dim, degree = 3, 6
    nodes = np.ascontiguousarray(np.random.default_rng(8).random((1, dim, degree + 1)))
    expected = _lib.elevate_nodes(nodes)
    poisoned = np.full((1, dim, degree + 2), -1.0)
    _lib.lib.bc_elevate_nodes(
        _lib._addr(nodes), 1, dim, degree, _lib._addr(poisoned)
    )
    np.testing.assert_array_equal(poisoned, expected)
    np.testing.assert_allclose(
        expected[0], up.elevate_nodes(np.asfortranarray(nodes[0])),
        rtol=RTOL, atol=ATOL,
    )
