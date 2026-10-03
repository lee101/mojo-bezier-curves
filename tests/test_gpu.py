"""The device path: same numbers, whole grid, and a clean fallback.

`bc_evaluate_multi_s` runs a large batch on the GPU and a small one on the
host, from the same arithmetic. These tests check that the two agree exactly,
that a grid whose size is not a multiple of the block size is fully covered,
and that a box with no device simply gets the host answer. They skip when
there is no device, which is what the host path is for.
"""

import numpy as np
import pytest

from mojo_bezier_curves import Curves, _lib

#: Outputs below which `vs_gpu` declines to launch. Anything under this is the
#: host path by construction, so the device is probed above it.
_THRESHOLD = 1 << 17


def _evaluate(nodes, grid):
    """`Curves.evaluate_multi`, reporting where it ran."""
    result = Curves(nodes).evaluate_multi(grid)
    return result, _lib.used_gpu


def _has_device():
    rng = np.random.default_rng(0)
    nodes = np.ascontiguousarray(rng.random((256, 2, 9)))
    _, used = _evaluate(nodes, np.linspace(0.0, 1.0, 2048))
    return used


requires_device = pytest.mark.skipif(
    not _has_device(), reason="no usable GPU; the host path is the answer here"
)


@requires_device
def test_large_batch_runs_on_the_device():
    rng = np.random.default_rng(1)
    nodes = np.ascontiguousarray(rng.random((256, 2, 9)))
    grid = np.linspace(0.0, 1.0, 2048)
    result, used = _evaluate(nodes, grid)
    assert used, "a batch this size should reach the device"
    assert result.shape == (256, 2, 2048)


@requires_device
def test_small_batch_stays_on_the_host():
    """A 256-point grid costs a launch more than the whole evaluation."""
    rng = np.random.default_rng(2)
    nodes = np.ascontiguousarray(rng.random((4, 2, 9)))
    _, used = _evaluate(nodes, np.linspace(0.0, 1.0, 256))
    assert not used


@requires_device
@pytest.mark.parametrize("num_vals", [2049, 4097, 8191])
def test_device_matches_host_exactly(num_vals):
    """Same arithmetic, so the two paths agree bit for bit, tail included.

    The grid sizes here are prime-ish and none is a multiple of the block
    size, so a kernel that stopped short of the end of the grid would show up
    as uninitialised output rather than as a rounding difference.
    """
    rng = np.random.default_rng(3)
    curves = (1 << 17) // (2 * num_vals) + 1
    nodes = np.ascontiguousarray(rng.random((curves, 2, 9)))
    grid = np.linspace(0.0, 1.0, num_vals)
    on_device, used = _evaluate(nodes, grid)
    assert used
    serial = _lib.evaluate_multi_vs(
        _lib.Batch(nodes), 1.0 - grid, grid
    )
    np.testing.assert_array_equal(on_device, serial)


def test_host_path_is_the_answer_without_a_device():
    """Whatever ran, the numbers are the ones upstream's helpers produce."""
    from mojo_bezier_curves import curve_helpers as mine

    rng = np.random.default_rng(4)
    nodes = np.ascontiguousarray(rng.random((64, 2, 9)))
    grid = np.linspace(0.0, 1.0, 2048)
    result, _ = _evaluate(nodes, grid)
    for index in (0, 31, 63):
        expected = mine.evaluate_multi(nodes[index], grid)
        np.testing.assert_allclose(result[index], expected, rtol=1e-12)
