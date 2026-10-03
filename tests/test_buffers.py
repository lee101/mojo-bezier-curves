"""Every kernel writes inside the buffers it was handed, and nowhere else.

A Mojo kernel indexes raw addresses with no bounds check, so an off-by-one in
an offset is silent until it eats an unrelated allocation. Each scratch buffer
here is over-allocated by 32 doubles of a sentinel and checked afterwards, which
turns that class of bug into a test failure instead of a heap corruption that
shows up somewhere else entirely.
"""

import ctypes

import numpy as np
import pytest

from conftest import ATOL, RTOL
from mojo_bezier_curves import _lib

_SENTINEL = 1.2345678e300
_GUARD = 32


def _guarded(n):
    buffer = np.full(n + _GUARD, _SENTINEL, dtype=np.float64)
    return buffer, int(buffer.ctypes.data), n


def _check(buffer, used, label):
    leaked = np.nonzero(buffer[used:] != _SENTINEL)[0]
    assert leaked.size == 0, f"{label} wrote {leaked.size} doubles past its end"


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 8])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("curves", [1, 3])
def test_kernels_stay_inside_their_buffers(degree, dim, curves):
    lib = _lib.lib
    addr = _lib._addr
    rng = np.random.default_rng(degree * 100 + dim * 10 + curves)
    width = degree + 1
    nodes = np.ascontiguousarray(rng.random((curves, dim, width)))
    values = np.linspace(0.0, 1.0, 5)
    lambda1 = 1.0 - values
    left_matrix, right_matrix = _lib.make_subdivision_matrices(degree)

    work, work_addr, work_used = _guarded(dim * width)
    left = np.full((curves, dim, width, _GUARD), _SENTINEL)
    right = np.full((curves, dim, width, _GUARD), _SENTINEL)
    lib.bc_subdivide_nodes(
        addr(nodes), addr(left_matrix), addr(right_matrix), curves, dim, degree,
        addr(left), addr(right),
    )
    _check(left.ravel(), curves * dim * width, "bc_subdivide_nodes left")
    _check(right.ravel(), curves * dim * width, "bc_subdivide_nodes right")
    _check(work, work_used, "bc_subdivide_nodes")

    # The result layout is fixed, and the binomial row lives in `degree + 1`
    # doubles past the output.
    used = curves * dim * values.size
    out = np.full(used + degree + 1 + _GUARD, _SENTINEL)
    lib.bc_evaluate_multi_vs(
        addr(nodes), addr(lambda1), addr(values), curves, dim, degree,
        values.size, addr(out),
    )
    _check(out, used + degree + 1, "bc_evaluate_multi_vs")

    # `bc_evaluate_multi_s` forms the first weight as `1 - s` in the kernel
    # rather than reading a second grid, so it must land on the same numbers.
    from_s = np.full(used + degree + 1 + _GUARD, _SENTINEL)
    lib.bc_evaluate_multi_s(
        addr(nodes), addr(values), curves, dim, degree, values.size,
        addr(from_s),
    )
    _check(from_s, used + degree + 1, "bc_evaluate_multi_s")
    np.testing.assert_array_equal(
        from_s[:used].reshape((curves, dim, values.size)),
        out[:used].reshape((curves, dim, values.size)),
    )

    work, work_addr, work_used = _guarded(dim * max(degree, 1))
    out = np.full((curves, dim, values.size, _GUARD), _SENTINEL)
    lib.bc_evaluate_multi_de_casteljau(
        addr(nodes), addr(lambda1), addr(values), work_addr, curves, dim, degree,
        values.size, addr(out),
    )
    _check(out.ravel(), curves * dim * values.size, "bc_evaluate_multi_de_casteljau")
    _check(work, work_used, "bc_evaluate_multi_de_casteljau")

    work, work_addr, work_used = _guarded(dim * width)
    out = np.full((curves, values.size, _GUARD), _SENTINEL)
    lib.bc_vec_size(
        addr(nodes), addr(lambda1), addr(values), work_addr, curves, dim, degree,
        values.size, addr(out),
    )
    _check(out.ravel(), curves * values.size, "bc_vec_size")
    _check(work, work_used, "bc_vec_size")

    work, work_addr, work_used = _guarded((degree + dim) * width)
    out = np.full(curves + _GUARD, _SENTINEL)
    lib.bc_compute_length(addr(nodes), work_addr, curves, dim, degree, 64, addr(out))
    _check(out, curves, "bc_compute_length")
    _check(work, work_used, "bc_compute_length")

    out = np.full((curves, dim, width + 1, _GUARD), _SENTINEL)
    lib.bc_elevate_nodes(addr(nodes), curves, dim, degree, addr(out))
    _check(out.ravel(), curves * dim * (width + 1), "bc_elevate_nodes")

    if degree >= 1:
        out = np.full((curves, dim, degree, _GUARD), _SENTINEL)
        lib.bc_de_casteljau_one_round(
            addr(nodes), curves, dim, degree, ctypes.c_double(0.3),
            ctypes.c_double(0.7), addr(out),
        )
        _check(out.ravel(), curves * dim * degree, "bc_de_casteljau_one_round")

    work, work_addr, work_used = _guarded(2 * dim * width)
    out = np.full((curves, dim, width, _GUARD), _SENTINEL)
    lib.bc_specialize_curve(
        addr(nodes), work_addr, curves, dim, degree, ctypes.c_double(0.2),
        ctypes.c_double(0.8), addr(out),
    )
    _check(out.ravel(), curves * dim * width, "bc_specialize_curve")
    _check(work, work_used, "bc_specialize_curve")

    if degree >= 1:
        work, work_addr, work_used = _guarded(_hodograph_work(degree, dim))
        out = np.full((curves, dim, values.size, _GUARD), _SENTINEL)
        lib.bc_evaluate_hodograph(
            addr(nodes), addr(values), work_addr, curves, dim, degree,
            values.size, addr(out),
        )
        _check(out.ravel(), curves * dim * values.size, "bc_evaluate_hodograph")
        _check(work, work_used, "bc_evaluate_hodograph")

    if dim == 2 and degree >= 1:
        tangent = np.ascontiguousarray(rng.random((curves, 2, values.size)))
        work, work_addr, work_used = _guarded(
            2 * dim * max(degree, 1) + dim + degree
        )
        out = np.full((curves, values.size, _GUARD), _SENTINEL)
        lib.bc_get_curvature(
            addr(nodes), addr(tangent), addr(values), work_addr, curves, degree,
            values.size, addr(out),
        )
        _check(out.ravel(), curves * values.size, "bc_get_curvature")
        _check(work, work_used, "bc_get_curvature")

    if degree >= 1:
        points = np.ascontiguousarray(rng.random((curves, 4, dim)))
        starts = np.ascontiguousarray(rng.random((curves, 4)))
        work, work_addr, work_used = _guarded(dim * degree + dim * width + dim)
        out = np.full((curves, 4, _GUARD), _SENTINEL)
        lib.bc_newton_refine(
            addr(nodes), addr(points), addr(starts), work_addr, curves, 4, dim,
            degree, addr(out),
        )
        _check(out.ravel(), curves * 4, "bc_newton_refine")
        _check(work, work_used, "bc_newton_refine")


@pytest.mark.parametrize("degree", [0, 1, 4, 9])
def test_subdivision_matrices_stay_inside_their_buffers(degree):
    width = degree + 1
    left, left_addr, _ = _guarded(width * width)
    right, right_addr, _ = _guarded(width * width)
    _lib.lib.bc_make_subdivision_matrices(degree, left_addr, right_addr)
    _check(left, width * width, "bc_make_subdivision_matrices left")
    _check(right, width * width, "bc_make_subdivision_matrices right")


#: `dim * degree` forward differences, `dim` accumulator doubles for the scalar
#: tail, `degree` for the binomial row.
def _hodograph_work(degree, dim):
    return dim * degree + dim + degree


@pytest.mark.parametrize("num_s", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 17])
@pytest.mark.parametrize("degree", [1, 2, 3, 5, 8, 12])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_hodograph_work_buffer_is_exactly_what_the_kernel_uses(
    dim, degree, num_s, up
):
    """The scratch buffer sized from the kernel's own regions, tightly.

    The guard tests catch a kernel that writes past what the Python layer
    allocated; this pins the other direction, the exact count the kernel
    needs, by handing it a buffer with no guard words at all. A size formula
    that is one short -- as it was, before the scalar tail stopped running
    the recurrence in place over its own input -- is a heap overflow here
    rather than a number that happens to come out right.

    Every grid length around the SIMD block boundary is swept, because only
    the `num_s % W` leftover parameters take the scalar path.
    """
    work = np.full(_hodograph_work(degree, dim), _SENTINEL, dtype=np.float64)
    out = np.full((2, dim, num_s), _SENTINEL, dtype=np.float64)
    nodes = np.ascontiguousarray(
        np.random.default_rng(degree + dim + num_s).random((2, dim, degree + 1))
    )
    values = np.linspace(0.0, 1.0, num_s)
    _lib.lib.bc_evaluate_hodograph(
        _lib._addr(nodes), _lib._addr(values), int(work.ctypes.data), 2, dim,
        degree, num_s, _lib._addr(out),
    )
    for index, curve_nodes in enumerate(nodes):
        expected = np.column_stack(
            [
                up.evaluate_hodograph(s, np.asfortranarray(curve_nodes)).ravel()
                for s in values
            ]
        ) if num_s else np.empty((dim, 0))
        np.testing.assert_allclose(out[index], expected, rtol=RTOL, atol=ATOL)


def test_every_exported_symbol_is_asserted_on_its_own_output(up):
    """Each ctypes call above is also checked for what it wrote, not just where.

    The guard checks in this module answer "did it write outside"; they do not
    answer "did it write the right numbers". A kernel that computed garbage
    inside its own buffer and stopped short of the guard passes them. So every
    symbol called above is called again here, into plain buffers, and compared
    against upstream's own function -- one assertion per exported symbol, on
    that symbol's output rather than on the public API that reaches it.
    """
    lib, addr = _lib.lib, _lib._addr
    rng = np.random.default_rng(0)
    dim, degree, num_vals = 3, 4, 7
    width = degree + 1
    curves = 2
    nodes = np.ascontiguousarray(rng.random((curves, dim, width)))
    values = np.linspace(0.0, 0.9, num_vals)
    lambda1 = 1.0 - values
    seeds = [
        np.asfortranarray(curve) for curve in nodes
    ]
    used = set()

    def record(name, got, expected):
        used.add(name)
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64).reshape(np.shape(expected)),
            np.asarray(expected, dtype=np.float64),
            rtol=RTOL,
            atol=ATOL,
        )

    # bc_make_subdivision_matrices
    size = width * width
    left, right = np.zeros(size), np.zeros(size)
    lib.bc_make_subdivision_matrices(degree, addr(left), addr(right))
    record(
        "bc_make_subdivision_matrices",
        left.reshape((width, width)),
        up.make_subdivision_matrices(degree)[0],
    )
    record(
        "bc_make_subdivision_matrices",
        right.reshape((width, width)),
        up.make_subdivision_matrices(degree)[1],
    )

    # bc_subdivide_nodes
    out_left = np.zeros((curves, dim, width))
    out_right = np.zeros((curves, dim, width))
    left_mat, right_mat = _lib.make_subdivision_matrices(degree)
    lib.bc_subdivide_nodes(
        addr(nodes), addr(left_mat), addr(right_mat), curves, dim, degree,
        addr(out_left), addr(out_right),
    )
    record(
        "bc_subdivide_nodes", out_left,
        np.stack([up.subdivide_nodes(s)[0] for s in seeds]),
    )
    record(
        "bc_subdivide_nodes", out_right,
        np.stack([up.subdivide_nodes(s)[1] for s in seeds]),
    )

    # bc_evaluate_multi_vs. The result buffer carries `degree + 1` spare doubles
    # past the output for the VS binomial row, so the buffer is flat and the
    # output is its prefix.
    spare = degree + 1
    out = np.zeros(curves * dim * num_vals + spare)
    lib.bc_evaluate_multi_vs(
        addr(nodes), addr(lambda1), addr(values), curves, dim, degree,
        num_vals, addr(out),
    )
    record(
        "bc_evaluate_multi_vs",
        out[: curves * dim * num_vals].reshape((curves, dim, num_vals)),
        np.stack([up.evaluate_multi_vs(s, lambda1, values) for s in seeds]),
    )

    # bc_evaluate_multi_s
    out_s = np.zeros(curves * dim * num_vals + spare)
    lib.bc_evaluate_multi_s(
        addr(nodes), addr(values), curves, dim, degree, num_vals, addr(out_s)
    )
    record(
        "bc_evaluate_multi_s",
        out_s[: curves * dim * num_vals].reshape((curves, dim, num_vals)),
        np.stack([up.evaluate_multi(s, values) for s in seeds]),
    )

    # bc_evaluate_multi_de_casteljau
    out = np.zeros((curves, dim, num_vals))
    work = np.zeros(dim * max(degree, 1))
    lib.bc_evaluate_multi_de_casteljau(
        addr(nodes), addr(lambda1), addr(values), addr(work), curves, dim,
        degree, num_vals, addr(out),
    )
    record(
        "bc_evaluate_multi_de_casteljau", out,
        np.stack(
            [up.evaluate_multi_de_casteljau(s, lambda1, values) for s in seeds]
        ),
    )

    # bc_vec_size
    out = np.zeros((curves, num_vals))
    work = np.zeros(dim * width)
    lib.bc_vec_size(
        addr(nodes), addr(lambda1), addr(values), addr(work), curves, dim,
        degree, num_vals, addr(out),
    )
    record(
        "bc_vec_size", out,
        np.array([[up.vec_size(s, v) for v in values] for s in seeds]),
    )

    # bc_compute_length. Simpson is not exact on a general integrand, so this
    # is the one symbol whose direct call is compared at the panel count the
    # package uses by default rather than at 1e-12: upstream integrates with
    # QUADPACK and the kernel with a fixed-panel composite rule, and at 256
    # panels the two differ in the eighth significant figure. The call itself
    # is asserted against the same kernel through `_lib.compute_length`, so a
    # broken `bc_compute_length` still fails here; what is left is the accuracy
    # claim, which `tests/test_paths.py` and `tests/test_batch.py` pin by
    # convergence in the panel count.
    out = np.zeros(curves)
    work = np.zeros((degree + dim) * width)
    lib.bc_compute_length(
        addr(nodes), addr(work), curves, dim, degree, 256, addr(out)
    )
    np.testing.assert_allclose(
        out, _lib.compute_length(nodes, 256), rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        out, np.array([up.compute_length(s) for s in seeds]), rtol=1e-7
    )
    used.add("bc_compute_length")

    # bc_elevate_nodes
    out = np.zeros((curves, dim, width + 1))
    lib.bc_elevate_nodes(addr(nodes), curves, dim, degree, addr(out))
    record(
        "bc_elevate_nodes", out,
        np.stack([up.elevate_nodes(s) for s in seeds]),
    )

    # bc_de_casteljau_one_round
    out = np.zeros((curves, dim, degree))
    lib.bc_de_casteljau_one_round(
        addr(nodes), curves, dim, degree, ctypes.c_double(0.3),
        ctypes.c_double(0.7), addr(out),
    )
    record(
        "bc_de_casteljau_one_round", out,
        np.stack(
            [up.de_casteljau_one_round(s, 0.3, 0.7) for s in seeds]
        ),
    )

    # bc_specialize_curve
    out = np.zeros((curves, dim, width))
    work = np.zeros(2 * dim * width)
    lib.bc_specialize_curve(
        addr(nodes), addr(work), curves, dim, degree, ctypes.c_double(0.2),
        ctypes.c_double(0.8), addr(out),
    )
    record(
        "bc_specialize_curve", out,
        np.stack([up.specialize_curve(s, 0.2, 0.8) for s in seeds]),
    )

    # bc_evaluate_hodograph
    out = np.zeros((curves, dim, num_vals))
    work = np.zeros(_hodograph_work(degree, dim))
    lib.bc_evaluate_hodograph(
        addr(nodes), addr(values), addr(work), curves, dim, degree, num_vals,
        addr(out),
    )
    record(
        "bc_evaluate_hodograph", out,
        np.stack(
            [
                np.column_stack(
                    [up.evaluate_hodograph(v, s).ravel() for v in values]
                )
                for s in seeds
            ]
        ),
    )

    # bc_get_curvature, planar, from the hodograph the kernel above produced
    planar = np.ascontiguousarray(rng.random((curves, 2, width)))
    planar_values = np.linspace(0.1, 0.9, num_vals)
    tangents = np.ascontiguousarray(
        rng.random((curves, 2, num_vals))
    )
    out = np.zeros((curves, num_vals))
    work = np.zeros(2 * 2 * max(degree, 1) + 2 + degree)
    lib.bc_get_curvature(
        addr(planar), addr(tangents), addr(planar_values), addr(work), curves,
        degree, num_vals, addr(out),
    )
    record(
        "bc_get_curvature", out,
        np.array(
            [
                [
                    up.get_curvature(
                        np.asfortranarray(planar[i]), tangents[i][:, [j]],
                        planar_values[j],
                    )
                    for j in range(num_vals)
                ]
                for i in range(curves)
            ]
        ),
    )

    # bc_newton_refine
    points = np.ascontiguousarray(rng.random((curves, 3, dim)))
    starts = np.ascontiguousarray(rng.random((curves, 3)))
    out = np.zeros((curves, 3))
    work = np.zeros(dim * degree + dim * width + dim)
    lib.bc_newton_refine(
        addr(nodes), addr(points), addr(starts), addr(work), curves, 3, dim,
        degree, addr(out),
    )
    record(
        "bc_newton_refine", out,
        np.array(
            [
                [
                    up.newton_refine(
                        seeds[i], points[i, k, :].reshape(-1, 1), starts[i, k]
                    )
                    for k in range(3)
                ]
                for i in range(curves)
            ]
        ),
    )

    exported = {
        name
        for name in _lib._SIGNATURES
    }
    assert used == exported, f"no value assertion for: {sorted(exported - used)}"
