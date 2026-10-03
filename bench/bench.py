"""Benchmarks against the upstream `bezier` package.

Every case checks its answer against upstream before timing, so a fast wrong
number cannot be reported as a win. Timing is the best of `REPEATS` runs after
a warm-up, in the same process, with the upstream side timed first.

Two kinds of case, answering different questions:

  * **one curve** -- the same problem handed to both implementations once, so
    the number is kernel arithmetic against kernel arithmetic. A loss here is a
    loss.
  * **a family** -- N curves against a loop of N upstream calls, because
    upstream has no batched API and that is what the port adds. Those cases are
    labelled `N calls` rather than pretending the comparison is like for like.

Where upstream has a vectorised formulation available (`evaluate_multi`,
`evaluate_multi_de_casteljau`) the baseline uses it. Where it does not (the
scalar `evaluate_hodograph` and `get_curvature`), the baseline is the fastest
vectorised NumPy expression of the same mathematics, built from upstream's own
helpers, because timing upstream's scalar API in a Python loop would flatter the
port.

Run it through `pixi run bench`, never directly: that task holds a machine-wide
flock so a concurrent factory job cannot distort the numbers.
"""

import platform
import time

import numpy as np

import bezier
from bezier.hazmat import curve_helpers as up

import mojo_bezier_curves as mbc
from mojo_bezier_curves import Curves, _lib
from mojo_bezier_curves import curve_helpers as mine

REPEATS = 5
WARMUP = 1


def _family(count, dim, degree, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((count, dim, degree + 1)))


def _check(ours, theirs, label):
    ours = np.asarray(ours)
    theirs = np.asarray(theirs)
    if ours.shape != theirs.shape:
        raise AssertionError(
            f"{label}: shape {ours.shape} != upstream {theirs.shape}"
        )
    scale = max(float(np.max(np.abs(theirs))), 1.0)
    error = float(np.max(np.abs(ours - theirs))) / scale
    if not error < 1e-9:
        raise AssertionError(f"{label}: relative error {error:.3e}")


def _time(function):
    for _ in range(WARMUP):
        function()
    best = float("inf")
    for _ in range(REPEATS):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def _loop(function, count):
    return lambda: [function(index) for index in range(count)]


def _upstream_hodograph(nodes, values):
    """Upstream's mathematics, vectorised: `n * evaluate_multi(differences, s)`."""
    first_deriv = nodes[:, 1:] - nodes[:, :-1]
    return (nodes.shape[1] - 1) * up.evaluate_multi(first_deriv, values)


def _upstream_curvature(nodes, values):
    """Upstream's mathematics, vectorised over the parameter grid."""
    num_nodes = nodes.shape[1]
    first_deriv = nodes[:, 1:] - nodes[:, :-1]
    second_deriv = first_deriv[:, 1:] - first_deriv[:, :-1]
    tangent = _upstream_hodograph(nodes, values)
    concavity = (num_nodes - 1) * (num_nodes - 2) * up.evaluate_multi(
        second_deriv, values
    )
    speed = np.linalg.norm(tangent, axis=0)
    return (tangent[0] * concavity[1] - tangent[1] * concavity[0]) / speed**3


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------


def case_evaluate_one_curve():
    nodes = np.asfortranarray(_family(1, 2, 8, seed=1)[0])
    values = np.linspace(0.0, 1.0, 65536)
    ours = lambda: mine.evaluate_multi(nodes, values)
    theirs = lambda: up.evaluate_multi(nodes, values)
    _check(ours(), theirs(), "evaluate one curve")
    return "evaluate, 1 curve x 65536 s (deg 8)", ours, theirs


def case_evaluate_family():
    nodes = _family(256, 2, 8, seed=2)
    values = np.linspace(0.0, 1.0, 2048)
    family = Curves(nodes)
    ours = lambda: family.evaluate_multi(values)
    theirs = _loop(lambda i: up.evaluate_multi(nodes[i], values), 256)
    result = ours()
    for index in (0, 128, 255):
        _check(result[index], up.evaluate_multi(nodes[index], values), "evaluate")
    return "evaluate, 256 curves x 2048 s (deg 8), 1 call vs 256", ours, theirs


def case_evaluate_de_casteljau_family():
    nodes = _family(64, 2, 10, seed=3)
    values = np.linspace(0.0, 1.0, 4096)
    ours = lambda: _lib.evaluate_multi_de_casteljau(nodes, 1.0 - values, values)
    theirs = _loop(
        lambda i: up.evaluate_multi_de_casteljau(nodes[i], 1.0 - values, values), 64
    )
    _check(
        ours()[0],
        up.evaluate_multi_de_casteljau(nodes[0], 1.0 - values, values),
        "de casteljau",
    )
    return (
        "evaluate (de Casteljau), 64 curves x 4096 s (deg 10), 1 call vs 64",
        ours,
        theirs,
    )


def case_hodograph_family():
    nodes = _family(64, 2, 8, seed=7)
    values = np.linspace(0.0, 1.0, 4096)
    family = Curves(nodes)
    ours = lambda: family.evaluate_hodograph(values)
    theirs = _loop(lambda i: _upstream_hodograph(nodes[i], values), 64)
    for index in (0, 63):
        _check(ours()[index], _upstream_hodograph(nodes[index], values), "hodograph")
    return "hodograph, 64 curves x 4096 s (deg 8), 1 call vs 64", ours, theirs


def case_curvature_family():
    nodes = _family(64, 2, 8, seed=8)
    values = np.linspace(0.0, 1.0, 4096)
    family = Curves(nodes)
    ours = lambda: family.get_curvature(values)
    theirs = _loop(lambda i: _upstream_curvature(nodes[i], values), 64)
    for index in (0, 63):
        _check(ours()[index], _upstream_curvature(nodes[index], values), "curvature")
    return "curvature, 64 curves x 4096 s (deg 8), 1 call vs 64", ours, theirs


def case_newton_refine_family():
    nodes = _family(64, 2, 8, seed=9)
    values = np.linspace(0.0, 1.0, 1024)
    # One Newton step from 0.5 towards a point within an eighth of the parameter
    # away: a well-conditioned refinement, which is what the kernel is for.
    # Aiming at an arbitrary point on the curve makes the step chaotic and the
    # two implementations would legitimately disagree in the last digits.
    points = np.ascontiguousarray(
        Curves(nodes).evaluate_multi(0.5 + 0.25 * (values - 0.5)).transpose(0, 2, 1)
    )
    starts = np.full((64, values.size), 0.5)
    family = Curves(nodes)
    ours = lambda: family.newton_refine(points, starts)
    theirs = _loop(
        lambda i: np.array(
            [
                # Upstream wants the point as a (dimension, 1) column; handing
                # it a flat vector broadcasts to (dimension, dimension) and
                # quietly computes a different step.
                up.newton_refine(
                    nodes[i], points[i, k, :].reshape(-1, 1), starts[i, k]
                )
                for k in range(values.size)
            ]
        ),
        64,
    )
    result = ours()
    for index in (0, 63):
        for k in (0, 500, 1023):
            expected = up.newton_refine(
                nodes[index], points[index, k, :].reshape(-1, 1), 0.5
            )
            assert abs(result[index, k] - expected) <= 1e-12 * max(1.0, abs(expected))
    return "newton refine, 64 curves x 1024 points (deg 8), 1 call vs 65536", ours, theirs


def case_subdivide_family():
    nodes = _family(512, 2, 10, seed=4)
    family = Curves(nodes)
    ours = lambda: family.subdivide()
    theirs = _loop(lambda i: up.subdivide_nodes(nodes[i]), 512)
    left, right = ours()
    for index in (0, 511):
        their_left, their_right = up.subdivide_nodes(nodes[index])
        _check(left.nodes[index], their_left, "subdivide left")
        _check(right.nodes[index], their_right, "subdivide right")
    return "subdivide, 512 curves (deg 10), 1 call vs 512", ours, theirs


def case_specialize_family():
    nodes = _family(512, 2, 10, seed=5)
    family = Curves(nodes)
    ours = lambda: family.specialize(0.25, 0.75)
    theirs = _loop(lambda i: up.specialize_curve(nodes[i], 0.25, 0.75), 512)
    _check(ours()[3].nodes, up.specialize_curve(nodes[3], 0.25, 0.75), "specialize")
    return "specialize, 512 curves (deg 10), 1 call vs 512", ours, theirs


def case_elevate_family():
    nodes = _family(4096, 2, 8, seed=6)
    family = Curves(nodes)
    ours = lambda: family.elevate()
    theirs = _loop(lambda i: up.elevate_nodes(nodes[i]), 4096)
    _check(ours()[17].nodes, up.elevate_nodes(nodes[17]), "elevate")
    return "elevate, 4096 curves (deg 8), 1 call vs 4096", ours, theirs


def case_length_family():
    nodes = _family(64, 2, 6, seed=10)
    family = Curves(nodes)
    ours = lambda: family.length(panels=4096)
    theirs = _loop(lambda i: up.compute_length(nodes[i]), 64)
    result = ours()
    for index in (0, 63):
        _check(result[index], up.compute_length(nodes[index]), "length")
    return "length, 64 curves (deg 6), 1 call vs 64", ours, theirs


def case_curve_objects():
    """The drop-in path: one `Curve` object per curve, upstream against ours."""
    count = 2048
    nodes = _family(count, 2, 6, seed=11)
    values = np.linspace(0.0, 1.0, 256)
    ours_curves = [mbc.Curve(nodes[i], 6) for i in range(count)]
    their_curves = [bezier.Curve(nodes[i], degree=6) for i in range(count)]
    ours = lambda: [curve.evaluate_multi(values) for curve in ours_curves]
    theirs = lambda: [curve.evaluate_multi(values) for curve in their_curves]
    _check(ours()[3], theirs()[3], "Curve objects")
    return "evaluate via Curve objects, 2048 x 256 s (deg 6)", ours, theirs


def case_pure_python_helpers():
    """The same drop-in path, but against upstream's *Python* helpers.

    `bezier.Curve` dispatches to a compiled Cython backend, so the row above
    compares a compiled object against a ctypes call. This row compares like
    with like -- upstream's pure Python `curve_helpers.evaluate_multi` against
    this port's -- and is the honest measure of the per-call path.
    """
    count = 2048
    nodes = _family(count, 2, 6, seed=12)
    values = np.linspace(0.0, 1.0, 256)
    ours_nodes = [np.asfortranarray(curve) for curve in nodes]
    ours = lambda: [mine.evaluate_multi(curve, values) for curve in ours_nodes]
    theirs = lambda: [
        up.evaluate_multi(curve, values) for curve in ours_nodes
    ]
    _check(ours()[3], theirs()[3], "pure python helpers")
    return "evaluate via curve_helpers (pure Python), 2048 x 256 s", ours, theirs


#: Output counts for the device threshold sweep: (curves, s values) with a
#: degree-8 curve, so `curves * 2 * s` is the number of outputs the kernel
#: decides on.
DEVICE_SIZES = [
    (1, 131072),
    (8, 8192),
    (32, 8192),
    (256, 2048),
    (1024, 8192),
]

#: Below this the kernel never launches, so a device run cannot be timed.
GPU_MIN_OUTPUTS = 1 << 17


def device_sweep():
    """The device path against the host SIMD path: same batch, same arithmetic.

    `bc_evaluate_multi_s` is the entry point that decides between the two and
    reports which it took; `bc_evaluate_multi_vs` is the host path on its own,
    handed the `1 - s` grid the kernel would otherwise form in registers. Same
    recurrence, so the results are compared exactly and only the time is at
    stake. Returns `None` when no device took a batch, which is what the
    library is built to answer to.
    """
    rows = []
    for curves, s_vals in DEVICE_SIZES:
        nodes = _family(curves, 2, 8, seed=21)
        values = np.linspace(0.0, 1.0, s_vals)
        # Held in a name, never inlined into `_addr(...)`: a temporary is
        # released as soon as the inner call returns, so the kernel would go on
        # to read an address whose array is already freed.
        complement = 1.0 - values
        batch = _lib.Batch(nodes)
        size = curves * 2 * s_vals
        spare = batch.degree + 1

        def device():
            buffer = np.empty(size + spare, dtype=np.float64)
            used = _lib.lib.bc_evaluate_multi_s(
                _lib._addr(nodes), _lib._addr(values), curves, 2, batch.degree,
                s_vals, _lib._addr(buffer),
            )
            return buffer, bool(used)

        def host():
            buffer = np.empty(size + spare, dtype=np.float64)
            _lib.lib.bc_evaluate_multi_vs(
                batch.address, _lib._addr(complement), _lib._addr(values),
                curves, 2, batch.degree, s_vals, _lib._addr(buffer),
            )
            return buffer

        device_result, used = device()
        if size < GPU_MIN_OUTPUTS:
            rows.append((size, curves, s_vals, None, _time(host)))
            continue
        if not used:
            return None
        # Only the result itself: the host path also writes the VS binomial row
        # into the spare doubles past it, and the device path does not.
        np.testing.assert_array_equal(device_result[:size], host()[:size])
        rows.append((size, curves, s_vals, _time(device), _time(host)))
    return rows


def print_device_table():
    """Print the device threshold sweep, or say why it did not run."""
    rows = device_sweep()
    print()
    if rows is None:
        print("no usable device on this box; every evaluation took the host path.")
        return
    print("| outputs (curves x s) | device | host SIMD | device/host |")
    print("| ---: | ---: | ---: | ---: |")
    for size, curves, s_vals, device_seconds, host_seconds in rows:
        label = f"{size} ({curves} x {s_vals} s)"
        if device_seconds is None:
            print(
                f"| {label} | not launched, under the {GPU_MIN_OUTPUTS} "
                f"output threshold | {host_seconds * 1e3:.3f} ms | host only |"
            )
            continue
        print(
            f"| {label} | {device_seconds * 1e3:.3f} ms "
            f"| {host_seconds * 1e3:.3f} ms "
            f"| {device_seconds / host_seconds:.2f} |"
        )


CASES = [
    case_evaluate_one_curve,
    case_evaluate_family,
    case_evaluate_de_casteljau_family,
    case_hodograph_family,
    case_curvature_family,
    case_newton_refine_family,
    case_subdivide_family,
    case_specialize_family,
    case_elevate_family,
    case_length_family,
    case_curve_objects,
    case_pure_python_helpers,
]


def main():
    print("mojo-bezier-curves against upstream bezier\n")
    print(f"python      {platform.python_version()} on {platform.machine()}")
    print(f"numpy       {np.__version__}")
    print(f"upstream    bezier {bezier.__version__}")
    print(f"timing      best of {REPEATS} runs\n")
    print("| case | upstream | mojo-bezier-curves | speedup |")
    print("| --- | ---: | ---: | ---: |")
    rows = []
    for case in CASES:
        label, ours, theirs = case()
        upstream_seconds = _time(theirs)
        ours_seconds = _time(ours)
        ratio = upstream_seconds / ours_seconds
        rows.append(ratio)
        verdict = f"{ratio:.2f}x"
        if ratio < 1.0:
            verdict = f"**{ratio:.2f}x, slower**"
        print(
            f"| {label} | {upstream_seconds * 1e3:.2f} ms "
            f"| {ours_seconds * 1e3:.2f} ms | {verdict} |"
        )
    wins = sum(1 for ratio in rows if ratio >= 1.0)
    print(f"\n{wins} of {len(rows)} cases at or above upstream parity.")
    print_device_table()


if __name__ == "__main__":
    main()
