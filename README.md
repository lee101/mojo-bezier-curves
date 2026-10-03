# mojo-bezier-curves

A Mojo port of the compute core of [`bezier`](https://github.com/dhermes/bezier)
(release 2024.6.20): Bernstein evaluation by two independent algorithms, de
Casteljau subdivision and interval restriction, degree elevation, the
hodograph, signed curvature, Newton's method for point location, and arc
length.

`Curve` has the names, argument order, defaults and return shapes of
`bezier.Curve` for the covered subset, and `mojo_bezier_curves.curve_helpers`
has the names and signatures of `bezier.hazmat.curve_helpers`, so a caller
swaps the import and gets the same numbers. `Curves` is the addition: a family of
curves sharing one parameter grid, which the compiled kernels are batched over.

```python
import numpy as np
import mojo_bezier_curves as mbc

nodes = np.asfortranarray([[0.0, 0.5, 1.0], [0.0, 1.0, 0.0]])

curve = mbc.Curve.from_nodes(nodes)
curve.evaluate(0.25)                      # (2, 1): B(0.25)
curve.evaluate_multi(np.linspace(0, 1, 5))
left, right = curve.subdivide()           # de Casteljau halves
curve.specialize(0.25, 0.75).nodes        # re-parameterised onto [0, 1]
curve.elevate().nodes                     # one degree higher, same geometry
curve.length                              # arc length (a property, as upstream)
curve.locate(curve.evaluate(0.3))         # 0.3, up to Newton's last step

family = mbc.Curves(np.random.default_rng(0).random((512, 2, 9)))
family.evaluate_multi(np.linspace(0, 1, 1024))   # 512 curves, one call
family.length(panels=4096)                       # 512 arc lengths, one call
```

The first block prints `<Curve (degree=2, dimension=2)>` for `repr(curve)`, and
`0.2999999999999201` for the `locate`, which is `0.3` to the last place the
subdividing and the Newton step can be expected to carry.

## Install

```bash
pixi install
pixi run build      # -> dist/libmojo-bezier-curves.so
pixi run test
pixi run bench
```

The example above runs as-is under `pixi run python your_script.py`, or under
`pixi shell`; `PYTHONPATH=python` is part of the task activation, so nothing
needs installing to use the package from inside the workspace. Outside a task,
set `PYTHONPATH` yourself, or install the package into an environment that has
the shared library built. There is no wheel on PyPI; this is a workspace, not a
published distribution.
The environment is pinned to Python 3.12 because the upstream `bezier` package
ships `cp312` wheels and is the parity reference; `bezier` itself is declared as
a PyPI dependency of the workspace, so `pixi install` fetches the exact wheel
recorded in `pixi.lock`. The environment also pins `max`, which is where the GPU
host API lives now (`max.gpu.host`); it is released in lockstep with `mojo`, and
the large-batch evaluation path uses it.

`MOJO_NOTES.md` records the dialect facts of that exact compiler -- the
`max.gpu.host` move, `simd_width_of` from `std.sys`, `@export` needing an
explicit `abi("C")` -- each one checked by compiling it. It is worth reading
before changing anything in `src/ported.mojo`.

## What is ported

Every function keeps upstream's name, argument order and defaults, and the
kernels in `src/ported.mojo` are emitted in the source order of
`bezier/hazmat/curve_helpers.py`, so the two files can be read side by side.

| upstream `bezier.hazmat.curve_helpers` | here | notes |
| --- | --- | --- |
| `make_subdivision_matrices(degree)` | same | exact, every entry is dyadic |
| `subdivide_nodes(nodes)` | same | `nodes @ matrix`, batched |
| `evaluate_multi_vs(nodes, lambda1, lambda2)` | same | VS / modified Horner, batched |
| `evaluate_multi_de_casteljau(nodes, lambda1, lambda2)` | same | one triangle per (curve, s); a one-node curve raises here, as it does upstream |
| `evaluate_multi(nodes, s_vals)` | same | calls the dispatcher |
| `vec_size(nodes, s_val)` | same | the norm of `B(s)` |
| `compute_length(nodes)` | `compute_length(nodes, panels=1024)` | Simpson, not QUADPACK |
| `elevate_nodes(nodes)` | same | |
| `de_casteljau_one_round(nodes, lambda1, lambda2)` | same | |
| `specialize_curve(nodes, start, end)` | same | |
| `evaluate_hodograph(s, nodes)` | same | note the argument order |
| `get_curvature(nodes, tangent_vec, s)` | same | planar, as the cross product is |
| `newton_refine(nodes, point, s)` | same | one step, not a solve |
| `locate_point(nodes, point)` | same | candidates subdivide in one call |

`Curve` covers `from_nodes`, `nodes`, `degree`, `dimension`, `length`, `copy`,
`evaluate`, `evaluate_multi`, `evaluate_hodograph`, `subdivide`, `elevate`,
`specialize` and `locate`, with upstream's `__init__(nodes, degree, *, copy,
verify)` signature and upstream's `repr`.

### Divergences, all of them deliberate

* **`compute_length` takes a `panels` argument.** Upstream hands the integral to
  QUADPACK through SciPy. A kernel cannot call an adaptive Python integrator, so
  the port integrates with composite Simpson over `panels` subintervals, rounded
  up to an even count. Accuracy is a parameter, not a claim: the tests pin
  convergence to QUADPACK at two panel counts rather than asserting a magic
  tolerance. Lines are exact, as upstream returns them, and a single node is 0.0.
  The panel count is also the kernel's loop trip count, so
  `_lib.compute_length` rejects anything outside
  `[1, _lib.MAX_LENGTH_PANELS]` rather than handing an unbounded loop to a call
  that cannot be interrupted.
* **Everything is batched over a leading curve index.** Upstream evaluates one
  curve per call; a single curve here is the same call with `num_curves == 1`.
  `Curve` and `curve_helpers` hide the extra index.
* **A one-node curve has no de Casteljau triangle.** Upstream raises
  `IndexError` from that shape; `curve_helpers.evaluate_multi_de_casteljau`
  raises `ValueError` before the call, and the kernel returns the node itself
  rather than reading its scratch buffer to find out.
* **`specialize_curve` walks a row instead of building a table.** Upstream keys
  partial values by which weight pair produced them and reads the diagonal
  `(0, ..., 0, 1, ..., 1)` off the resulting table. That table is two de
  Casteljau triangles, and the diagonal is walked here one row at a time: the
  same recurrence, the same order of operations, no table.
* **`subdivide_nodes` always builds the subdivision matrices.** Upstream ships
  hard-coded tables for degrees 1, 2 and 3 and builds them above that; the
  entries are the same dyadic numbers either way.
* **`locate_point` subdivides all surviving candidates in one kernel call**
  rather than one call per candidate. The candidate set, the bounding-box test,
  the iteration count (`_MAX_LOCATE_SUBDIVISIONS`), the standard-deviation cap
  and the single Newton step are unchanged.
* **Buffers cross the C ABI as `Int` addresses** and are rebuilt in Mojo as
  `Pointer[Float64, AnyOrigin[mut=True]]`, because `@export` rejects parametric
  functions. The upstream array layouts (`(dimension, num_nodes)` Fortran order
  in, `(dimension, num_vals)` Fortran order out) are preserved; the reshape
  happens in the Python layer, not in the kernel.
* **Evaluation has a second kernel entry point, one parameter grid wide.**
  `evaluate_multi` knows the first barycentric weight is `1 - s`, so
  `bc_evaluate_multi_s` is handed `s` alone and forms `1 - s` in registers.
  The arithmetic, the order of it and the output are the same as
  `bc_evaluate_multi_vs` with both grids; what goes away is a NumPy temporary,
  a pass over it and a second address through the FFI on every call.
* **The evaluation result buffer carries `degree + 1` spare doubles** past the
  output, which the kernels use for the binomial row of the VS recurrence.
  The returned array is a prefix view of that buffer, so the cost is one
  allocation either way and no scratch pointer crosses the FFI.
* **A batch big enough to be worth a launch runs on the GPU.** One launch, the
  data copied in and out around it, and the same scalar recurrence in the same
  scalar order, so the device result is the host result bit for bit -- not a
  faster approximation of it. It engages at 131072 outputs or more, refuses
  allocations over 2 GiB, and declines when the device has less than 4 GiB
  free, because the GPU on this box is shared with production work. Every
  other kernel, and every per-curve call, is host-only: at the sizes the
  drop-in API is used at, a launch costs more than the whole evaluation. See
  [Where the work runs](#where-the-work-runs).

### Not ported

`Triangle` and the higher-order Bezier objects, curve/triangle and
curve/curve intersection (the algebraic method needs a polynomial resultant
solver), `plot`, the SymPy symbolic output, degree *reduction* (`reduce_`, which
is a per-degree pseudo-inverse rather than a curve kernel), and the
power-polynomial conversion. Those are plotting, symbolic algebra, or a
different algorithm, not spline evaluation.

## Parity testing

The tests run against the real upstream package, installed in the workspace
environment, not against a reimplementation:

* `tests/test_hazmat_parity.py` compares every function above against the
  same-named function in `bezier.hazmat.curve_helpers`, over named curves and
  random families across degrees 0 to 20 and dimensions 1 to 3, plus the
  degree-55 dispatch boundary where both sides switch algorithm.
* `tests/test_curve_parity.py` compares `Curve` against `bezier.Curve` method by
  method, including `repr`, the return shapes and the fact that `length` is a
  property in both.
* `tests/test_batch.py` compares the batched `Curves` against upstream curve by
  curve, and against the single-curve API.
* `tests/test_gpu.py` covers the device path where there is a device: that a
  large batch reaches it, that a small one does not, and that the two agree
  bit for bit on grids whose size is not a multiple of the block size, so a
  kernel that stopped short of the end of the grid would show up as
  uninitialised output rather than as a rounding difference. It skips when
  there is no usable device, which is what the host path is for.
* `tests/test_doctest_values.py` checks the numbers upstream publishes in its own
  docstrings: the `evaluate`, `evaluate_hodograph`, `subdivide` and `specialize`
  examples, `get_curvature` at -12.0, and the documented log2 convergence rates
  of `newton_refine` on the cusp.
* `tests/test_identities.py` needs no reference at all: the Bernstein polynomial
  written out longhand with `math.comb`, the endpoints, the hodograph identity
  against forward differences, the closed-form arc length and curvature of
  `y = x^2`, subdivision composing with evaluation, degree-elevation invariance,
  and the curvature sign flip under node reversal.
* `tests/test_simd_tail.py` sweeps a parameter grid across the block boundary
  -- 0, 1, 2, 3, 4, 5 values and on up, for degrees 0 to 9 and dimensions 1 to
  3 -- against upstream, for every kernel with a vector body. The last
  `n % W` parameters go through the scalar routine; a tail that ran off the end
  of the grid, dropped a parameter, or happened to be right at the sizes the
  other tests use would fail here. The same grid lengths are swept at degrees 54
  to 70, where `evaluate_multi` dispatches to de Casteljau instead and there is
  no vector body left to get wrong.
* `tests/test_paths.py` covers the Python paths around the cached batch: that a
  `Curve` or `Curves` with its nodes resolved at construction answers exactly
  as the uncached helpers do, that a second call does not read a stale address,
  that `copy=False` still adopts the caller's array, and that the finiteness
  check still fires on the paths a caller can get wrong.
* `tests/test_buffers.py` over-allocates every scratch buffer by 32 doubles of a
  sentinel and checks them afterwards, because a Mojo kernel indexes raw
  addresses with no bounds check and an off-by-one is otherwise a heap
  corruption that surfaces somewhere unrelated. It also calls all twelve
  exported symbols directly through ctypes, so each one is observed on its own
  output rather than only through the public API that happens to reach it, and
  it runs `bc_evaluate_hodograph` twice more against buffers sized to the
  letter -- exactly `dim * degree + dim + degree` doubles, no guard words at all
  -- over every grid length around the SIMD boundary. That second form catches
  the other direction: a size formula one region short, which is what the
  hodograph's scalar tail was before it stopped evaluating in place over the
  forward differences it reads.

Tolerances are `rtol=1e-12`: Mojo contracts a multiply and an add into one FMA
where NumPy does not, so the two round differently on the same arithmetic.
Exact equality is asserted only where the operation is exact -- the
subdivision matrices, the `specialize(0, 1/2)` doctest identity, endpoint
values, two calls at the same batch size, which are the same code on different
addresses, and the device result against the host result, which is the same
recurrence in the same order. The batched and single-curve evaluations are
*not* asserted bit for bit any more: one runs `W` parameters at a time and the
other runs the tail routine, so they differ by the FMA rounding above, and
asserting otherwise would be asserting an accident of code generation.

## Benchmarks

Best of 5 runs in one process, upstream timed first, every case checked against
upstream before timing. The two tables below are the output of one
`pixi run bench` on the machine below; nothing here is carried over from an
earlier run.

* Intel Xeon E5-2697 v4 @ 2.30 GHz, 2 sockets, 18 cores / 36 threads each,
  90 MiB L3 per socket
* Linux 6.8.0, Python 3.12.14, NumPy 2.5.3, upstream `bezier` 2024.6.20
* Mojo 1.2.0.dev2026092905, host kernels single-threaded and SIMD, an RTX 5090
  for the evaluation rows the device path takes

| case | upstream | mojo-bezier-curves | speedup |
| --- | ---: | ---: | ---: |
| evaluate, 1 curve x 65536 s (deg 8) | 6.12 ms | 0.96 ms | 6.39x |
| evaluate, 256 curves x 2048 s (deg 8), 1 call vs 256 | 85.50 ms | 1.62 ms | 52.64x |
| evaluate (de Casteljau), 64 curves x 4096 s (deg 10), 1 call vs 64 | 2545.60 ms | 25.27 ms | 100.75x |
| hodograph, 64 curves x 4096 s (deg 8), 1 call vs 64 | 28.70 ms | 1.36 ms | 21.17x |
| curvature, 64 curves x 4096 s (deg 8), 1 call vs 64 | 68.55 ms | 2.80 ms | 24.49x |
| newton refine, 64 curves x 1024 points (deg 8), 1 call vs 65536 | 11961.99 ms | 5.07 ms | 2359.77x |
| subdivide, 512 curves (deg 10), 1 call vs 512 | 23.13 ms | 0.22 ms | 104.85x |
| specialize, 512 curves (deg 10), 1 call vs 512 | 161.18 ms | 0.42 ms | 387.87x |
| elevate, 4096 curves (deg 8), 1 call vs 4096 | 49.60 ms | 0.24 ms | 210.30x |
| length, 64 curves (deg 6), 1 call vs 64 | 1212.56 ms | 7.47 ms | 162.24x |
| evaluate via Curve objects, 2048 x 256 s (deg 6) | 19.80 ms | 30.95 ms | **0.64x, slower** |
| evaluate via curve_helpers (pure Python), 2048 x 256 s | 208.13 ms | 46.45 ms | 4.48x |

11 of 12 cases at or above upstream parity. The first two rows are the two the
device path takes, and the two whose numbers depend on the box's other tenant.
The guard described below catches a device with no memory free, not a device
with work queued on it, and there is no cheap way to tell those apart from
inside the library. If a caller cannot tolerate that spread, the host path is
one constant away: `GPU_MIN_OUTPUTS` in `src/ported.mojo`.

Read the loss as a loss. The eleventh row is the drop-in object path, 2048 tiny
`Curve.evaluate_multi` calls: `bezier.Curve` dispatches to a compiled Cython
backend, so that row measures a compiled object against a ctypes call across
Python, and per-call marshalling -- not the kernel -- is what it measures. The
answer to it is still not a faster kernel, it is not making 2048 calls, which
is what `Curves` is for; the arithmetic behind both is the same. The twelfth row
is the same work against upstream's pure Python `curve_helpers`, which is the
like-for-like comparison, and the port is ahead there.

The large ratios on the family rows are mostly the batching, not the
arithmetic: upstream has no batched API, so "1 call vs 256" is one call against
256 Python calls of the same work. The row to read for kernel quality is the
first one, where both sides make exactly one call. `length` gains most of its
ratio because upstream's QUADPACK is an adaptive Python integrator and this is
a fixed-panel Simpson sum, which is a different amount of work as much as a
faster one.

Upstream's own numbers move by tens of percent between runs on a shared box --
across the five runs taken while writing this README the second row's 256
upstream calls ranged from 75.00 ms to 86.76 ms and the de Casteljau row from
2315.47 ms to 2756.11 ms -- so the ratio is the headline and the absolute
columns are one run. Reproduce with `pixi run bench`, which
holds a machine-wide lock so a concurrent job on the box cannot distort the
numbers.

### Where the device starts to pay

Also printed by `pixi run bench`, from the same run, comparing
`bc_evaluate_multi_s` on the device against `bc_evaluate_multi_vs` on the host
over the same batches. The two are the same recurrence in the same order, and
the bench compares them exactly before it times either.

| outputs (curves x s) | device | host SIMD | device/host |
| ---: | ---: | ---: | ---: |
| 262144 (1 x 131072 s) | 0.675 ms | 0.762 ms | 0.89 |
| 131072 (8 x 8192 s) | 0.363 ms | 0.366 ms | 0.99 |
| 524288 (32 x 8192 s) | 0.873 ms | 1.443 ms | 0.61 |
| 1048576 (256 x 2048 s) | 1.760 ms | 3.169 ms | 0.56 |
| 16777216 (1024 x 8192 s) | 68.603 ms | 87.148 ms | 0.79 |

A ratio below 1 means the device was faster. In this run it was ahead at every
size, but only just at the smallest: 1.12x at 262144 outputs and 1.01x at
131072, then 1.6x at 524288, 1.8x at 1048576 and 1.3x on the largest, where the
copy in and out is a large enough share of the total that the advantage shrinks.
Every device row moves by tens of percent between runs, because the GPU on this
box is shared: across the five runs taken while writing this README the smallest
batch ranged from 1.01x to 1.25x, and one put it level with the host.

## What the optimisation changed

The targets were the rows at or near parity: evaluation, the hodograph,
curvature, and the per-call path behind the two `evaluate` rows. The rows
already far ahead -- de Casteljau, Newton, subdivision, elevation, arc length --
were left alone.

**Vectorised (kept).** The VS recurrence runs `W` parameters at a time
(`W = simd_width_of[float64]()`), which is the contiguous index in every batched
layout, with the accumulator and the `lambda2` power living in registers for the
whole sweep, one contiguous store per block instead of a load and a store per
node, and the binomial coefficients read from a table rather than divided per
block. The leftover `num_vals % W` parameters go through the scalar routine.
Applied to `bc_evaluate_multi_vs`, `bc_evaluate_multi_s`,
`bc_evaluate_hodograph` and `bc_get_curvature`; the curvature path also keeps
the evaluated concavity in registers instead of writing it back.
`tests/test_simd_tail.py` is what pins the tail.

**Removed redundant per-call work (kept).** `Curve` and `Curves` resolve their
`(curves, dim, num_nodes)` batch and its address once, at construction
(`_lib.Batch`), rather than re-checking the finiteness of every node and
rebuilding a ctypes object on every call; and `1.0 - s` is formed in the kernel
instead of in a NumPy temporary that costs an allocation, a pass and an address
per call. On a 256-point grid that per-call work costs more than the evaluation
it feeds, which is what the eleventh row of the benchmark is measuring.

**GPU for large batches (kept, at `GPU_MIN_OUTPUTS` outputs and up).** One launch
per call, data copied in and out around it, the grid covering every output with
the threads past the end returning. The device table above is printed by
`pixi run bench` itself, from a run on this machine; the two paths are the same
recurrence in the same order, and the bench compares them exactly before it
times either.

The threshold sits where the device is worth a launch at all, which is also why
the drop-in per-curve row does not use it: at 256 points a launch costs several
times the whole evaluation. Allocations are capped at 2 GiB, and the device is
left alone below 4 GiB free -- so on a box where the GPU is held by another
tenant, `tests/test_gpu.py` skips and the benchmark prints the host path
instead, which is the honest reading of what ran. A device that is busy but has
memory free is not caught by that guard.

**Parallelism (tried, reverted).** `max.algorithm.parallelize` compiles on this
toolchain and works from a ctypes host -- a lambda over a top-level function with
the addresses captured is the form that type-checks; a closure over a local
pointer does not. Splitting the batch by curve, and a lone curve by chunks of
its grid, measured worse than the serial kernels at every size measured on this
box, whose cores are oversubscribed by other jobs: the launch cost is the same
whatever the batch. The parallel path was removed and the kernels are
single-threaded.

**Register-resident scalar recurrence (tried, reverted).** Keeping the
accumulator in a register instead of writing and reading it back per node is
what the vectorised path does, but applied to `vs_recurrence` itself it measured
worse on arc length, Newton and the single-curve column, because the coefficient
and the power then have to be carried once per coordinate instead of once per
sweep.

**Writing the single-curve result in Fortran order (tried, reverted).** The
upstream API returns `(dimension, num_vals)` Fortran-ordered, which is
`stride_m == dim` -- a strided store, so no vector store, and the scalar tail
routine per parameter. Measured against running the vectorised kernel and
transposing the result, it lost at every size and degree measured, so the strided
path was removed and the transposing copy stayed.

Those three were measured against the serial kernels on this box and are
recorded here because the decision is not visible in the source; the numbers
themselves are not reproduced, because the code they measured is not in the
tree. What is left is the SIMD path, the cheaper per-call path, and the device
path for batches large enough to want one.

## Where the work runs

| work | where | when |
| --- | --- | --- |
| every other kernel | host, one thread | always |
| batched evaluation, fewer than 131072 outputs | host, one thread, SIMD | always |
| batched evaluation, 131072 outputs or more | GPU, one launch, two copies | device present, buffers under 2 GiB, over 4 GiB free |
| `Curve.evaluate_multi` | host | always: 256 points is far below the launch cost |

`mojo_bezier_curves._lib.used_gpu` says whether the last `evaluate_multi_s` ran
on the device, which is how the table above and the tests tell a device run
from a host fallback.

The device rows carry the risk that comes with a shared GPU: memory being free
is not the same as the device being idle, and a launch behind someone else's
work costs more than the host path it replaced. The free-memory guard catches
the first case and not the second.

## How it works

**FFI.** All kernels live in `src/ported.mojo`, one compilation unit, and
`build/build.sh` compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-bezier-curves.so`. Every exported symbol takes buffer addresses as
plain `Int` values and rebuilds the pointer inside the body:

```mojo
@export("bc_evaluate_multi_vs")
def bc_evaluate_multi_vs(
    nodes_addr: Int, lambda1_addr: Int, lambda2_addr: Int, num_curves: Int,
    dim: Int, degree: Int, num_vals: Int, dst_addr: Int
) abi("C"):
```

`@export` rejects parametric functions, so an inferred pointer origin would make
the symbol parametric; the origin is annotated and the addresses are `Int`
because the C ABI has no pointer type here. `python/mojo_bezier_curves/_lib.py`
declares the argtypes (addresses as `c_int64`, which matters: `c_int` truncates
them and segfaults), owns every array, and hands the kernel a 64-bit address.
`bc_evaluate_multi_s` is the second evaluation entry point -- one parameter
grid, `1 - s` formed in the kernel -- and returns an `Int` saying whether the
device took it, which is what `used_gpu` reports.

**Per-call cost.** A `Curve` and a `Curves` hold a `_lib.Batch`: the validated
`(curves, dim, num_nodes)` array and the address of it, both resolved once.
`int(array.ctypes.data)` builds a fresh ctypes object each time and `as_batch`
re-checks the finiteness of every node; on a small grid both cost more than the
evaluation they feed, which is what the eleventh row of the benchmark is
measuring. `Batch.trusted` is the version for arrays this package has just
produced, where the check would be a pass over a whole family for numbers a
kernel wrote a moment earlier.

**Memory layout.** A family of curves is `(num_curves, dim, num_nodes)`
C-contiguous float64, so element `(d, j)` of curve `b` is at
`b * dim * width + d * width + j`. An evaluation over a parameter grid is
`(num_curves, dim, num_vals)` with element `(d, m)` at
`b * dim * num_vals + d * num_vals + m`. The coordinate dimension comes before
the varying one in both, so those two strides are the whole layout; getting them
the wrong way round is transposed output rather than garbage, which is why the
tests check coordinates individually as well as in bulk.

Scratch space arrives as one buffer addressed by explicit index offsets.
`Pointer.__add__` is deprecated in this dialect, and a named offset per region
makes an overlap between two live regions a compile-time-visible fact instead of
heap corruption -- `bc_newton_refine` needs the hodograph nodes, the evaluated
tangent and the evaluated point to be three separate regions, and the hodograph
kernel needs the forward differences and the accumulator of the scalar tail to
be two more. `tests/test_buffers.py` is what proves they are.

The evaluation result buffer is allocated with `degree + 1` spare doubles
after the output, which is where the VS recurrence keeps its binomial row; the
returned array is a prefix view of that buffer. A second scratch pointer would
have meant a second address per call, which is exactly what the per-call path
cannot afford.

The Python layer is where the layouts meet: `curve_helpers` accepts upstream's
Fortran-ordered `(dimension, num_nodes)` arrays, hands the kernel the
C-ordered batch, and returns upstream's shapes back.

## License

MIT, Lee Penkman. See [LICENSE](LICENSE).
