# mojo-bezier-curves

`mojo-bezier-curves` is a Mojo port of the compute core of the
[`bezier`](https://pypi.org/project/bezier/) package: Bernstein evaluation by
two independent algorithms, de Casteljau subdivision and interval restriction,
degree elevation, the hodograph, signed curvature, Newton's method for point
location, and arc length by composite Simpson.

```python
import numpy as np
import mojo_bezier_curves as mbc

curve = mbc.Curve.from_nodes(np.asfortranarray([[0., .5, 1.], [0., 1., 0.]]))
curve.evaluate(0.25)                      # (2, 1): B(0.25)
curve.evaluate_multi(np.linspace(0, 1, 5))  # (2, 5)
left, right = curve.subdivide()            # de Casteljau halves
curve.specialize(0.25, 0.75).nodes         # re-parameterised onto [0, 1]
curve.elevate().nodes                      # one degree higher, same geometry
curve.length(panels=4096)                  # arc length
curve.locate(curve.evaluate(0.3)[:, 0])    # 0.3
```

## What is ported, and why

Every operation below is an inner loop over a curve's control points. The
reason the port exists is the **batched layout**: upstream evaluates one curve
per Python call, so a plot of a thousand curves, a quadrature over a thousand
curves or a Newton refinement of a thousand points costs a thousand calls.
`Curves` lays a family out as `(num_curves, dim, degree + 1)` with one shared
parameter grid, and the kernels are batched over it.

| area | implemented API | kernel |
| --- | --- | --- |
| Evaluation | `evaluate_multi`, `evaluate` | `bc_evaluate_multi` (VS / modified Horner), `bc_evaluate_de_casteljau` (classical triangle) |
| Splitting | `subdivide`, `restrict`, `specialize` | `bc_split_at` (both de Casteljau edges for two weight pairs) |
| Degree elevation | `elevate` | `bc_elevate` |
| Derivatives | `hodograph`, `curvature` | `bc_hodograph`, `bc_curvature` |
| Point location | `locate`, `newton_refine` | `bc_newton_refine` (many (curve, point) pairs per call) |
| Arc length | `length` | `bc_length` (composite Simpson on `\|B'(s)\|` over a caller-chosen panel count) |

Two evaluation algorithms are kept on purpose. They agree to rounding, they
fail differently, and a caller can take the de Casteljau triangle on a badly
conditioned control polygon. The tests compare them against each other and
against `mojo_bezier_curves.bernstein`, the Bernstein definition written out
longhand with `math.comb`.

## Not implemented

`Triangle` and higher-order Bezier objects, implicitisation, curve/triangle
intersection, curve-curve intersection (the algebraic method needs a
polynomial-resultant solver), `plot`, symbolic SymPy output, and degree
*reduction* (upstream's `reduce_` needs an over-determined pseudo-inverse per
degree, which is a small dense solve rather than a curve kernel). `specialize`
keeps the degree, as upstream's does; it is a restriction and a
re-parameterisation, not a re-fit.

`length` is a quadrature with a caller-chosen panel count, not QUADPACK. The
panel count is a parameter, so accuracy is something you choose and the tests
pin the convergence rather than a magic tolerance. Lines are exact.

## Parity testing without the upstream package

The upstream `bezier` package requires NumPy 2 and is not installed in the
shared parity environment (checked: neither the toolchain env nor the test venv
has it, and it cannot be added without NumPy 2). The tests therefore check two
independent things:

1. **An independent reference.** `mojo_bezier_curves.bernstein` evaluates
   `sum_j C(n, j) (1 - s)^(n - j) s^j v_j` with Python's `math.comb` and scalar
   powers. It shares no code with the VS recurrence or the de Casteljau
   triangle, so a mistake in either cannot hide behind the same mistake in the
   reference.
2. **Analytic identities.** Endpoints, the hodograph of a curve against the
   Bernstein form of its forward differences, the closed-form curvature of a
   quadratic at its apex, zero curvature for straight curves, exact length for
   straight segments, subdivision closure (`left(u) == B(u/2)`,
   `right(u) == B((1+u)/2)`), degree-elevation invariance, and the sign flip of
   the curvature when the node order is reversed.

Mojo emits FMA, so comparisons use `assert_allclose` with `rtol` around
`1e-11`; exact equality is asserted only where the operation is exact (index
arrays, padded outputs, the batched/per-curve node agreement, which is a
bit-for-bit copy of the same arithmetic).

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-bezier-curves.so`. Set
`PYTHONPATH=python` outside a Pixi task. With the shared toolchain:

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

## Performance

Best-of-three wall clock, same process, against the fastest reasonable NumPy
formulation of the same mathematics (vectorised over the parameter grid, not a
Python loop NumPy would never be asked to run). Every case verifies agreement
with the independent reference before timing.

| case | NumPy | mojo-bezier-curves | result |
| --- | ---: | ---: | ---: |
| subdivide 512 curves (deg 10) | 90.60 ms | 0.35 ms | 256x faster |
| newton 64 curves x 4096 points | 162397.24 ms | 692.22 ms | 235x faster |
| elevate 4096 curves (deg 8) | 1.35 ms | 0.41 ms | 3.3x faster |
| evaluate 256 curves x 2048 s (deg 8) | 13.99 ms | 36.11 ms | 0.39x, **slower** |
| evaluate one curve x 65536 s (deg 12) | 1.55 ms | 7.86 ms | 0.20x, **slower** |
| length 64 curves, 4096 panels (deg 6) | 0.10 ms | 13.52 ms | 0.01x, **slower** |

Read the losses as losses. Evaluation is the interesting one: a dense Bernstein
matrix times a control polygon is a GEMV that OpenBLAS already does at memory
speed, while the VS recurrence is a serial dependent chain that recomputes the
binomial coefficient at every step. Batching helps Mojo's call overhead but not
its arithmetic, and 0.39x means Mojo loses. Subdivision and Newton are the
opposite: both are inherently sequential (a de Casteljau round depends on the
previous one, a Newton step depends on the previous step), which is exactly
where a compiled scalar loop beats a NumPy expression that has to allocate an
array per round or per iteration -- 256x and 235x are real. `length` loses
because the reference builds the hodograph once and multiplies it by a Simpson
weight vector, while the kernel re-evaluates the hodograph at every panel to
keep the memory footprint at `O(degree)`.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit; `build/build.sh`
compiles it with `mojo build --emit shared-lib` into
`dist/libmojo-bezier-curves.so`. The `python/mojo_bezier_curves` layer owns every
array, normalises inputs to contiguous `float64` and rejects non-finite nodes,
and makes one call per operation. Buffers cross the C ABI as 64-bit addresses
and are rebuilt in Mojo as `Pointer[Float64, AnyOrigin[mut=True]]`, keeping the
exported symbols non-parametric.

Nodes are C-contiguous with the coordinate dimension before the node dimension,
so element `(d, j)` of curve `b` is at `b * dim * width + d * width + j` and
the result for parameter `m` is at `b * dim * num_s + d * num_s + m`. Both
strides are documented at the top of `kernels.mojo` because getting them the
wrong way round yields transposed output rather than garbage.

Scratch regions are passed in as one buffer and addressed by explicit index
offsets: `Pointer.__add__` is deprecated in this dialect, and, more usefully,
an explicit offset for each region makes an overlap between the hodograph
nodes, the residual and the derivative a compile-time-visible fact rather than
a heap corruption. `bc_newton_refine`'s three scratch regions are the reason
for that.

## License

MIT
