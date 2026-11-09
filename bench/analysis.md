# Performance analysis — mojo-bezier-curves

ANALYZE phase only. No algorithm was changed; nothing under `src/` or `tests/`
was touched. Every number below comes from a run in this session: `pixi run
bench` (flock-wrapped), the repo's `bench/profile_curve_path.py`, and targeted
timing probes run under the same machine-wide lock. Times are best-of-5/7 after
warm-up, matching `bench/bench.py`.

Machine (this run): Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 threads, Linux 6.8.0,
Python 3.12.14, NumPy 2.5.3, upstream `bezier` 2024.6.20, Mojo
1.2.0.dev2026092905, shared RTX 5090 (32 GB, ~7.6 GB free at probe time). DRAM
streaming floor measured this session with a NumPy 8 MB copy: **27.8 GB/s**.

## Ranked hotspot table

Sizes are the ones `bench/bench.py` and the README examples use (the sizes
real callers pass); the per-call rows at the bottom are the single-curve
`curve_helpers` sizes. Bytes touched counts the buffers the kernel reads and
writes (nodes, grids, output) in float64; cache-resident re-reads (nodes and
grids stay in L1/L2 at these sizes) are not counted as DRAM traffic. Flops are
counted from the recurrences in `src/ported.mojo` (`vs_recurrence`: 1 mul at
j=0, 3 mul + 1 add per inner step, 2 mul + 1 add at the last node, plus the
amortised binomial/power updates). "Glue or kernel" is the split between
Python/ctypes work and the Mojo kernel, measured by comparing each wrapper to
the raw `lib.bc_*` call with a pre-resolved batch and a reused output buffer.

| rank | function (entry point) | input size | time (ours) | bytes touched | flops | flops/byte | memory- or compute-bound | Python/ctypes glue or kernel |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 | `evaluate_multi_de_casteljau` batched (`_lib`, 64 curves × 4096 s, deg 10) | 262,144 s-values × 64 curves = 16.8 M outputs | 26.0 ms | ~134 MB (16.8 M outputs × 8 B; nodes + 2 grids ~4 MB) | ~5,500 M (330/output: `dim·n(n+1)/2` triangle entries × 3 flops) | ~41 | compute-bound | kernel (scalar triangle, no SIMD; glue ~1%) |
| 2 | `Curves.evaluate_multi` (256 curves × 2048 s, deg 8) — device path taken | 1,048,576 outputs | 1.79 ms | ~8.6 MB (8 MB out + 0.2 MB nodes + 0.02 MB grid) + GPU copies | ~34 M (32/output) | ~4.0 | compute-bound at this size | kernel on GPU (1 launch + 2 copies); host fallback is kernel too |
| 3 | `Curves.newton_refine` (64 curves × 1024 points, deg 8) | 65,536 evaluations of VS(n) + VS(n−1) | 5.34 ms | ~1.6 MB (nodes, points, starts, out) | ~4,400 M (~67 k/output pair) | ~2,800 | compute-bound | kernel (two scalar `vs_recurrence` calls per point; no SIMD body) |
| 4 | `Curves.get_curvature` (64 × 4096, deg 8) | 524,288 outputs | 2.71 ms | ~4.6 MB (tangents in, out, nodes, grid) | ~24 M (2× VS(n−2) + cross/norm per output) | ~5.2 | compute-bound | kernel (SIMD `vs_block` body; two guesstimate passes for tangents+concavity) |
| 5 | `Curves.length(panels=4096)` (64 curves, deg 6) | 64 × 4097 hodograph evals | 7.82 ms | ~0.03 MB (nodes + work, all in registers/L1) | ~122 M (~1,900/curve: 4097 × VS(5)) | ~4,000 | compute-bound | kernel (scalar `vs_recurrence` per panel; no SIMD/parallel) |
| 6 | `evaluate_multi`, 1 curve × 65,536 s (deg 8), device path taken | 131,072 outputs | 0.92 ms | ~1.05 MB | ~4.2 M (32/output) | ~4.0 | compute-bound | kernel on GPU; host SIMD path 0.72 ms measured in the same `bench` run |
| 7 | `Curves.evaluate_hodograph` (64 × 4096, deg 8) | 524,288 outputs | 1.20 ms | ~4.3 MB | ~15 M (VS(7) per output) | ~3.5 | compute-bound | kernel (SIMD `vs_block` body + scalar tail) |
| 8 | `Curves.subdivide` / `specialize` / `elevate` (512–4096 curves) | 512×2×11 nodes; 4096×2×9 | 0.23–0.29 ms each | 0.09–0.3 MB | 2.3–4.7 M (matmul/elevation per node) | ~17–50 | compute-bound (tiny) | kernel, one call per family; no per-curve ctypes re-entry |
| 9 | Per-curve drop-in path: `Curve.evaluate_multi` × 2048, 256 s each (deg 6) — **SLOWER than upstream (0.79×)** | 2048 × 512 outputs | 25.3 ms (12.3 µs/call) | 524,288 outputs total (4 MB) | ~12.6 M (24/output) | ~2.4 | **glue-bound, not memory- or compute-bound** | **Python/ctypes glue: kernel alone is ~2.6 µs (5.4 ms/2048), wrapper+transpose is the rest** |
| 10 | `locate_point` (single curve, on-curve point) | 1 curve, deg 8 | 1.22 ms | ~0.003 MB | ~0.1 M | ~30 | glue/control-flow-bound | **Python: 21 sequential batched `bc_subdivide_nodes` ctypes calls + `_contains_nd`/`np.stack` per level** |

## Where the time actually goes

### 1. The only losing row is glue, not arithmetic

`Curve.evaluate_multi` in a 2048-call loop (256 s, deg 6) is the one case
slower than upstream (0.79× in the final measurement pass, 0.85× in the
`pixi run bench` run — the loss is reproducible either way). Decomposition from
`bench/profile_curve_path.py` and the raw-kernel probes:

- kernel alone (reused buffer, `bc_evaluate_multi_s`, 1 curve × 256 s): **~2.6 µs/call**
- full `Curve.evaluate_multi`: **~12.3 µs/call** — so **~9.7 µs/call is wrapper**,
  of which `_column`'s Fortran-order transpose copy is ~2.8 µs, `_lib.evaluate_multi_s`'s
  `np.empty` + prefix-slice/reshape ~1.1 µs, `_grid` ~0.3 µs, and the two
  `_addr` (`int(array.ctypes.data)`, which builds a fresh ctypes object each
  time) + function-call overhead the remainder.
- Upstream `bezier.Curve` is a compiled Cython object and has none of this.

At the batched sizes the wrapper overhead is 1% (16.9 µs on a 1570 µs call) and
8% on the single-curve 65,536 s call (37 µs on 440 µs). It only dominates when
the call is small — which is exactly the drop-in path. The kernel is already
ahead of upstream there; the path to it is not.

**Next phase: none of unroll/simd/parallel/gpu.** The flops/byte of the kernel
(~2.4) says memory-bound, but the kernel is 2.6 µs of a 12.3 µs call, so
optimising it cannot close the gap. The fix is glue (caching the transposed
result layout / folding `_column` into the call, reusing the address ints),
which is a Python-side change, not a kernel phase. Verdict: **none** —
recorded as a next-phase action on the wrapper, not on `src/`.

### 2. de Casteljau is the biggest kernel cost and the clearest compute hotspot

64 × 4096 deg 10 takes 26.0 ms, ~20× the whole `Curves.evaluate_multi` row and
the single largest kernel time in the table. It is 330 flops/output against
~12 bytes/output (~41 flops/byte), so it is compute-bound by a wide margin —
the DRAM floor for 134 MB at 27.8 GB/s is 4.8 ms, and the kernel is 5× above
it. It is also the only evaluation kernel with **no SIMD body at all**:
`bc_evaluate_multi_de_casteljau` runs a fully scalar triangle
(`for m; for i; while index>0`) per (curve, s), and the `vs_block`/`W`-wide
machinery used by VS/hodograph/curvature is not applied. Its upstream ratio
(88×) looks large only because upstream builds the full
`(dimension, num_vals, degree)` cube in Python; the absolute cost is still the
row that dominates a workload that evaluates many curves at high degree.

**Next phase: simd.** ~41 flops/byte justifies SIMD (the triangle is
independent over `s`, exactly like the VS sweep that is already vectorised), and
rules out GPU on arithmetic-intensity grounds alone is *not* the argument —
the argument is that SIMD ports the existing `W`-wide pattern with no copies
(peak intensity 41 flops/byte is far above the ~2 flops/byte GPU rule in
MOJO_NOTES §5, so GPU would also be legitimate, but SIMD is the smaller,
proven step for this kernel; GPU would need the same launch+copy pattern the
evaluation path already pays). `parallel` is a secondary option (independent
over curves × s) but SIMD first: it needs no thread-launch overhead and the
README records `parallelize` as measured-worse on this oversubscribed box.

### 3. Evaluation (VS) is already SIMD + GPU, and the remaining cost is honest arithmetic

`Curves.evaluate_multi` 256 × 2048 (1.79 ms ours, 1.20 ms in the numpy-VS
baseline probe run at a similar size) is 32 flops/output over ~8.3 bytes/output
(~4 flops/byte), compute-bound but close to the streaming floor;
`evaluate_multi` 1 × 65,536 is 0.92 ms with the device path taken. The device
table from the `pixi run bench` run (same session):

| outputs | device | host SIMD | device/host |
| ---: | ---: | ---: | ---: |
| 262,144 (1×131072) | 0.626 ms | 0.721 ms | 0.87 |
| 131,072 (8×8192) | 0.321 ms | 0.378 ms | 0.85 |
| 524,288 (32×8192) | 0.761 ms | 1.534 ms | 0.50 |
| 1,048,576 (256×2048) | 1.255 ms | 2.835 ms | 0.45 |
| 16,777,216 (1024×8192) | 80.703 ms | 98.589 ms | 0.82 |

Both paths are the same scalar recurrence, so the split is purely launch+copy
vs. host FLOPs. A plain-NumPy VS baseline is 20–32× slower than the port at
64×4096 and 256×2048, so the host kernel is not the problem; the GPU already
covers the large batches.

**Next phase: gpu — already done, none further.** ~4 flops/byte is above the
~2 flops/byte CPU-only rule, and the device is 1.8× faster at 1 M outputs; the
verdict is that GPU is justified *and present*, but only pays above
`GPU_MIN_OUTPUTS = 131072` outputs. Below that the host SIMD path is the right
one and SIMD is already applied. No new GPU work is recommended; if anything,
the remaining host gap is SIMD tail coverage (`num_vals % W`), which is a
small `simd` follow-up, not a GPU phase.

### 4. Newton refine and length are scalar-loop compute hotspots

`newton_refine` (5.34 ms for 65,536 points) calls `vs_recurrence` twice per
point — once for `B(s)`, once for `B'(s)` — with the `dim`-sized accumulator
spilled to `work` each time and no `W`-wide body, even though points are an
independent contiguous axis exactly like `s`. ~67 k flops per point over ~24
bytes of point/start/out traffic (~2,800 flops/byte) is unambiguously
compute-bound and SIMD-eligible. `length` (7.82 ms for 64 curves × 4096 panels)
re-evaluates the hodograph via the same scalar `vs_recurrence` 4097 times per
curve; at ~1,900 flops/curve with only ~50 bytes of traffic it is likewise
scalar-compute-bound. Both are already 160–2300× upstream, so these are
"make the already-fast faster" rows, not losses.

**Next phase: simd** for `newton_refine` (vectorise over points, fl/byte ~2,800
removes any GPU question), **simd** for `length` (vectorise over the panel
index, or reuse the hodograph's `vs_block` body — same argument, far above the
2 flops/byte GPU line). `parallel` was measured and reverted in this repo
(README "Parallelism (tried, reverted)") on this oversubscribed box, so it is
not recommended over SIMD.

### 5. Hodograph and curvature are SIMD-covered and near their intensity

Hodograph (1.20 ms, ~3.5 flops/byte) and curvature (2.71 ms, ~5.2 flops/byte)
both run the `vs_block` vector body with a scalar tail; both are
compute-bound and both are already ahead of their vectorised-NumPy upstream
baselines (21× and 26×). The curvature kernel also re-reads the tangent buffer
that `Curves.get_curvature` computes first, which is a second pass the
intensity number above accounts for.

**Next phase: simd (done) / none.** SIMD is applied; ~3.5–5.2 flops/byte is
above the GPU CPU-only line but the kernels are already GPU-adjacent only via
`evaluate_multi_s`, and extending the device path here would copy the tangent
buffer unnecessarily for a 2× win at sizes where host SIMD is 1.2–2.7 ms.
Verdict: **none** (SIMD complete, GPU not justified at these sizes).

### 6. Subdivide / specialize / elevate are already the fastest rows

0.23–0.29 ms for 512–4096 curves, 17–50 flops/byte, one batched call each, no
per-curve ctypes re-entry. They are kernel-bound and far ahead of upstream
(90–500×), because upstream is a Python-call-per-curve loop.

**Next phase: none.**

### 7. `locate_point` is sequential ctypes re-entry

One `locate_point` call makes **21 `bc_subdivide_nodes` ctypes calls** plus one
`bc_newton_refine` call (counted with a wrapped `lib` in this session), each
followed by Python-side `_contains_nd`, `np.stack`, `np.std`/`np.mean` work.
At 1.22 ms vs upstream's 1.70 ms this is a small win only because upstream does
the same subdivision in Python. The Mojo kernels involved cost ~30 µs
(batched 2-curve subdivide); the rest is 21 rounds of Python control flow.

**Next phase: none of the four.** The flops/byte (~30) is a red herring — the
cost is sequential Python↔ctypes round-trips per subdivision level, and the fix
is moving the candidate loop into one kernel call (the batched-subdivide
pattern already exists), not SIMD/parallel/GPU. Flagged here as a Python-loop
hotspot per the brief.

## Python-level loops and per-call allocations flagged

1. `curve_helpers.evaluate_multi` and `Curve.evaluate_multi` (the 2048-call
   path): per call this does `as_batch` validation (a finiteness pass over the
   nodes), `int(array.ctypes.data)` twice (each builds a fresh ctypes object),
   `np.empty(size + degree + 1)`, a prefix slice + `reshape`, `_grid`, and
   `_column`'s `np.asfortranarray` transpose copy. At 256 s the kernel is 2.6 µs
   and the list above is ~9.7 µs. (`Curve`/`Curves` already cache `Batch` for
   the evaluation path; the `curve_helpers` stateless path and `_column` do not.)
2. `locate_point`: 21 sequential `bc_subdivide_nodes` calls with
   `np.stack`/`_contains_nd` between each — a Python loop that re-enters ctypes
   once per subdivision level instead of once per call.
3. `Curves.get_curvature`: allocates the tangent buffer in Python
   (`_lib.evaluate_hodograph`), then `bc_get_curvature` re-reads it; one extra
   full-size buffer and pass per call. `evaluate_multi_de_casteljau` and
   `vec_size` similarly allocate `work`/`dst` in Python per call (small, but
   per-call).
4. Upstream-side loops (for context, not port cost): the bench's "1 call vs N"
   rows are upstream Python loops (256–4096 calls of `up.evaluate_multi` /
   `up.subdivide_nodes` / etc. per case), which is why those ratios are large.
   The `de_casteljau_one_round` single-curve helper is the only entry point
   where the port is slower than upstream's NumPy expression
   (~15 µs vs ~4 µs at 1×32 in this session's probe), because it is a tiny
   NumPy op with the same ctypes overhead; it does not appear in `bench.py`.

## Per-call latency (single curve, Python entry points)

Measured this session (deg 8, best-of-7), the sizes a caller naming one curve
actually passes:

| entry point | 1×1 or 1×256 | time |
| --- | ---: | ---: |
| `curve_helpers.evaluate_multi` (1×256) | | 20.2 µs |
| `curve_helpers.vec_size` | 1×1 | 22.1 µs |
| `curve_helpers.evaluate_hodograph` | 1×1 | 17.8 µs |
| `curve_helpers.get_curvature` | 1×1 | 121.5 µs |
| `curve_helpers.newton_refine` | 1×1 | 107.5 µs |
| `curve_helpers.compute_length` (panels=1024) | 1 curve | 54.0 µs |
| `curve_helpers.subdivide_nodes` | 1 curve | 17.7 µs |
| `curve_helpers.elevate_nodes` | 1 curve | 11.3 µs |
| `curve_helpers.specialize_curve` | 1 curve | 15.4 µs |
| `curve_helpers.make_subdivision_matrices` | deg 8, cached | 1.5 µs |
| `curve_helpers.locate_point` | on-curve point | ~1,220 µs |
| `curve_helpers.evaluate_multi_de_casteljau` (1×1024, deg 8) | | 103.9 µs |

At these sizes every row is glue-dominated (the kernel work is O(deg) or
O(deg×256) flops, i.e. sub-µs of arithmetic); `get_curvature`/`newton_refine`
are ~5–6× `vec_size` because their wrappers allocate several buffers and do
extra `reshape`/`asfortranarray` steps.

## Full `pixi run bench` output (this session)

```
mojo-bezier-curves against upstream bezier

python      3.12.14 on x86_64
numpy       2.5.3
upstream    bezier 2024.6.20
timing      best of 5 runs

| case | upstream | mojo-bezier-curves | speedup |
| --- | ---: | ---: | ---: |
| evaluate, 1 curve x 65536 s (deg 8) | 6.85 ms | 0.91 ms | 7.56x |
| evaluate, 256 curves x 2048 s (deg 8), 1 call vs 256 | 84.99 ms | 1.32 ms | 64.49x |
| evaluate (de Casteljau), 64 curves x 4096 s (deg 10), 1 call vs 64 | 2913.85 ms | 27.56 ms | 105.73x |
| hodograph, 64 curves x 4096 s (deg 8), 1 call vs 64 | 32.47 ms | 1.44 ms | 22.52x |
| curvature, 64 curves x 4096 s (deg 8), 1 call vs 64 | 78.04 ms | 3.00 ms | 26.00x |
| newton refine, 64 curves x 1024 points (deg 8), 1 call vs 65536 | 13031.62 ms | 5.28 ms | 2467.16x |
| subdivide, 512 curves (deg 10), 1 call vs 512 | 26.09 ms | 0.23 ms | 113.97x |
| specialize, 512 curves (deg 10), 1 call vs 512 | 180.79 ms | 0.36 ms | 500.04x |
| elevate, 4096 curves (deg 8), 1 call vs 4096 | 50.80 ms | 0.24 ms | 213.50x |
| length, 64 curves (deg 6), 1 call vs 64 | 1365.58 ms | 8.00 ms | 170.63x |
| evaluate via Curve objects, 2048 x 256 s (deg 6) | 21.44 ms | 25.24 ms | **0.85x, slower** |
| evaluate via curve_helpers (pure Python), 2048 x 256 s | 218.47 ms | 43.99 ms | 4.97x |

11 of 12 cases at or above upstream parity.

| outputs (curves x s) | device | host SIMD | device/host |
| ---: | ---: | ---: | ---: |
| 262144 (1 x 131072 s) | 0.626 ms | 0.721 ms | 0.87 |
| 131072 (8 x 8192 s) | 0.321 ms | 0.378 ms | 0.85 |
| 524288 (32 x 8192 s) | 0.761 ms | 1.534 ms | 0.50 |
| 1048576 (256 x 2048 s) | 1.255 ms | 2.835 ms | 0.44 |
| 16777216 (1024 x 8192 s) | 80.703 ms | 98.589 ms | 0.82 |
```

A second, independent measurement pass in the same session (same shapes,
best-of-7) gave: evaluate 1×65536 0.92 ms vs 6.22 ms upstream; 256×2048
1.79 ms vs 80.10 ms; de Casteljau 64×4096 25.99 ms vs 2301.84 ms; hodograph
1.20 ms vs 28.02 ms; curvature 2.71 ms vs 69.74 ms; Newton 5.34 ms vs
12150.88 ms; subdivide 0.254 ms vs 22.95 ms; specialize 0.293 ms vs 157.56 ms;
elevate 0.227 ms vs 47.62 ms; length 7.82 ms vs 1298.82 ms; Curve objects
25.27 ms vs 20.08 ms (**slower**); pure-Python helpers 46.57 ms vs 213.57 ms.
A plain-NumPy VS baseline at the same shapes was 13.77 ms (64×4096) and
38.40 ms (256×2048) — 20–32× slower than the port — and a NumPy
vectorised-over-s de Casteljau loop was 268.9 ms for one curve × 4096
(deg 10) against the port's 0.426 ms.

## Recommended next phases, in order

| hotspot | next phase | evidence |
| --- | --- | --- |
| de Casteljau evaluation (26 ms, ~41 flops/byte, scalar) | **simd** | compute-bound by 5× over the 27.8 GB/s floor; independent over `s`; zero SIMD body today |
| `newton_refine` (5.3 ms, ~2,800 flops/byte, scalar) | **simd** | vectorise over points with the `vs_block` pattern; intensity rules GPU out |
| `length` (7.8 ms, scalar hodograph per panel) | **simd** | vectorise over panels / reuse `vs_block`; current best is already 170× upstream |
| drop-in per-curve path (0.79×, glue-bound) | **none** (Python wrapper fix, not a kernel phase) | kernel is 2.6 µs of a 12.3 µs call; `_column` transpose + `np.empty` + addresses dominate |
| `locate_point` (21 ctypes round-trips) | **none** (one kernel call, not a kernel phase) | sequential Python loop; flops/byte is not the cost |
| VS evaluation at large batch (device table above) | **gpu — already shipped** | ~4 flops/byte; device 1.8× host at 1 M outputs; threshold at 131072 outputs is correct |
| hodograph / curvature / subdivide / specialize / elevate | **none** | SIMD applied; 20–500× upstream; further work buys <1.2 ms |

No row was dropped, including the two slower-than-upstream ones
(`Curve.evaluate_multi` × 2048 and, at single-curve size,
`de_casteljau_one_round`).
