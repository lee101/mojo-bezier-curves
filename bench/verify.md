# Verify — mojo-bezier-curves vs upstream bezier 2024.6.20 (second session)

Run in this session, in order, through `pixi run` (which holds the `flock`
in `bench/bench.py`):

1. `pixi run build && pixi run test` — green. 995 passed.
2. `pixi run bench` — run 1 (saved to `/tmp/bench_verify_run1.txt` at the time;
   full output reproduced below).
3. `pixi run bench` — run 2 (saved to `/tmp/bench_verify_2.txt`; full output
   reproduced below), run to check whether run 1's deviations were stable.

Both bench runs time upstream first, best of 5 after 1 warm-up, in one
process, and every case asserts parity against upstream before timing
(`_check` / `assert`, relative error < 1e-9, exact equality for the device
sweep). Python 3.12.14, NumPy 2.5.3, `bezier` 2024.6.20,
Mojo 1.2.0.dev2026092905, Intel Xeon E5-2697 v4 (shared box, RTX 5090 when
the device path engages).

## What reproduced (within 20%)

Main table, all 12 rows, run 1 (the run the main table was corrected to):

| case | README (before) | measured (run 1) | max dev vs README |
| --- | --- | --- | --- |
| evaluate, 1 curve x 65536 s (deg 8) | 5.94 / 0.97 = 6.10x | 6.28 / 0.91 = 6.88x | 12.8% |
| evaluate, 256 x 2048 (deg 8) | 76.40 / 1.42 = 53.78x | 77.79 / 1.19 = 65.50x | 21.8% (speedup column only) |
| evaluate (de Casteljau), 64 x 4096 (deg 10) | 2800.80 / 26.24 = 106.72x | 2305.49 / 26.80 = 86.03x | 19.4% |
| hodograph, 64 x 4096 (deg 8) | 30.23 / 1.27 = 23.76x | 28.50 / 1.31 = 21.76x | 8.4% |
| curvature, 64 x 4096 (deg 8) | 73.78 / 2.81 = 26.26x | 69.53 / 2.88 = 24.18x | 7.9% |
| newton refine, 64 x 1024 (deg 8) | 12812.93 / 5.17 = 2477.95x | 11289.25 / 5.38 = 2100.05x | 15.3% |
| subdivide, 512 (deg 10) | 23.32 / 0.20 = 117.17x | 21.67 / 0.19 = 114.36x | 7.1% |
| specialize, 512 (deg 10) | 159.16 / 0.37 = 426.65x | 150.58 / 0.36 = 413.67x | 5.4% |
| elevate, 4096 (deg 8) | 48.66 / 0.22 = 216.58x | 45.20 / 0.23 = 197.09x | 9.0% |
| length, 64 (deg 6) | 1297.26 / 7.97 = 162.83x | 1199.19 / 7.85 = 152.71x | 7.6% |
| evaluate via Curve objects, 2048 x 256 (deg 6) | 20.50 / 30.60 = **0.67x, slower** | 20.39 / 27.44 = **0.74x, slower** | 10.4% |
| evaluate via curve_helpers (pure Python), 2048 x 256 | 214.07 / 45.88 = 4.67x | 187.84 / 39.14 = 4.80x | 14.7% |

Deviation is computed per column as `|measured − README| / README`, with the
speedup recomputed as upstream ÷ port from the measured columns. Exactly one
cell in run 1 exceeds the 20% band: row 2's speedup column (65.50x vs 53.78x,
21.8%). Both absolute columns of that row are within the band (1.8% and
16.2%), so the row is a ratio of two in-band numbers landing just outside
it — this is the row the README flags as device-path and tenant-dependent.
Run 2 gives the same signal (66.10x, 22.9% out of band), so it is not a
one-run artefact: the README's ratio for this row was low, and the corrected
table carries the measured 65.50x. No row flips its win/loss verdict in
either run: 11 of 12 at or above parity in both, the Curve-objects row
slower in both (0.74x in run 1, 0.87x in run 2, against 0.67x in the README).

Run 2 against the pre-correction README: every cell within 20% except row 2's
speedup (22.9%) and row 11 (port column 25.6%, speedup 29.9% — the
per-call-marshalling row, which is the noisiest per-call measurement; it
reproduced within 10.4% in run 1). The ratio spread between the two runs
for row 2 (65.50x vs 66.10x) is small; the row-2 upstream column moved
77.79 → 87.47 ms, which is the shared-box "tens of percent" movement the
README warns about.

The one slower row (Curve objects) was left slower in the README: same
wording, same `**…x, slower**` formatting, `0.67x` → `0.74x` only because
the measured ratio is 0.74x. It was not removed and not reworded.

The device table reproduced within 20% on every cell in run 1 (max dev
14.3%, the 524288 ratio), and in run 2 on all but the largest device cell
(80.477 vs 107.137 ms, 24.9% — the 16.8M-output copy-dominated row, the one
most sensitive to the shared GPU). In run 1 the device was level with the
host at the smallest size (ratio 1.00) rather than the README's 0.90; the
corrected prose says so. The variance warning below the table is unchanged.

## What did not reproduce

Nothing beyond the band items above. Every main-table figure was replaced
with the run-1 measurement regardless, per the README's own "the output of
one `pixi run bench`" provenance claim — a figure within 20% this time is
still a figure from an earlier session's box state. The device table was
replaced with the run-1 numbers for the same reason (run 1 and run 2 agree
within 20% on all but the largest device cell, so run 1 is not an outlier
this time and there was no reason to prefer run 2). The one correction
beyond mechanical replacement: the device-table prose sentence (1.11x /
1.06x / 1.77x / 2.12x / 1.06x) now reads from the new ratios (level at
262144, 1.02x at 131072, 2.06x at 524288, 2.18x at 1048576, 1.22x at
16777216).

## What was changed and why

* **README main table**: every figure replaced with the run-1 measurement.
  Nothing was invented: every new figure is a number measured in this
  session. Row 11 stays slower, same wording.
* **README device table + prose**: replaced with the run-1 numbers and the
  prose corrected to match those ratios (1.00 / 0.98 / 0.48 / 0.46 / 0.82
  device/host, i.e. level, 1.02x, 2.06x, 2.18x, 1.22x ahead). The
  pre-existing variance warning is unchanged.
* **No code changed.** The brief allows code changes only to fix a bug
  shown by a failing test. No test failed (995 passed) and no failing
  test identified a bug, so `src/`, `python/`, and `tests/` are untouched.
  This file is the only new file.

## Same-input / no-skip-work audit (`bench/bench.py`)

Reading `bench/bench.py` case by case:

* Every case builds its input once and hands the same object to both sides:
  `nodes` + `values` are constructed in the case function and captured by
  both the `ours` and `theirs` lambdas. Family cases loop upstream once per
  curve (`_loop(…, 64/256/512/4096)`) which is exactly the work the
  unbatched upstream forces; the label says "1 call vs N" in the row name.
* Every case asserts parity before timing: `case_evaluate_one_curve`,
  `case_evaluate_family`, `case_evaluate_de_casteljau_family`,
  `case_hodograph_family`, `case_curvature_family`,
  `case_newton_refine_family` (spot-checked at indices 0/63, k0/500/1023),
  `case_subdivide_family`, `case_specialize_family`,
  `case_elevate_family`, `case_length_family`, `case_curve_objects`,
  `case_pure_python_helpers` each call `_check` or `assert` against
  upstream's own function on the same input. A fast-but-wrong number would
  raise before `_time` runs.
* `case_newton_refine_family` builds its target points from *our*
  `Curves.evaluate_multi` (`0.5 + 0.25*(values-0.5)` mapped on the curve),
  then checks the refine result against upstream's `newton_refine` — the
  check would fail if our evaluation were wrong, so the point-generation is
  not a skip.
* `_upstream_hodograph` / `_upstream_curvature` are vectorised NumPy
  expressions built from upstream's own `evaluate_multi` (no scalar
  upstream helpers exist to time for those two): `n * evaluate_multi(Δv, s)`
  and the cross/norm formula. The README and the docstring say so. The port
  does the same mathematics through its kernel.
* `_time` does 1 warm-up + best-of-5 identical calls on both sides. No
  conditional skipping on either side; the only branch is the documented
  device/host split in `device_sweep`, which asserts exact device-vs-host
  equality before timing and returns `None` honestly when no device takes
  the batch.
* `case_length_family` compares our Simpson (4096 panels) against upstream
  QUADPACK through `_check` with the 1e-9 relative-error gate, so the
  faster integral is not silently less accurate on the timed inputs.
  README already flags `length` as a different amount of work, so this is
  disclosed, not hidden.

No case skips work on one side. The one upstream baseline that is not
upstream's public API is the hodograph/curvature pair, which is disclosed in
the docstring and is the hardest reasonable baseline (vectorised NumPy on
upstream's own `evaluate_multi`, not a Python loop over the scalar API).

## Parity-test history

`git log -p -- tests/` across all commits:

* `c3c29a1` (mojo 1.2.0 migration): adds `tests/test_evaluate.py` and
  `tests/test_shape.py`.
* `54188bb` (the Mojo port): deletes those two files, replaces them with
  `test_batch.py`, `test_buffers.py`, `test_curve_parity.py`,
  `test_doctest_values.py`, `test_gpu.py`, `test_hazmat_parity.py`,
  `test_identities.py`, `test_paths.py`, `test_simd_tail.py`, and rewrites
  `conftest.py`. The deleted tests compared against `mbc.bernstein`
  (a longhand Bernstein definition) because the upstream package was not
  installed in that environment at that point; the replacement tests
  compare against the real upstream `bezier` package installed in the
  workspace. No tolerance was present in the old files that the new files
  loosened: the old ones used `rtol=1e-12/1e-11` against `mbc.bernstein`,
  the new ones use `RTOL=1e-12, ATOL=1e-13` in `conftest.py` against
  upstream — the same band, a stricter reference.
* `4e157ac` (phase test): adds `tests/test_parity_extra.py` (460 lines),
  tightening coverage (barycentric, hodograph helper-to-helper, degenerate
  inputs). Adds assertions; removes none.
* `3183e5e` (phase analyze): touches only `bench/analysis.md`. No test file
  changed, no tolerance changed.
* `e6bcc1a` (prior phase verify): touches only `README.md` and
  `bench/verify.md`. No test file changed.

`RTOL=1e-12, ATOL=1e-13` is identical at every commit that has a
`conftest.py` (`54188bb`, `4e157ac`, `3183e5e`, `e6bcc1a`). `git diff
54188bb..HEAD -- tests/` contains zero removed lines — only the 460 added
lines of `test_parity_extra.py`. No assertion was removed or loosened by
an earlier phase that this phase needed to restore; per the brief's rule
(code changes only to fix a bug shown by a failing test), nothing was
restored because nothing was loosened. The one former exact-equality check
that changed (`batched vs per-curve evaluation`) was bit-equality, not
parity-with-upstream, and is documented in the README as an FMA-rounding
consequence; upstream parity is still `rtol=1e-12` there.

## Raw bench output, run 1 (main table source)

```
mojo-bezier-curves against upstream bezier

python      3.12.14 on x86_64
numpy       2.5.3
upstream    bezier 2024.6.20
timing      best of 5 runs

| case | upstream | mojo-bezier-curves | speedup |
| --- | ---: | ---: | ---: |
| evaluate, 1 curve x 65536 s (deg 8) | 6.28 ms | 0.91 ms | 6.88x |
| evaluate, 256 curves x 2048 s (deg 8), 1 call vs 256 | 77.79 ms | 1.19 ms | 65.50x |
| evaluate (de Casteljau), 64 curves x 4096 s (deg 10), 1 call vs 64 | 2305.49 ms | 26.80 ms | 86.03x |
| hodograph, 64 curves x 4096 s (deg 8), 1 call vs 64 | 28.50 ms | 1.31 ms | 21.76x |
| curvature, 64 curves x 4096 s (deg 8), 1 call vs 64 | 69.53 ms | 2.88 ms | 24.18x |
| newton refine, 64 curves x 1024 points (deg 8), 1 call vs 65536 | 11289.25 ms | 5.38 ms | 2100.05x |
| subdivide, 512 curves (deg 10), 1 call vs 512 | 21.67 ms | 0.19 ms | 114.36x |
| specialize, 512 curves (deg 10), 1 call vs 512 | 150.58 ms | 0.36 ms | 413.67x |
| elevate, 4096 curves (deg 8), 1 call vs 4096 | 45.20 ms | 0.23 ms | 197.09x |
| length, 64 curves (deg 6), 1 call vs 64 | 1199.19 ms | 7.85 ms | 152.71x |
| evaluate via Curve objects, 2048 x 256 s (deg 6) | 20.39 ms | 27.44 ms | **0.74x, slower** |
| evaluate via curve_helpers (pure Python), 2048 x 256 s | 187.84 ms | 39.14 ms | 4.80x |

11 of 12 cases at or above upstream parity.

| outputs (curves x s) | device | host SIMD | device/host |
| ---: | ---: | ---: | ---: |
| 262144 (1 x 131072 s) | 0.638 ms | 0.638 ms | 1.00 |
| 131072 (8 x 8192 s) | 0.336 ms | 0.342 ms | 0.98 |
| 524288 (32 x 8192 s) | 0.688 ms | 1.420 ms | 0.48 |
| 1048576 (256 x 2048 s) | 1.251 ms | 2.726 ms | 0.46 |
| 16777216 (1024 x 8192 s) | 93.326 ms | 114.076 ms | 0.82 |
```

## Raw bench output, run 2 (reproducibility check)

```
mojo-bezier-curves against upstream bezier

python      3.12.14 on x86_64
numpy       2.5.3
upstream    bezier 2024.6.20
timing      best of 5 runs

| case | upstream | mojo-bezier-curves | speedup |
| --- | ---: | ---: | ---: |
| evaluate, 1 curve x 65536 s (deg 8) | 6.29 ms | 1.02 ms | 6.19x |
| evaluate, 256 curves x 2048 s (deg 8), 1 call vs 256 | 87.47 ms | 1.32 ms | 66.10x |
| evaluate (de Casteljau), 64 curves x 4096 s (deg 10), 1 call vs 64 | 2699.02 ms | 27.46 ms | 98.29x |
| hodograph, 64 curves x 4096 s (deg 8), 1 call vs 64 | 30.02 ms | 1.24 ms | 24.20x |
| curvature, 64 curves x 4096 s (deg 8), 1 call vs 64 | 72.11 ms | 2.69 ms | 26.77x |
| newton refine, 64 curves x 1024 points (deg 8), 1 call vs 65536 | 11467.13 ms | 5.24 ms | 2189.46x |
| subdivide, 512 curves (deg 10), 1 call vs 512 | 23.54 ms | 0.21 ms | 110.03x |
| specialize, 512 curves (deg 10), 1 call vs 512 | 154.08 ms | 0.34 ms | 456.28x |
| elevate, 4096 curves (deg 8), 1 call vs 4096 | 46.35 ms | 0.24 ms | 193.25x |
| length, 64 curves (deg 6), 1 call vs 64 | 1134.21 ms | 7.82 ms | 145.02x |
| evaluate via Curve objects, 2048 x 256 s (deg 6) | 19.89 ms | 22.77 ms | **0.87x, slower** |
| evaluate via curve_helpers (pure Python), 2048 x 256 s | 200.94 ms | 40.04 ms | 5.02x |

11 of 12 cases at or above upstream parity.

| outputs (curves x s) | device | host SIMD | device/host |
| ---: | ---: | ---: | ---: |
| 262144 (1 x 131072 s) | 0.598 ms | 0.623 ms | 0.96 |
| 131072 (8 x 8192 s) | 0.302 ms | 0.329 ms | 0.92 |
| 524288 (32 x 8192 s) | 0.762 ms | 1.313 ms | 0.58 |
| 1048576 (256 x 2048 s) | 1.224 ms | 2.631 ms | 0.47 |
| 16777216 (1024 x 8192 s) | 80.477 ms | 101.072 ms | 0.80 |
```

## Final gate

`pixi run build && pixi run test` — re-run at the end (see the session log);
both green, 995 passed, before and after the README edit. No source file was
changed after the first green run, so the library the benchmarks ran against
is the library the tests verified.
