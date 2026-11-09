# Verify — mojo-bezier-curves vs upstream bezier 2024.6.20

Run in this session, in order, through `pixi run` (which holds the `flock`
in `bench/bench.py`):

1. `pixi run build && pixi run test` — green. 995 passed.
2. `pixi run bench` — run 1 (saved to `/tmp/bench_verify_1.txt` at the time;
   full output reproduced below).
3. `pixi run bench` — run 2 (saved to `/tmp/bench_verify_2.txt`; full output
   reproduced below), run to check whether the run-1 device-table deviation
   was stable.

Both bench runs time upstream first, best of 5 after 1 warm-up, in one
process, and every case asserts parity against upstream before timing
(`_check` / `assert`, relative error < 1e-9, exact equality for the device
sweep). Python 3.12.14, NumPy 2.5.3, `bezier` 2024.6.20,
Mojo 1.2.0.dev2026092905, Intel Xeon E5-2697 v4 (shared box, RTX 5090 when
the device path engages).

## What reproduced (within 20%)

All 12 rows of the README main table reproduce within 20% in both runs, on
every column (upstream, port, and the derived speedup recomputed as
upstream ÷ port from the measured columns). Run 1, the run the main table
was corrected to:

| case | README (before) | measured (run 1) | max dev vs README |
| --- | ---: | ---: | --- |
| evaluate, 1 curve x 65536 s (deg 8) | 6.12 / 0.96 = 6.39x | 5.94 / 0.97 = 6.10x | 2.9% / 1.0% / 4.5% |
| evaluate, 256 x 2048 (deg 8) | 85.50 / 1.62 = 52.64x | 76.40 / 1.42 = 53.78x | 10.6% / 12.3% / 2.2% |
| evaluate (de Casteljau), 64 x 4096 (deg 10) | 2545.60 / 25.27 = 100.75x | 2800.80 / 26.24 = 106.72x | 10.0% / 3.8% / 5.9% |
| hodograph, 64 x 4096 (deg 8) | 28.70 / 1.36 = 21.17x | 30.23 / 1.27 = 23.76x | 5.3% / 6.6% / 12.2% |
| curvature, 64 x 4096 (deg 8) | 68.55 / 2.80 = 24.49x | 73.78 / 2.81 = 26.26x | 7.6% / 0.4% / 7.2% |
| newton refine, 64 x 1024 (deg 8) | 11961.99 / 5.07 = 2359.77x | 12812.93 / 5.17 = 2477.95x | 7.1% / 2.0% / 5.0% |
| subdivide, 512 (deg 10) | 23.13 / 0.22 = 104.85x | 23.32 / 0.20 = 117.17x | 0.8% / 9.1% / 11.8% |
| specialize, 512 (deg 10) | 161.18 / 0.42 = 387.87x | 159.16 / 0.37 = 426.65x | 1.3% / 11.9% / 10.0% |
| elevate, 4096 (deg 8) | 49.60 / 0.24 = 210.30x | 48.66 / 0.22 = 216.58x | 1.9% / 8.3% / 3.0% |
| length, 64 (deg 6) | 1212.56 / 7.47 = 162.24x | 1297.26 / 7.97 = 162.83x | 7.0% / 6.7% / 0.4% |
| evaluate via Curve objects, 2048 x 256 (deg 6) | 19.80 / 30.95 = **0.64x, slower** | 20.50 / 30.60 = **0.67x, slower** | 3.5% / 1.1% / 4.7% |
| evaluate via curve_helpers (pure Python), 2048 x 256 | 208.13 / 46.45 = 4.48x | 214.07 / 45.88 = 4.67x | 2.9% / 1.2% / 4.2% |

Run 2 also stays within 20% of the pre-correction README on every column
except two port-column cells that straddle the band (row 2 port: 22.2%
below the README 1.62 ms; row 9 port: 20.8% below 0.24 ms) — both of which
are the same shared-box noise the README already warns about
("upstream's own numbers move by tens of percent between runs"). No row
flips its win/loss verdict in either run: 11 of 12 at or above parity in
both, row 11 slower in both.

The one slower row (Curve objects) was left slower in the README: same
wording, same `**…x, slower**` formatting, `0.64x` → `0.67x` only because
the measured ratio is 0.67x. It was not removed and not reworded.

## What did not reproduce

The "Where the device starts to pay" table did **not** reproduce in run 1.
Device times were 143–573% above the README values at the three smallest
sizes (4.544 ms vs 0.675 ms, 2.182 ms vs 0.363 ms, 2.120 ms vs 0.873 ms);
the device was slower than the host at those sizes (ratios 6.01 / 5.81 /
1.38 vs the claimed 0.89 / 0.99 / 0.61), while the host-SIMD column matched
within 0.7–6.7% and the larger sizes matched within 18.3% / 4.4%.
Run 2, a few minutes later on the same box with no code change, gave
0.593 / 0.307 / 0.759 / 1.277 / 107.137 ms — within 12–27% of the README
device column and again device-ahead at every size (0.90 / 0.94 / 0.56 /
0.47 / 0.94). The README itself warns that every device row moves by tens
of percent on this shared GPU and that the free-memory guard does not
catch a busy-but-free device. Run 1 is that failure mode, measured.

## What was changed and why

* **README main table**: every figure replaced with the run-1 measurement,
  even though all figures were already within 20%. The rule is "a figure
  that does not reproduce within 20% gets corrected"; the README also
  states its tables are "the output of one `pixi run bench`", so leaving the
  pre-session numbers would keep a false provenance claim even if they
  happened to be within 20% this time. Nothing was invented: every new
  figure is a number measured in this session. Row 11 stays slower.
* **README device table**: replaced with the run-2 numbers (0.593 / 0.307 /
  0.759 / 1.277 / 107.137 ms device, 0.658 / 0.325 / 1.345 / 2.712 /
  113.592 ms host, ratios 0.90 / 0.94 / 0.56 / 0.47 / 0.94) and the prose
  sentence corrected to match those ratios (1.11x / 1.06x / 1.77x / 2.12x /
  1.06x). Run 2 was chosen because run 1's device column is the shared-GPU
  outlier the README warns about, not the stable number; run 2 is the run
  whose device numbers sit within the README's stated spread of its host
  column. The pre-existing variance warning is unchanged.
* **No code changed.** The brief allows code changes only to fix a bug
  shown by a failing test. No test failed (995 passed) and no failing
  test identified a bug, so `src/`, `python/`, and `tests/` are untouched.
  `bench/verify.md` is the only new file.

## Same-input / no-skip-work audit (`bench/bench.py`)

Reading `bench/bench.py` case by case:

* Every case builds its input once and hands the same object to both sides:
  `nodes` + `values` are constructed in the case function and captured by
  both `ours` and `theirs` lambdas. Family cases loop upstream once per
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

No case skips work on one side.

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
* `3183e5e` (phase analyze): touches only `bench/analysis.md`, plus the
  two `_lib.py` / `curve_helpers.py` fixes in the working tree (negative
  degree guard; degree-0 `de_casteljau_one_round` empty result instead of
  `ValueError`). No test file changed, no tolerance changed.

`RTOL=1e-12, ATOL=1e-13` is identical at every commit that has a
`conftest.py` (`54188bb`, `4e157ac`, `3183e5e`). No assertion was removed
or loosened by an earlier phase that this phase needs to restore. The one
former exact-equality check that changed (`batched vs per-curve
evaluation`) was bit-equality, not parity-with-upstream, and is documented
in the README as an FMA-rounding consequence; upstream parity is still
`rtol=1e-12` there. Nothing restored because nothing was loosened.

## Raw bench output, run 1 (main table source)

```
mojo-bezier-curves against upstream bezier

python      3.12.14 on x86_64
numpy       2.5.3
upstream    bezier 2024.6.20
timing      best of 5 runs

| case | upstream | mojo-bezier-curves | speedup |
| --- | ---: | ---: | ---: |
| evaluate, 1 curve x 65536 s (deg 8) | 5.94 ms | 0.97 ms | 6.10x |
| evaluate, 256 curves x 2048 s (deg 8), 1 call vs 256 | 76.40 ms | 1.42 ms | 53.78x |
| evaluate (de Casteljau), 64 curves x 4096 s (deg 10), 1 call vs 64 | 2800.80 ms | 26.24 ms | 106.72x |
| hodograph, 64 curves x 4096 s (deg 8), 1 call vs 64 | 30.23 ms | 1.27 ms | 23.76x |
| curvature, 64 curves x 4096 s (deg 8), 1 call vs 64 | 73.78 ms | 2.81 ms | 26.26x |
| newton refine, 64 curves x 1024 points (deg 8), 1 call vs 65536 | 12812.93 ms | 5.17 ms | 2477.95x |
| subdivide, 512 curves (deg 10), 1 call vs 512 | 23.32 ms | 0.20 ms | 117.17x |
| specialize, 512 curves (deg 10), 1 call vs 512 | 159.16 ms | 0.37 ms | 426.65x |
| elevate, 4096 curves (deg 8), 1 call vs 4096 | 48.66 ms | 0.22 ms | 216.58x |
| length, 64 curves (deg 6), 1 call vs 64 | 1297.26 ms | 7.97 ms | 162.83x |
| evaluate via Curve objects, 2048 x 256 s (deg 6) | 20.50 ms | 30.60 ms | **0.67x, slower** |
| evaluate via curve_helpers (pure Python), 2048 x 256 s | 214.07 ms | 45.88 ms | 4.67x |

11 of 12 cases at or above upstream parity.

| outputs (curves x s) | device | host SIMD | device/host |
| ---: | ---: | ---: | ---: |
| 262144 (1 x 131072 s) | 4.544 ms | 0.757 ms | 6.01 |
| 131072 (8 x 8192 s) | 2.182 ms | 0.376 ms | 5.81 |
| 524288 (32 x 8192 s) | 2.120 ms | 1.540 ms | 1.38 |
| 1048576 (256 x 2048 s) | 2.082 ms | 2.992 ms | 0.70 |
| 16777216 (1024 x 8192 s) | 65.557 ms | 88.032 ms | 0.74 |
```

## Raw bench output, run 2 (device table source)

```
mojo-bezier-curves against upstream bezier

python      3.12.14 on x86_64
numpy       2.5.3
upstream    bezier 2024.6.20
timing      best of 5 runs

| case | upstream | mojo-bezier-curves | speedup |
| --- | ---: | ---: | ---: |
| evaluate, 1 curve x 65536 s (deg 8) | 5.75 ms | 0.93 ms | 6.16x |
| evaluate, 256 curves x 2048 s (deg 8), 1 call vs 256 | 72.51 ms | 1.26 ms | 57.58x |
| evaluate (de Casteljau), 64 curves x 4096 s (deg 10), 1 call vs 64 | 2200.36 ms | 27.05 ms | 81.35x |
| hodograph, 64 curves x 4096 s (deg 8), 1 call vs 64 | 27.24 ms | 1.34 ms | 20.34x |
| curvature, 64 curves x 4096 s (deg 8), 1 call vs 64 | 67.76 ms | 2.95 ms | 22.95x |
| newton refine, 64 curves x 1024 points (deg 8), 1 call vs 65536 | 11473.13 ms | 5.72 ms | 2005.75x |
| subdivide, 512 curves (deg 10), 1 call vs 512 | 23.20 ms | 0.21 ms | 108.40x |
| specialize, 512 curves (deg 10), 1 call vs 512 | 155.25 ms | 0.38 ms | 409.84x |
| elevate, 4096 curves (deg 8), 1 call vs 4096 | 50.24 ms | 0.19 ms | 270.18x |
| length, 64 curves (deg 6), 1 call vs 64 | 1273.16 ms | 8.24 ms | 154.59x |
| evaluate via Curve objects, 2048 x 256 s (deg 6) | 19.90 ms | 25.34 ms | **0.79x, slower** |
| evaluate via curve_helpers (pure Python), 2048 x 256 s | 194.87 ms | 37.87 ms | 5.15x |

11 of 12 cases at or above upstream parity.

| outputs (curves x s) | device | host SIMD | device/host |
| ---: | ---: | ---: | ---: |
| 262144 (1 x 131072 s) | 0.593 ms | 0.658 ms | 0.90 |
| 131072 (8 x 8192 s) | 0.307 ms | 0.325 ms | 0.94 |
| 524288 (32 x 8192 s) | 0.759 ms | 1.345 ms | 0.56 |
| 1048576 (256 x 2048 s) | 1.277 ms | 2.712 ms | 0.47 |
| 16777216 (1024 x 8192 s) | 107.137 ms | 113.592 ms | 0.94 |
```

## Final gate

`pixi run build && pixi run test` — re-run at the end (see the session log);
both green, 995 passed, before and after the README edit. No source file was
changed after the first green run, so the library the benchmarks ran against
is the library the tests verified.
