import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

_LIB = _ROOT / "dist" / "libmojo-bezier-curves.so"

if not _LIB.exists():
    pytest.skip(
        "libmojo-bezier-curves.so not built; run `pixi run build` first",
        allow_module_level=True,
    )

upstream_curve_helpers = pytest.importorskip(
    "bezier.hazmat.curve_helpers",
    reason="the upstream `bezier` package is the parity reference",
)

#: Looser than machine epsilon because Mojo contracts a multiply and an add
#: into a single FMA where NumPy does not, so the two round differently even
#: when the arithmetic is the same. A real algorithmic difference shows up at
#: 1e-3 or worse, so this still fails loudly.
RTOL = 1e-12
ATOL = 1e-13


@pytest.fixture(scope="session")
def up():
    """The upstream ``bezier.hazmat.curve_helpers`` module."""
    return upstream_curve_helpers


def curve_family(count, dim, degree, seed=0, scale=1.0):
    """``count`` random curves, Fortran ordered per curve, as `bezier` stores."""
    rng = np.random.default_rng(seed)
    nodes = rng.random((count, dim, degree + 1)) * scale
    return [np.asfortranarray(curve) for curve in nodes]


#: Curves with shapes worth covering: a line, a quadratic, a cubic, a planar
#: cusp, and a spatial curve.
NAMED_NODES = {
    "line": np.asfortranarray([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
    "planar_line": np.asfortranarray([[0.0, 1.0], [0.0, 2.0]]),
    "quadratic": np.asfortranarray([[0.0, 0.625, 1.0], [0.0, 0.5, 0.5]]),
    "cubic": np.asfortranarray([[0.0, 0.375, 0.75, 1.0], [0.0, 0.75, 0.375, 0.0]]),
    "cusp": np.asfortranarray([[6.0, -2.0, -2.0, 6.0], [-3.0, 3.0, -3.0, 3.0]]),
    "spatial": np.asfortranarray(
        [[0.0, 1.0, 2.0, 3.0], [0.0, -1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    ),
    "high_degree": np.asfortranarray(
        [np.linspace(0.0, 1.0, 9), np.sin(np.linspace(0.0, 3.0, 9))]
    ),
}
