"""Bezier curve kernels, ported from the `bezier` package.

Source of truth: `bezier/hazmat/curve_helpers.py` of
https://github.com/dhermes/bezier (release 2024.6.20). The functions below are
emitted in that file's source order and keep its names, argument order and
branch structure, so this file can be read against the upstream one:

    upstream                        here
    ---------------------------     ------------------------------
    make_subdivision_matrices       bc_make_subdivision_matrices
    subdivide_nodes                 bc_subdivide_nodes
    evaluate_multi_vs               bc_evaluate_multi_vs
    evaluate_multi_de_casteljau     bc_evaluate_multi_de_casteljau
    vec_size                        bc_vec_size
    compute_length                  bc_compute_length
    elevate_nodes                   bc_elevate_nodes
    de_casteljau_one_round          bc_de_casteljau_one_round
    specialize_curve                bc_specialize_curve
    evaluate_hodograph              bc_evaluate_hodograph
    get_curvature                   bc_get_curvature
    newton_refine                   bc_newton_refine

Three upstream functions are not kernels and live in the Python layer, which is
where their control flow belongs: `evaluate_multi` and
`evaluate_multi_barycentric` are a degree dispatch between the two evaluation
routines above, and `locate_point` grows a list of candidate sub-curves.

Divergences forced by the FFI, all of them in the calling convention rather than
in the mathematics:

  * Every exported symbol takes buffer addresses as plain `Int` values and
    rebuilds the pointer inside the body, because `@export` rejects parametric
    functions and an inferred pointer origin would make the symbol parametric.
  * Every routine is batched over a leading curve index, because upstream
    evaluates one curve per Python call. A single curve is the same call with
    `num_curves == 1`.
  * `bc_compute_length` integrates with composite Simpson rather than QUADPACK
    (an adaptive Python integrator with a SciPy dependency), so the panel count
    is an argument instead of a hidden tolerance.

Layout: a family of curves is `(num_curves, dim, num_nodes)` C-contiguous
float64, so element `(d, j)` of curve `b` is at `b * dim * width + d * width + j`
where `width == num_nodes`. An evaluation over a parameter grid is
`(num_curves, dim, num_vals)` with element `(d, m)` at
`b * dim * num_vals + d * num_vals + m`. The coordinate dimension comes before the
varying one in both, so those two strides are the whole layout; getting them the
wrong way round gives transposed output rather than garbage.

Scratch space is passed in as one buffer addressed by explicit index offsets:
`Pointer.__add__` is deprecated in this dialect, and a named offset per region
makes an overlap between two live regions a compile-time-visible fact.

The evaluation kernels are the hot ones and are built for it three ways.
`vs_block` runs the VS recurrence `W` parameters at a time -- the parameter
index is the contiguous one in every batched layout here -- keeping the
accumulator and the `lambda2` power in registers and taking the binomial
coefficients from a table rather than a division per block; the leftover
`num_vals % W` parameters go through `vs_recurrence`, the same arithmetic one
parameter at a time. The binomial table itself lives in the `degree + 1`
doubles after the output, which the Python layer allocates as part of it.
At `GPU_MIN_OUTPUTS` outputs or more the batch runs on the GPU instead, in one
launch with the data copied in and out around it; the arithmetic there is the
scalar recurrence in the scalar order, so the device result is the host
result bit for bit.
"""

from std.math import sqrt
from std.sys import simd_width_of

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.ffi import c_size_t

#: Outputs below which the device path is not used. A launch plus the two
#: copies costs about 0.15 ms whatever the batch size, which the SIMD path
#: beats until the result is this big; see `vs_gpu` for the measurements.
comptime GPU_MIN_OUTPUTS = 1 << 17
#: Ceiling on the device allocation, so a caller cannot make this library ask
#: the shared device for more than 2 GiB.
comptime GPU_MAX_BYTES = 1 << 31
comptime GPU_BLOCK_DIM = 256

#: Free device memory below which the device is left alone. The box's GPU is
#: shared with production work, and a launch that has to fight an allocation
#: failure is slower than the host path it replaced.
comptime GPU_MIN_FREE_BYTES = 4000 * 1024 * 1024

comptime W = simd_width_of[DType.float64]()
comptime VS = SIMD[DType.float64, W]

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)

def fp_offset(addr: Int, offset: Int) -> FPtr:
    """A pointer `offset` doubles past `addr`; `Pointer.__add__` is deprecated."""
    return FPtr(unsafe_from_address=addr + offset * 8)


# ---------------------------------------------------------------------------
# make_subdivision_matrices / subdivide_nodes
# ---------------------------------------------------------------------------


@export("bc_make_subdivision_matrices")
def bc_make_subdivision_matrices(
    degree: Int, left_addr: Int, right_addr: Int
) abi("C"):
    """The matrices that convert a curve's nodes into its two halves.

    `left[0, 0] = 1`, then column by column each entry is half the entry to its
    left plus half the entry above-left; the right matrix is the same columns read
    back to front, which is why upstream calls them symmetric enough not to
    bother reversing them. Both are `(degree + 1, degree + 1)`, row `i` and
    column `j` at `i * width + j`.
    """
    var left = fp(left_addr)
    var right = fp(right_addr)
    var width = degree + 1
    for i in range(width):
        for j in range(width):
            left[unsafe_offset=i * width + j] = 0.0
            right[unsafe_offset=i * width + j] = 0.0
    left[unsafe_offset=0] = 1.0
    right[unsafe_offset=degree * width + degree] = 1.0
    for col in range(1, width):
        for row in range(col):
            left[unsafe_offset=row * width + col] = (
                0.5 * left[unsafe_offset=row * width + col - 1]
            )
        for row in range(1, col + 1):
            left[unsafe_offset=row * width + col] += (
                0.5 * left[unsafe_offset=(row - 1) * width + col - 1]
            )
        var complement = degree - col
        for row in range(col + 1):
            right[unsafe_offset=(degree - col + row) * width + complement] = left[
                unsafe_offset=row * width + col
            ]


@export("bc_subdivide_nodes")
def bc_subdivide_nodes(
    nodes_addr: Int, left_mat_addr: Int, right_mat_addr: Int, num_curves: Int,
    dim: Int, degree: Int, left_addr: Int, right_addr: Int
) abi("C"):
    """`left = nodes @ left_mat` and `right = nodes @ right_mat`.

    Upstream ships hard-coded subdivision matrices for degrees 1, 2 and 3 and
    builds them for higher degrees; the entries are the same dyadic numbers
    either way, so this always uses the built form. The outputs are
    `(num_curves, dim, degree + 1)`.
    """
    var nodes = fp(nodes_addr)
    var left_mat = fp(left_mat_addr)
    var right_mat = fp(right_mat_addr)
    var left = fp(left_addr)
    var right = fp(right_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        for d in range(dim):
            for j in range(width):
                var acc = Float64(0.0)
                for k in range(width):
                    acc += nodes[unsafe_offset=base + d * width + k] * left_mat[
                        unsafe_offset=k * width + j
                    ]
                left[unsafe_offset=base + d * width + j] = acc
                acc = Float64(0.0)
                for k in range(width):
                    acc += nodes[unsafe_offset=base + d * width + k] * right_mat[
                        unsafe_offset=k * width + j
                    ]
                right[unsafe_offset=base + d * width + j] = acc


# ---------------------------------------------------------------------------
# evaluate_multi_vs / evaluate_multi_de_casteljau
# ---------------------------------------------------------------------------


def vs_recurrence(
    nodes: FPtr, node_offset: Int, width: Int, lambda1: Float64,
    lambda2: Float64, dst: FPtr, dst_offset: Int, dst_stride: Int, dim: Int,
    degree: Int
):
    """The VS (modified Horner) recurrence behind `evaluate_multi_vs`.

    `result = lambda1 * v_0`; then for `j = 1 .. degree - 1`,
    `result = lambda1 * (result + C(degree, j) lambda2^j v_j)`; then
    `result + lambda2^degree v_degree`. The binomial coefficient is carried by
    `C(n, j) = C(n, j - 1) (n - j + 1) / j` rather than recomputed, which is
    what makes this the published VS algorithm and not an accident. The
    coefficient and the `lambda2` power are carried across the coordinates,
    so the node loop stays innermost. Coordinate `d` of the result lands at
    `dst[dst_offset + d * dst_stride]`.
    """
    for d in range(dim):
        dst[unsafe_offset=dst_offset + d * dst_stride] = (
            lambda1 * nodes[unsafe_offset=node_offset + d * width]
        )
    var binom_val = Float64(1.0)
    var lambda2_pow = Float64(1.0)
    for index in range(1, degree):
        lambda2_pow *= lambda2
        binom_val = binom_val * Float64(degree - index + 1) / Float64(index)
        for d in range(dim):
            var acc = dst[unsafe_offset=dst_offset + d * dst_stride]
            acc = acc + binom_val * lambda2_pow * nodes[
                unsafe_offset=node_offset + d * width + index
            ]
            dst[unsafe_offset=dst_offset + d * dst_stride] = lambda1 * acc
    for d in range(dim):
        var acc = dst[unsafe_offset=dst_offset + d * dst_stride]
        dst[unsafe_offset=dst_offset + d * dst_stride] = (
            acc + lambda2 * lambda2_pow * nodes[
                unsafe_offset=node_offset + d * width + degree
            ]
        )


def fill_binom(binom: FPtr, degree: Int):
    """`binom[j] = C(degree, j)`, by the recurrence `vs_recurrence` walks."""
    binom[unsafe_offset=0] = 1.0
    for j in range(1, degree + 1):
        binom[unsafe_offset=j] = (
            binom[unsafe_offset=j - 1] * Float64(degree - j + 1) / Float64(j)
        )


def vs_block(
    src: FPtr, node_offset: Int, binom: FPtr, lambda1: VS, lambda2: VS,
    degree: Int, scale: Float64
) -> VS:
    """`vs_recurrence` for `W` parameters at once, in registers.

    The parameter index is the contiguous one in every batched layout here, so
    a whole vector of `W` parameters runs the same recurrence: the
    accumulator and the `lambda2` power never leave registers, the binomial
    coefficients come from `binom` rather than a division per block, and the
    result is one contiguous store instead of a load and a store per node.
    """
    var acc = lambda1 * src[unsafe_offset=node_offset]
    var lambda2_pow = VS(1.0)
    var index = 1
    while index < degree:
        lambda2_pow = lambda2_pow * lambda2
        acc = lambda1 * (
            acc
            + binom[unsafe_offset=index] * lambda2_pow
            * src[unsafe_offset=node_offset + index]
        )
        index += 1
    return scale * (
        acc
        + lambda2 * lambda2_pow * src[unsafe_offset=node_offset + degree]
    )


def vs_gpu_kernel(
    nodes: FPtr, s_vals: FPtr, dst: FPtr, num_curves: Int32, dim: Int32,
    degree: Int32, num_vals: Int32
):
    """One output per thread: `vs_recurrence` with `lambda1 = 1 - s`.

    The grid covers every output and the threads past the end return, so the
    tail of a grid that is not a multiple of the block size is the threads
    that do not exist rather than a silent gap. The arithmetic is the scalar
    recurrence in the scalar order, so the device result is the host result
    bit for bit, not a faster approximation of it.
    """
    var gid = Int(global_idx.x)
    var total = Int(num_curves) * Int(dim) * Int(num_vals)
    if gid >= total:
        return
    var per_curve = Int(dim) * Int(num_vals)
    var curve = gid // per_curve
    var rest = gid - curve * per_curve
    var d = rest // Int(num_vals)
    var m = rest - d * Int(num_vals)
    var s = s_vals[unsafe_offset=m]
    var lambda1 = 1.0 - s
    var width = Int(degree) + 1
    var offset = curve * Int(dim) * width + d * width
    var acc = lambda1 * nodes[unsafe_offset=offset]
    var lambda2_pow = Float64(1.0)
    var binom_val = Float64(1.0)
    for index in range(1, Int(degree)):
        lambda2_pow *= s
        binom_val = (
            binom_val * Float64(degree - Int32(index) + 1) / Float64(index)
        )
        acc = lambda1 * (
            acc + binom_val * lambda2_pow * nodes[unsafe_offset=offset + index]
        )
    dst[unsafe_offset=gid] = (
        acc + s * lambda2_pow * nodes[unsafe_offset=offset + Int(degree)]
    )


def vs_gpu(
    nodes: FPtr, s_vals: FPtr, dst: FPtr, num_curves: Int, dim: Int,
    degree: Int, num_vals: Int
) -> Bool:
    """The whole batch on the device, in one launch and two copies.

    False when the batch is not worth a launch, when it would need more
    device memory than the cap, when the device is busy with someone else's
    work, or when there is no device to run on; the caller then runs the SIMD
    path instead. Measured on an RTX 5090 against the SIMD path: break-even
    at about 65536 outputs, 1.3x at 131072, 2.7x at 1048576 and 3.4x at
    16777216, so the threshold is where the device is clearly ahead rather
    than where it merely ties.
    """
    var outputs = num_curves * dim * num_vals
    if outputs < GPU_MIN_OUTPUTS or outputs * 8 > GPU_MAX_BYTES:
        return False
    try:
        var ctx = DeviceContext()
        var (free, total) = ctx.get_memory_info()
        if free < c_size_t(GPU_MIN_FREE_BYTES):
            return False
        var width = degree + 1
        var d_nodes = ctx.enqueue_create_buffer[DType.float64](
            num_curves * dim * width
        )
        var d_s = ctx.enqueue_create_buffer[DType.float64](num_vals)
        var d_dst = ctx.enqueue_create_buffer[DType.float64](outputs)
        ctx.enqueue_copy(d_nodes, nodes)
        ctx.enqueue_copy(d_s, s_vals)
        ctx.enqueue_function[vs_gpu_kernel](
            d_nodes, d_s, d_dst, Int32(num_curves), Int32(dim), Int32(degree),
            Int32(num_vals),
            grid_dim=Int32((outputs + GPU_BLOCK_DIM - 1) // GPU_BLOCK_DIM),
            block_dim=Int32(GPU_BLOCK_DIM),
        )
        ctx.enqueue_copy(dst, d_dst)
        ctx.synchronize()
        return True
    except:
        return False


def vs_eval(
    nodes: FPtr, lambda1: FPtr, lambda2: FPtr, num_curves: Int, dim: Int,
    degree: Int, num_vals: Int, dst: FPtr, binom: FPtr, complement: Bool
):
    """The VS recurrence over a whole batch of curves and parameters.

    The result is `(num_curves, dim, num_vals)`, so the parameter index is the
    contiguous one and `W` parameters go through the recurrence at a time.
    `complement` takes the first weight as `1 - lambda2` per block rather than
    reading a second grid, which is what the `s`-parameter entry point wants:
    the subtraction is one vector op on data already in registers, against a
    temporary array, a pass over it and a second address on the host.
    """
    var width = degree + 1
    for curve in range(num_curves):
        var base = curve * dim * width
        var out = curve * dim * num_vals
        var m = 0
        while m + W <= num_vals:
            var l2 = lambda2.unsafe_load[width=W](m)
            var l1 = lambda1.unsafe_load[width=W](m)
            if complement:
                l1 = 1.0 - l2
            for d in range(dim):
                dst.unsafe_store(
                    out + d * num_vals + m,
                    vs_block(
                        nodes, base + d * width, binom, l1, l2, degree, 1.0
                    ),
                )
            m += W
        while m < num_vals:
            var l2 = lambda2[unsafe_offset=m]
            var l1 = lambda1[unsafe_offset=m]
            if complement:
                l1 = 1.0 - l2
            vs_recurrence(
                nodes, base, width, l1, l2, dst, out + m, num_vals, dim, degree
            )
            m += 1


@export("bc_evaluate_multi_vs")
def bc_evaluate_multi_vs(
    nodes_addr: Int, lambda1_addr: Int, lambda2_addr: Int, num_curves: Int,
    dim: Int, degree: Int, num_vals: Int, dst_addr: Int
) abi("C"):
    """Evaluate the Bezier type function with the VS algorithm.

    `lambda1` and `lambda2` are parallel `(num_vals,)` grids; `dst` is
    `(num_curves, dim, num_vals)` with `degree + 1` spare doubles after it,
    which the caller allocates and this writes the binomial row into. The
    parameter index is the contiguous one, so `W` parameters at a time go
    through the recurrence in registers and the leftover ones go through
    `vs_recurrence`, which is the same arithmetic one parameter at a time.
    """
    var binom = fp_offset(dst_addr, num_curves * dim * num_vals)
    fill_binom(binom, degree)
    vs_eval(
        fp(nodes_addr), fp(lambda1_addr), fp(lambda2_addr), num_curves, dim,
        degree, num_vals, fp(dst_addr), binom, False
    )


@export("bc_evaluate_multi_s")
def bc_evaluate_multi_s(
    nodes_addr: Int, s_addr: Int, num_curves: Int, dim: Int, degree: Int,
    num_vals: Int, dst_addr: Int
) abi("C") -> Int:
    """`B(s)` on a parameter grid, with `lambda1 = 1 - lambda2` read as `s`.

    Same arithmetic and same output as `bc_evaluate_multi_vs` with
    `lambda1 == 1 - lambda2`, one grid and one address narrower. A batch big
    enough to be worth a launch goes to the device instead, once, with the
    data moved in and out around it; the return value is 1 when it did, so a
    caller can tell a device run from a host one that fell back.
    """
    var nodes = fp(nodes_addr)
    var s_vals = fp(s_addr)
    var dst = fp(dst_addr)
    if vs_gpu(nodes, s_vals, dst, num_curves, dim, degree, num_vals):
        return 1
    var binom = fp_offset(dst_addr, num_curves * dim * num_vals)
    fill_binom(binom, degree)
    vs_eval(
        nodes, s_vals, s_vals, num_curves, dim, degree, num_vals, dst, binom,
        True
    )
    return 0


@export("bc_evaluate_multi_de_casteljau")
def bc_evaluate_multi_de_casteljau(
    nodes_addr: Int, lambda1_addr: Int, lambda2_addr: Int, work_addr: Int,
    num_curves: Int, dim: Int, degree: Int, num_vals: Int, dst_addr: Int
) abi("C"):
    """Evaluate the Bezier type function through the de Casteljau triangle.

    One triangle is built per (curve, parameter) in `work`, which needs
    `dim * degree` doubles. Upstream keeps the whole `(dimension, num_vals,
    degree)` cube and contracts it a level at a time; the level-by-level
    contraction is the same arithmetic, done once per parameter instead of once
    per parameter and level. A degree 0 curve is its own value, so there is no
    triangle to build; upstream raises on that shape, and the Python layer
    raises before getting here, but the kernel must not read `work` to find
    out. `dst` is `(num_curves, dim, num_vals)`.
    """
    var nodes = fp(nodes_addr)
    var lambda1 = fp(lambda1_addr)
    var lambda2 = fp(lambda2_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    if degree == 0:
        for b in range(num_curves):
            for d in range(dim):
                for m in range(num_vals):
                    dst[unsafe_offset=b * dim * num_vals + d * num_vals + m] = (
                        nodes[unsafe_offset=b * dim + d]
                    )
        return
    for b in range(num_curves):
        var base = b * dim * width
        var out = b * dim * num_vals
        for m in range(num_vals):
            var l1 = lambda1[unsafe_offset=m]
            var l2 = lambda2[unsafe_offset=m]
            for d in range(dim):
                for i in range(degree):
                    work[unsafe_offset=d * degree + i] = (
                        l1 * nodes[unsafe_offset=base + d * width + i]
                        + l2 * nodes[unsafe_offset=base + d * width + i + 1]
                    )
            var index = degree - 1
            while index > 0:
                for d in range(dim):
                    for i in range(index):
                        work[unsafe_offset=d * degree + i] = (
                            l1 * work[unsafe_offset=d * degree + i]
                            + l2 * work[unsafe_offset=d * degree + i + 1]
                        )
                index -= 1
            for d in range(dim):
                dst[unsafe_offset=out + d * num_vals + m] = work[
                    unsafe_offset=d * degree
                ]


# ---------------------------------------------------------------------------
# vec_size
# ---------------------------------------------------------------------------


@export("bc_vec_size")
def bc_vec_size(
    nodes_addr: Int, lambda1_addr: Int, lambda2_addr: Int, work_addr: Int,
    num_curves: Int, dim: Int, degree: Int, num_vals: Int, dst_addr: Int
) abi("C"):
    """`||B(s)||_2` for each (curve, parameter), result `(num_curves, num_vals)`.

    `work` needs `dim * (degree + 1)` doubles and holds one evaluation's nodes.
    """
    var nodes = fp(nodes_addr)
    var lambda1 = fp(lambda1_addr)
    var lambda2 = fp(lambda2_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        for m in range(num_vals):
            vs_recurrence(
                nodes, base, width, lambda1[unsafe_offset=m],
                lambda2[unsafe_offset=m], work, 0, width, dim, degree
            )
            var acc = Float64(0.0)
            for d in range(dim):
                acc += work[unsafe_offset=d * width] * work[unsafe_offset=d * width]
            dst[unsafe_offset=b * num_vals + m] = sqrt(acc)


# ---------------------------------------------------------------------------
# compute_length
# ---------------------------------------------------------------------------


@export("bc_compute_length")
def bc_compute_length(
    nodes_addr: Int, work_addr: Int, num_curves: Int, dim: Int, degree: Int,
    panels: Int, dst_addr: Int
) abi("C"):
    """Arc length as the integral of `||B'(s)||_2` over `[0, 1]`.

    A single node is 0.0 and two nodes are the exact distance between them, as
    upstream returns. Above that upstream hands the integral to QUADPACK; this
    uses composite Simpson over `panels` subintervals, rounded up to an even
    count, and re-evaluates the hodograph at every panel so the memory
    footprint stays at `O(degree)`. `work` needs `(degree + dim) * (degree + 1)`
    doubles: the hodograph nodes, then the evaluated tangent beside them.
    `dst` is `(num_curves,)`.
    """
    var nodes = fp(nodes_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        if degree == 0:
            dst[unsafe_offset=b] = 0.0
            continue
        if degree == 1:
            var acc = Float64(0.0)
            for d in range(dim):
                var delta = Float64(degree) * (
                    nodes[unsafe_offset=base + d * width + 1]
                    - nodes[unsafe_offset=base + d * width]
                )
                acc += delta * delta
            dst[unsafe_offset=b] = sqrt(acc)
            continue
        # first_deriv = (num_nodes - 1) * (nodes[:, 1:] - nodes[:, :-1])
        for d in range(dim):
            for j in range(degree):
                work[unsafe_offset=d * degree + j] = Float64(degree) * (
                    nodes[unsafe_offset=base + d * width + j + 1]
                    - nodes[unsafe_offset=base + d * width + j]
                )
        var intervals = panels
        if intervals < 2:
            intervals = 2
        intervals = (intervals + 1) // 2 * 2
        var h = 1.0 / Float64(intervals)
        var total = Float64(0.0)
        for i in range(intervals + 1):
            vs_recurrence(
                work, 0, degree, 1.0 - Float64(i) * h, Float64(i) * h, work,
                degree * width, width, dim, degree - 1
            )
            var acc = Float64(0.0)
            for d in range(dim):
                var speed = work[unsafe_offset=degree * width + d * width]
                acc += speed * speed
            var value = sqrt(acc)
            if i == 0 or i == intervals:
                total += value
            elif i % 2 == 0:
                total += 2.0 * value
            else:
                total += 4.0 * value
        dst[unsafe_offset=b] = total * h / 3.0


# ---------------------------------------------------------------------------
# elevate_nodes
# ---------------------------------------------------------------------------


@export("bc_elevate_nodes")
def bc_elevate_nodes(
    nodes_addr: Int, num_curves: Int, dim: Int, degree: Int, dst_addr: Int
) abi("C"):
    """`w_0 = v_0`, `w_j = (j v_{j-1} + (n + 1 - j) v_j) / (n + 1)`, `w_{n+1} = v_n`.

    The division is held off until after the weighted sum, as upstream does to
    avoid round-off, and it is applied to the interior nodes only: the two
    boundaries are copies and dividing them would read whatever the caller's
    buffer happened to hold there. `dst` is `(num_curves, dim, degree + 2)`.
    """
    var nodes = fp(nodes_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    var new_width = width + 1
    var denominator = Float64(width)
    for b in range(num_curves):
        var base = b * dim * width
        var obase = b * dim * new_width
        # The boundaries are copies, not weighted sums; they never divide.
        for d in range(dim):
            dst[unsafe_offset=obase + d * new_width] = nodes[
                unsafe_offset=base + d * width
            ]
            dst[unsafe_offset=obase + d * new_width + width] = nodes[
                unsafe_offset=base + d * width + degree
            ]
        for j in range(1, width):
            var multiplier = Float64(j)
            for d in range(dim):
                dst[unsafe_offset=obase + d * new_width + j] = (
                    multiplier * nodes[unsafe_offset=base + d * width + j - 1]
                    + (denominator - multiplier)
                    * nodes[unsafe_offset=base + d * width + j]
                )
        for d in range(dim):
            for j in range(1, width):
                dst[unsafe_offset=obase + d * new_width + j] /= denominator


# ---------------------------------------------------------------------------
# de_casteljau_one_round
# ---------------------------------------------------------------------------


@export("bc_de_casteljau_one_round")
def bc_de_casteljau_one_round(
    nodes_addr: Int, num_curves: Int, dim: Int, degree: Int, lambda1: Float64,
    lambda2: Float64, dst_addr: Int
) abi("C"):
    """`lambda1 * nodes[:, :-1] + lambda2 * nodes[:, 1:]`, one row shorter.

    `dst` is `(num_curves, dim, degree)`; the weights are assumed to sum to one,
    which upstream states and does not check either.
    """
    var nodes = fp(nodes_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    for b in range(num_curves):
        var base = b * dim * width
        var out = b * dim * degree
        for d in range(dim):
            for i in range(degree):
                dst[unsafe_offset=out + d * degree + i] = (
                    lambda1 * nodes[unsafe_offset=base + d * width + i]
                    + lambda2 * nodes[unsafe_offset=base + d * width + i + 1]
                )


# ---------------------------------------------------------------------------
# specialize_curve
# ---------------------------------------------------------------------------


@export("bc_specialize_curve")
def bc_specialize_curve(
    nodes_addr: Int, work_addr: Int, num_curves: Int, dim: Int, degree: Int,
    start: Float64, end: Float64, dst_addr: Int
) abi("C"):
    """Re-parameterise `[start, end]` onto `[0, 1]`.

    Upstream builds a table of partial values keyed by which of the two weight
    pairs produced them, and reads the diagonal `(0, ..., 0, 1, ..., 1)` off it:
    column `j` is the first entry of the vector left after `degree - j` rounds at
    `(1 - start, start)` followed by `j` rounds at `(1 - end, end)`. That
    diagonal is walked here one row at a time -- `degree - j` rounds at the
    start weights, keep that row, then `j` rounds at the end weights collapse it
    to the single value that is column `j` -- which is the same recurrence in
    the same order, without the table.

    `work` needs `2 * dim * (degree + 1)` doubles: the row of partials at the
    start weights, then the copy of it the end weights consume. `dst` is
    `(num_curves, dim, degree + 1)`.
    """
    var nodes = fp(nodes_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    var start1 = 1.0 - start
    var end1 = 1.0 - end
    for b in range(num_curves):
        var base = b * dim * width
        for d in range(dim):
            for k in range(width):
                work[unsafe_offset=d * width + k] = nodes[
                    unsafe_offset=base + d * width + k
                ]
        # `dim * width .. 2 * dim * width` holds the copy the end weights eat.
        var roll = dim * width
        for k in range(width):
            for d in range(dim):
                for i in range(width):
                    work[unsafe_offset=roll + d * width + i] = work[
                        unsafe_offset=d * width + i
                    ]
            # `degree - k` rounds on a row of `width - k` entries leave one.
            for r in range(degree - k):
                for d in range(dim):
                    for i in range(width - k - 1 - r):
                        work[unsafe_offset=roll + d * width + i] = (
                            end1 * work[unsafe_offset=roll + d * width + i]
                            + end * work[unsafe_offset=roll + d * width + i + 1]
                        )
            for d in range(dim):
                dst[unsafe_offset=base + d * width + degree - k] = work[
                    unsafe_offset=roll + d * width
                ]
            for d in range(dim):
                for i in range(width - k - 1):
                    work[unsafe_offset=d * width + i] = (
                        start1 * work[unsafe_offset=d * width + i]
                        + start * work[unsafe_offset=d * width + i + 1]
                    )


# ---------------------------------------------------------------------------
# evaluate_hodograph
# ---------------------------------------------------------------------------


@export("bc_evaluate_hodograph")
def bc_evaluate_hodograph(
    nodes_addr: Int, s_addr: Int, work_addr: Int, num_curves: Int, dim: Int,
    degree: Int, num_s: Int, dst_addr: Int
) abi("C"):
    """`B'(s) = n sum_j C(d, j) s^j (1 - s)^(d - j) Delta v_j` on a grid.

    The forward differences are written unscaled, the type function is evaluated
    on them, and the factor `n` is applied to the result, which is the order
    upstream uses. `work` needs `dim * degree + dim + degree` doubles: the
    `dim * degree` unscaled forward differences, then `dim` for the scalar
    tail's per-coordinate accumulator, then `degree` for the binomial row.
    `dst` is `(num_curves, dim, num_s)`.
    """
    var nodes = fp(nodes_addr)
    var s_vals = fp(s_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    # Two scratch regions follow the differences. The first is where the scalar
    # tail runs the recurrence one coordinate at a time: `vs_recurrence` is
    # given `node_offset == dst_offset == 0` and a stride of 1, so it reads
    # node `d + index` and accumulator `d` of the same buffer, and the regions
    # have to be distinct. The second is the binomial row the vector body reads.
    var tail_offset = dim * degree
    var binom_offset = tail_offset + dim
    var binom = fp_offset(work_addr, binom_offset)
    fill_binom(binom, degree - 1)
    for b in range(num_curves):
        var base = b * dim * width
        var out = b * dim * num_s
        for d in range(dim):
            for j in range(degree):
                work[unsafe_offset=d * degree + j] = (
                    nodes[unsafe_offset=base + d * width + j + 1]
                    - nodes[unsafe_offset=base + d * width + j]
                )
        var m = 0
        while m + W <= num_s:
            var s = s_vals.unsafe_load[width=W](m)
            var l2 = s
            var l1 = 1.0 - s
            for d in range(dim):
                dst.unsafe_store(
                    out + d * num_s + m,
                    vs_block(
                        work, d * degree, binom, l1, l2, degree - 1,
                        Float64(degree),
                    ),
                )
            m += W
        while m < num_s:
            var s = s_vals[unsafe_offset=m]
            # The accumulator lives in scratch, never in the differences it is
            # built from: overlapping the two makes `vs_recurrence` overwrite
            # `Delta v_j` with the partial sum before it has been read, and the
            # result is silently wrong rather than out of bounds.
            vs_recurrence(
                work, 0, degree, 1.0 - s, s, work, tail_offset, 1, dim,
                degree - 1
            )
            for d in range(dim):
                dst[unsafe_offset=out + d * num_s + m] = Float64(degree) * work[
                    unsafe_offset=tail_offset + d
                ]
            m += 1


# ---------------------------------------------------------------------------
# get_curvature
# ---------------------------------------------------------------------------


@export("bc_get_curvature")
def bc_get_curvature(
    nodes_addr: Int, tangent_addr: Int, s_addr: Int, work_addr: Int,
    num_curves: Int, degree: Int, num_s: Int, dst_addr: Int
) abi("C"):
    """`(B'(s) x B''(s)) / ||B'(s)||^3` from an already computed tangent.

    `work` needs `2 * dim * degree + dim + degree` doubles: the unscaled first
    and second forward differences, then the binomial row. The evaluated
    concavity stays in registers, so it is never written back. `dst` is
    `(num_curves, num_s)`.
    """
    var nodes = fp(nodes_addr)
    var tangent = fp(tangent_addr)
    var s_vals = fp(s_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    # The cross product is planar, as it is upstream.
    var dim = 2
    var first = dim * degree
    # `2 * first .. 2 * first + dim` is where the scalar tail parks the
    # evaluated concavity, so the binomial row starts past it.
    var binom_offset = 2 * first + dim
    var scale = Float64(degree) * Float64(degree - 1)
    var binom = fp_offset(work_addr, binom_offset)
    fill_binom(binom, degree - 2)
    for b in range(num_curves):
        var base = b * dim * width
        if degree == 1:
            for m in range(num_s):
                dst[unsafe_offset=b * num_s + m] = 0.0
            continue
        for d in range(dim):
            for j in range(degree):
                work[unsafe_offset=d * degree + j] = (
                    nodes[unsafe_offset=base + d * width + j + 1]
                    - nodes[unsafe_offset=base + d * width + j]
                )
        for d in range(dim):
            for j in range(degree - 1):
                work[unsafe_offset=first + d * (degree - 1) + j] = (
                    work[unsafe_offset=d * degree + j + 1]
                    - work[unsafe_offset=d * degree + j]
                )
        var m = 0
        while m + W <= num_s:
            var s = s_vals.unsafe_load[width=W](m)
            var l2 = s
            var l1 = 1.0 - s
            var concavity_x = scale * vs_block(
                work, first, binom, l1, l2, degree - 2, 1.0
            )
            var concavity_y = scale * vs_block(
                work, first + degree - 1, binom, l1, l2, degree - 2, 1.0
            )
            var tangent_x = tangent.unsafe_load[width=W](b * 2 * num_s + m)
            var tangent_y = tangent.unsafe_load[width=W](b * 2 * num_s + num_s + m)
            var norm = sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
            dst.unsafe_store(
                b * num_s + m,
                (tangent_x * concavity_y - tangent_y * concavity_x)
                / (norm * norm * norm),
            )
            m += W
        while m < num_s:
            var s = s_vals[unsafe_offset=m]
            vs_recurrence(
                work, first, degree - 1, 1.0 - s, s, work, 2 * first, 1, 2,
                degree - 2
            )
            var concavity_x = scale * work[unsafe_offset=2 * first]
            var concavity_y = scale * work[unsafe_offset=2 * first + 1]
            var tangent_x = tangent[unsafe_offset=b * 2 * num_s + m]
            var tangent_y = tangent[unsafe_offset=b * 2 * num_s + num_s + m]
            var norm_squared = tangent_x * tangent_x + tangent_y * tangent_y
            var norm = sqrt(norm_squared)
            dst[unsafe_offset=b * num_s + m] = (
                (tangent_x * concavity_y - tangent_y * concavity_x)
                / (norm * norm * norm)
            )
            m += 1


# ---------------------------------------------------------------------------
# newton_refine
# ---------------------------------------------------------------------------


@export("bc_newton_refine")
def bc_newton_refine(
    nodes_addr: Int, points_addr: Int, s_addr: Int, work_addr: Int,
    num_curves: Int, num_points: Int, dim: Int, degree: Int, dst_addr: Int
) abi("C"):
    """One Newton step on `B(s) = p` for many (curve, point) pairs.

    `delta_s = (p - B(s)) . B'(s) / B'(s) . B'(s)`, returned as `s + delta_s`,
    with the residual and the derivative re-evaluated from the nodes at every
    step exactly as upstream does. Batching is the only change: upstream
    refines one point per call, so refining a grid of points on one curve costs
    a grid of calls.

    `points_addr` is `(num_curves, num_points, dim)` and `s_addr` is
    `(num_curves, num_points)`; `dst` is `(num_curves, num_points)`. `work`
    needs `dim * degree + dim * (degree + 1) + dim` doubles, in that order: the
    unscaled forward differences, the evaluation of the hodograph at `s`, and
    the evaluation of the curve at `s`.
    """
    var nodes = fp(nodes_addr)
    var points = fp(points_addr)
    var s_vals = fp(s_addr)
    var work = fp(work_addr)
    var dst = fp(dst_addr)
    var width = degree + 1
    var deriv_offset = dim * degree
    var point_offset = deriv_offset + dim * degree
    for b in range(num_curves):
        var base = b * dim * width
        if degree >= 1:
            for d in range(dim):
                for j in range(degree):
                    work[unsafe_offset=d * degree + j] = (
                        nodes[unsafe_offset=base + d * width + j + 1]
                        - nodes[unsafe_offset=base + d * width + j]
                    )
        for k in range(num_points):
            var pbase = (b * num_points + k) * dim
            var s = s_vals[unsafe_offset=b * num_points + k]
            vs_recurrence(
                nodes, base, width, 1.0 - s, s, work, point_offset, 1, dim,
                degree
            )
            var numerator = Float64(0.0)
            var denominator = Float64(0.0)
            if degree >= 1:
                vs_recurrence(
                    work, 0, degree, 1.0 - s, s, work, deriv_offset, 1, dim,
                    degree - 1
                )
                for d in range(dim):
                    var pt_delta = (
                        points[unsafe_offset=pbase + d]
                        - work[unsafe_offset=point_offset + d]
                    )
                    # The stored forward differences are unscaled, so the
                    # tangent is `degree` times what was evaluated.
                    var derivative = Float64(degree) * work[
                        unsafe_offset=deriv_offset + d
                    ]
                    numerator += pt_delta * derivative
                    denominator += derivative * derivative
            dst[unsafe_offset=b * num_points + k] = (
                s + numerator / denominator
            )
