"""
Layer Normalization (RISC-V ELF)
================================

The three kernels of 05-layer-norm.py, compiled for riscv64, deployed to the
board and checked there against torch's layer norm and its autograd:

* _layer_norm_fwd_fused: y, mean and 1/std,
* _layer_norm_bwd_dx_fused: dx, and the per-lock-group partial sums of dw/db
  (accumulated under a spin lock),
* _layer_norm_bwd_dwdb: the final dw/db from the partial sums.

The board runner launches one kernel per executable, so each run gets the
inputs the previous kernel would have produced, computed with torch.
"""

import os

import torch

import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(DX,  # pointer to the input gradient
                             DY,  # pointer to the output gradient
                             DW,  # pointer to the partial sum of weights gradient
                             DB,  # pointer to the partial sum of biases gradient
                             X,  # pointer to the input
                             W,  # pointer to the weights
                             Mean,  # pointer to the mean
                             Rstd,  # pointer to the 1/std
                             Lock,  # pointer to the lock
                             stride,  # how much to increase the pointer when moving by 1 row
                             N,  # number of columns in X
                             GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(DW,  # pointer to the partial sum of weights gradient
                         DB,  # pointer to the partial sum of biases gradient
                         FINAL_DW,  # pointer to the weights gradient
                         FINAL_DB,  # pointer to the biases gradient
                         M,  # GROUP_SIZE_M
                         N,  # number of columns
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


# %%
# Unit Test
# ---------
#
# An irregular number of columns, so the masked tail of each row is tested.

torch.manual_seed(0)
triton.runtime.driver.set_active_to_cpu()

M, N = 64, 500
eps = 1e-5
BLOCK_SIZE = triton.next_power_of_2(N)
GROUP_SIZE_M = 8
DWDB_BLOCK_SIZE_M, DWDB_BLOCK_SIZE_N = 32, 128

x = torch.randn(M, N, dtype=torch.float32, requires_grad=True)
weight = torch.rand(N, dtype=torch.float32, requires_grad=True)
bias = torch.rand(N, dtype=torch.float32, requires_grad=True)
dy = 0.1 * torch.randn(M, N, dtype=torch.float32)

y = torch.nn.functional.layer_norm(x, (N, ), weight, bias, eps)
y.backward(dy)
with torch.no_grad():
    mean = x.mean(dim=1)
    rstd = 1 / torch.sqrt(x.var(dim=1, unbiased=False) + eps)
    xhat = (x - mean[:, None]) * rstd[:, None]
    # Partial sums of dw/db per lock group: row r goes to group r % GROUP_SIZE_M.
    partial_dw = torch.zeros(GROUP_SIZE_M, N)
    partial_db = torch.zeros(GROUP_SIZE_M, N)
    for g in range(GROUP_SIZE_M):
        partial_dw[g] = (dy * xhat)[g::GROUP_SIZE_M].sum(dim=0)
        partial_db[g] = dy[g::GROUP_SIZE_M].sum(dim=0)

from triton.backends.cpu.riscv import compile_deploy_and_run

RISCV_HOST = "chlee@140.114.78.64"
RISCV_REMOTE_DIR = "~/triton-riscv-elf"


def flat(t):
    return t.detach().flatten().tolist()


def run_on_board(kernel, arguments, grid, constexprs, expected, name, atol=1e-4, signature=None, repeatable=True):
    # The runner launches the kernel many times before it checks the result;
    # a kernel that accumulates into its output must only run once.
    saved = {var: os.environ.get(var) for var in ("TRITON_BENCH_ITERS", "TRITON_BENCH_WARMUP")}
    if not repeatable:
        os.environ["TRITON_BENCH_ITERS"] = "0"
        os.environ["TRITON_BENCH_WARMUP"] = "0"
    try:
        result = compile_deploy_and_run(kernel, arguments, grid, f"artifacts/riscv/{name}.elf", RISCV_HOST,
                                         constexprs=constexprs, signature=signature, expected=expected, atol=atol,
                                         remote_dir=RISCV_REMOTE_DIR)
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
    print(f"{name}:")
    print(result.stdout, end="")
    print(result.stderr, end="")


run_on_board(
    _layer_norm_fwd_fused, {
        "X": flat(x), "Y": [0.0] * (M * N), "W": flat(weight), "B": flat(bias),  #
        "Mean": [0.0] * M, "Rstd": [0.0] * M, "stride": N, "N": N, "eps": eps,
    }, (M, ), {"BLOCK_SIZE": BLOCK_SIZE}, {"Y": flat(y), "Mean": flat(mean), "Rstd": flat(rstd)},
    "rvv-layer-norm-fwd")

# run_on_board(
#     _layer_norm_bwd_dx_fused, {
#         "DX": [0.0] * (M * N), "DY": flat(dy), "DW": [0.0] * (GROUP_SIZE_M * N), "DB": [0.0] * (GROUP_SIZE_M * N),  #
#         "X": flat(x), "W": flat(weight), "Mean": flat(mean), "Rstd": flat(rstd),  #
#         "Lock": [0] * (2 * GROUP_SIZE_M), "stride": N, "N": N,
#     }, (M, ), {"GROUP_SIZE_M": GROUP_SIZE_M, "BLOCK_SIZE_N": BLOCK_SIZE},
#     {"DX": flat(x.grad), "DW": flat(partial_dw), "DB": flat(partial_db)}, "rvv-layer-norm-bwd-dx",
#     signature={"Lock": "*i32"}, repeatable=False)

# run_on_board(
#     _layer_norm_bwd_dwdb, {
#         "DW": flat(partial_dw), "DB": flat(partial_db), "FINAL_DW": [0.0] * N, "FINAL_DB": [0.0] * N,  #
#         "M": GROUP_SIZE_M, "N": N,
#     }, (triton.cdiv(N, DWDB_BLOCK_SIZE_N), ), {"BLOCK_SIZE_M": DWDB_BLOCK_SIZE_M, "BLOCK_SIZE_N": DWDB_BLOCK_SIZE_N},
#     {"FINAL_DW": flat(weight.grad), "FINAL_DB": flat(bias.grad)}, "rvv-layer-norm-bwd-dwdb")
