# Tail masks on an NVIDIA GPU (GX10 / GB10)

Can the tail-mask codegen optimization from triton-cpu be applied on triton-GPU? To find out, the `mask2d` kernel was compiled and run on the GX10 (NVIDIA GB10, compute capability 12.1, upstream Triton `1ed999d6b`), and its TTGIR / PTX were inspected.

## The experiment

`~/mask2d_gpu/mask2d_gpu.py` on the GX10 is the same `mask2d_kernel` as `python/tutorials/cpu/rvv_mask2d_elf.py` (`C = alpha * A + B`, box mask `(rm[:, None] < M) & (rn[None, :] < N)`), run on CUDA tensors and checked with `torch.allclose`. Optional arguments `BLOCK_M BLOCK_N M N` default to `8 16 37 45`. `mask2d_gpu_stride.py` is a variant with padded rows. It was run with the already-built Triton in `~/triton/.venv` and `TRITON_KERNEL_DUMP=1`; the dumps are in `~/mask2d_gpu/dump*/`. All runs passed the check.

| run | layout per thread | loads in PTX | stores | `setp` | PTX instructions |
|---|---|---|---|---|---|
| 8×16 tile, `N = 45` (the RISC-V example) | 1 element | 2 × `ld.global.b32` | 1 × `st.global.b32` | 2 | 34 |
| 64×64, `N = 45`, stride 45 | `[1, 1]` | 64 × `ld.global.b32` | 32 × `st.global.b32` | 33 | 500 |
| 64×64, `N = 48`, stride 48 | `[1, 4]` | **16 × `ld.global.v4.b32`** | 8 × `st.global.v4.b32` | 9 | 237 |
| 64×64, `N = 45`, **stride 48** (aligned rows) | `[1, 4]` | **64 × `ld.global.b32`** | 32 × `st.global.b32` | 12 | 339 |

The TTGIR mask chain (`make_range` / `splat` / `expand_dims` / `cmpi` / `broadcast` / `andi`) is the same as in TTIR on the CPU side.

## What this shows

**1. A small tile: the mask is nearly free.** In the 8×16 case each thread handles one element, and the whole mask is 3 instructions:

```ptx
setp.lt.s32  %p2, %r15, %r8        // row < M
setp.lt.s32  %p3, %r18, %r9        // col < N
and.pred     %p1, %p2, %p3
mov.u32      %r1, %r2              // other = 0
@%p1 ld.global.b32 { %r1 }, [ %rd1 + 0 ]
```

Masked-off threads just skip the load. That's why EVL, the poison pass-through and the masked reduction don't carry over: a GPU has no vector-length register and no tail-undisturbed policy, so there's nothing comparable to save.

**2. A realistic tile: the tail mask blocks vector loads.** The last row of the table is the decisive one. With rows aligned (stride 48), Triton already lays out 4 contiguous elements per thread (`[1, 4]`), but it still emits **scalar** loads: 64 `ld.global.b32` instead of 16 `ld.global.v4`.

The cause is the mask `rn < 45`. Triton's axis analysis can only allow a 4-wide load if the mask has the same value across those 4 columns. That's provable when `N` is a multiple of 4 (`N = 48` gives `v4`), but not for `N = 45`. A single `ld.global.v4` can't be partly predicated, so every load in every tile drops to scalar, including tiles that are completely inside the matrix. It costs about 4× the load and store instructions (339 vs 237 total PTX instructions).

(The `N = 45`, stride 45 row also has unaligned row starts, so it can't vectorize for two reasons. The stride-48 run separates the two: aligned rows, and still scalar because of the mask.)

## Piece by piece

| part of the RISC-V work | on an NVIDIA GPU |
|---|---|
| **`AnalyzeTailMasks`** (recognize box masks, compute per-dimension bounds) | **Applies as is.** It's a pure TTIR analysis with nothing RISC-V-specific. |
| **`tail_mask` → EVL** (`vsetvli`, `vp.load` / `vp.store`) | **No equivalent.** Masks become per-thread predicates (`@%p ld.global…`). |
| **Poison pass-through** (`TailMaskToEVL`, `ta` vs `tu`) | **Little to gain.** `other=0` is one `mov` per register before a predicated load. |
| **Masked reduction** (`vector.mask { vector.reduction }` → `vp.reduce`) | **Little to gain.** GPU reductions are register adds plus warp shuffles. |
| **Row-wise mask from bounds** (`getTailRowMask`) | **Possibly useful**, together with the bounds below. |

## Where the tail-mask analysis could help on the GPU

The opportunity isn't EVL. It's using the **box bounds** to get vector loads back:
- **Versioning on the bounds:** `AnalyzeTailMasks` already computes the bounds as scalars (`M - rm`, `N - rn`). A GPU pass could add a runtime check, `if (M - rm >= BLOCK_M && N - rn >= BLOCK_N)`. In that case the whole tile is in bounds, the mask is all true, and unmasked `v4` loads can be used. Otherwise it would use today's masked path. Only edge tiles would pay for scalar loads. Today users do this by hand (the `EVEN_K` style of kernel).
- **TMA:** on Blackwell, a box mask with `other = 0` is exactly what a tensor-descriptor (TMA) load does in hardware: out-of-bounds elements are filled with zero. Converting such loads would also remove the mask math. This needs alignment and stride conditions, and whether upstream already does it automatically hasn't been checked.

Both are ideas based on these measurements, not implemented or timed. A next step would be a hand-written version of the first one (a fully-in-bounds fast path in the kernel), timed against the current kernel for a large ragged matrix on the GX10.

## Result: a hand-written fully-in-bounds fast path

To test the versioning idea, `~/mask2d_gpu/mask2d_gpu_bench.py` on the GX10 compares two kernels:
- **`mask2d_kernel`:** the current kernel, with the box mask on every tile.
- **`mask2d_fast_kernel`:** if the tile is fully inside the matrix (`pid_m*BM + BM <= M` and `pid_n*BN + BN <= N`, i.e. the box bounds `M - rm` and `N - rn` cover the tile), it uses unmasked loads and stores. Otherwise it falls back to the masked code.

```python
full = (pid_m * BLOCK_M + BLOCK_M <= M) & (pid_n * BLOCK_N + BLOCK_N <= N)
if full:
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(c_ptr + offs, alpha * a + b)
else:
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    ...  # the masked code
```

Both were timed with `triton.testing.do_bench` on a large ragged matrix (64×64 tiles), in two storage cases, plus a non-ragged matrix as the upper bound. All six runs gave correct results.

| matrix | kernel | time | effective bandwidth | PTX loads |
|---|---|---|---|---|
| `8192 × 8189`, stride **8192** (ragged, aligned rows) | masked | 3693.6 µs | 217.9 GB/s | 0 × `v4`, 64 × `b32` |
| | fast path | 3718.5 µs | 216.5 GB/s | **16 × `v4`** (fast branch) + 64 × `b32` (masked branch) |
| `8192 × 8189`, stride **8189** (ragged, contiguous rows) | masked | 4362.6 µs | 184.5 GB/s | 0 × `v4`, 64 × `b32` |
| | fast path | 4503.0 µs | 178.8 GB/s | 0 × `v4`, 128 × `b32` |
| `8192 × 8192` (not ragged, upper bound) | masked | 3694.2 µs | 218.0 GB/s | 16 × `v4` |
| | fast path | 3694.3 µs | 218.0 GB/s | 32 × `v4` |

The effective bandwidth counts reading A and B and writing C (`3 × M × N × 4` bytes). Each case was run once; the variation between timings wasn't measured.

**The fast path is not faster for this kernel.**
- **It does produce vector loads.** On the aligned ragged matrix, its in-bounds branch has 16 `ld.global.v4`, which the masked kernel never gets.
- **It doesn't matter for speed.** The masked kernel with scalar loads already runs at about 218 GB/s, the same as the non-ragged upper bound. This kernel is limited by memory bandwidth. With scalar loads, the 32 threads of a warp still read neighboring addresses, which the hardware combines into the same memory transactions as vector loads. Fewer load instructions save issue slots the kernel isn't short of.
- **Contiguous rows (stride 8189):** the fast path can't vectorize anyway, because the row starts aren't 16-byte aligned. It's slightly slower (about 3%), probably from the extra branch and the code duplicated into two paths.

**So for the tail-mask idea on the GPU:** in a memory-bound kernel like this, recovering vector loads from the box bounds buys nothing measurable. It could still matter where instruction issue is the bottleneck, e.g. compute-heavy kernels that also do many masked loads, or kernels that load data into shared memory. The TMA idea is a different case, because TMA moves data asynchronously and frees threads entirely. Neither was tested here.

## Result: TMA (tensor descriptors)

A tensor descriptor carries the matrix shape `[M, N]`. A TMA load fills out-of-bounds elements with zero and a TMA store skips them, which is the box mask with `other = 0` done in hardware. The kernel has no mask at all:

```python
a = a_desc.load([off_m, off_n])               # out-of-bounds elements read as 0
b = b_desc.load([off_m, off_n])
c_desc.store([off_m, off_n], alpha * a + b)   # out-of-bounds elements are not written
```

TMA requires 16-byte-aligned row strides, so it was tested on matrices with stride 8192 only; the contiguous stride-8189 case can't use it. All eight runs gave correct results. Each case was run once; the variation between timings wasn't measured.

### Test 1: one tile per program

`~/mask2d_gpu/mask2d_gpu_tma.py` compares the masked kernel with TMA versions that build the descriptors on the host (`TensorDescriptor.from_tensor`) or in the kernel (`tl.make_tensor_descriptor`, with a scratch allocator from `triton.set_allocator`). The TMA kernels have 3 `cp.async.bulk.tensor` in the PTX and no `ld.global`.

| `8192 × N`, stride 8192 | masked | TMA, host descriptor | TMA, device descriptor |
|---|---|---|---|
| `N = 8189` (ragged) | 3709.9 µs, **217.0 GB/s** | 4000.7 µs, 201.2 GB/s | 3907.7 µs, 206.0 GB/s |
| `N = 8192` | 3691.1 µs, **218.2 GB/s** | 3984.5 µs, 202.1 GB/s | 3874.6 µs, 207.8 GB/s |

TMA is 5–8% slower. With one tile per program, each program starts its copies and then waits for them. The asynchronous copy has nothing to overlap with, and data makes an extra trip through shared memory.

### Test 2: persistent and pipelined

`~/mask2d_gpu/mask2d_gpu_tma_persistent.py` uses one program per SM (48), each looping over the tiles with `tl.range(…, num_stages=3)`, so Triton's software pipeliner prefetches the next tiles' loads while the current tile is computed. This is the setting where TMA's asynchronous copies should pay off.

| `8192 × N`, stride 8192 | masked (persistent) | TMA (persistent) |
|---|---|---|
| `N = 8189` (ragged) | 3535.3 µs, **227.7 GB/s** | 3623.0 µs, 222.2 GB/s |
| `N = 8192` | 3474.2 µs, **231.8 GB/s** | 3644.1 µs, 221.0 GB/s |

Pipelining helps both: TMA goes from about 206 to 222 GB/s, and the masked kernel from 217 to 228 GB/s. But TMA is still 2–5% behind.

The PTX explains why. The pipeliner turns the **masked** loads into asynchronous copies too: `cp.async` with predicates, the non-TMA asynchronous copy from earlier GPUs (192 of them in the ragged case). So the masked kernel gets the same overlap without TMA, and the mask is just a predicate on each copy.

### Conclusion

- **Memory-bound kernels:** for an elementwise, memory-bound kernel like `mask2d`, the tail mask costs essentially nothing on the GB10. Neither recovering vector loads (the fast-path test) nor replacing the mask with TMA makes it faster. The ceiling is memory bandwidth, about 232 GB/s here.
- **Where TMA's real benefits are:** fewer registers and threads spent on addresses and masks, and asynchronous copies that run alongside heavy computation. Those benefits are in compute-bound kernels like matmul or attention, where Triton already uses tensor descriptors. Those weren't tested.

## Result: tail masks in a Tensor Core matmul

`mask2d` is elementwise, so it can't use Tensor Cores (they only do matrix multiply-accumulate), and it's memory-bound anyway. To see what tail masks cost when Tensor Cores are involved, `~/mask2d_gpu/matmul_gpu_tail.py` on the GX10 runs an fp16 matmul (fp32 accumulate, 128×128×32 tiles, `num_warps=8`, `num_stages=3`) in three versions:
1. **masked:** every load has the M/N edge mask and the K-tail mask (`other=0`).
2. **fast path:** if the tile is fully inside in M and N, its loads have no M/N mask, and only the last K step is masked. Edge tiles use version 1.
3. **TMA:** tensor descriptors and no masks; out-of-bounds elements (M/N edges and the K tail) read as 0.

The ragged shape is `4093 × 4093 × 4093`, stored with a row stride of 4096 (TMA needs 16-byte-aligned strides, and all three versions get the same storage). A non-ragged `4096³` run is the upper bound. All six runs matched `torch.matmul`. Each case was run once; the variation between timings wasn't measured.

| shape (row stride 4096) | kernel | time | TFLOPS | Tensor Core instrs | copies to shared memory |
|---|---|---|---|---|---|
| **`4093³`** (ragged M, N, K) | masked | 6311.1 µs | 21.7 | 32 `mma.sync` | **none** (`cp.async = 0`) |
| | fast path | 6248.6 µs | 21.9 | 96 `mma.sync` | 12 `cp.async` |
| | TMA | 5989.5 µs | 22.9 | 32 `mma.sync` | 7 `cp.async.bulk.tensor` |
| **`4096³`** (not ragged) | masked | 3537.0 µs | **38.9** | 32 `mma.sync` | **12 `cp.async`** |
| | fast path | 6068.9 µs | 22.6 | 96 `mma.sync` | 24 `cp.async` |
| | TMA | 5932.4 µs | 23.2 | 32 `mma.sync` | 7 `cp.async.bulk.tensor` |

**What this shows:**

1. **Tensor Cores are used**, as warp-level `mma.sync` (with `ldmatrix` to load their operands). There's no `wgmma` and no `tcgen05`, which confirms the GB10 takes the consumer-Blackwell / Ampere-style Tensor Core path.
2. **The tail mask matters a lot here**, unlike in `mask2d`: the same masked kernel runs at **38.9 TFLOPS** on `4096³` but only **21.7** on `4093³`, about 1.8× slower. The PTX shows the reason:
   - **Aligned shape:** the loads become `cp.async`, asynchronous copies to shared memory that the pipeliner prefetches across the K loop (`num_stages=3`).
   - **Ragged shape:** there are no `cp.async` at all, so the loads are plain loads, and the K loop isn't pipelined.

   A 16-byte `cp.async` moves 8 fp16 values at once. With `K = 4093`, the K-tail mask can't be proven constant across 8 elements, so the async copy can't be used. Losing the pipelining is what makes it slower. The mask arithmetic itself is cheap.
3. **The hand-written fast path didn't help.** It does produce `cp.async` (12 of them), but it runs at about 22 TFLOPS even on the aligned shape, where the plain masked kernel reaches 38.9. Its three loops (interior, K tail, edge) triple the `mma.sync` code, and putting the main loop inside an `if` probably keeps it from being pipelined the way the simple loop is. Which of these causes it hasn't been confirmed. As written, it isn't a good implementation of the idea.
4. **TMA removes the masks but tops out around 23 TFLOPS on both shapes.** It slightly beats the ragged masked kernel, but it's far from 38.9 even when nothing is ragged. So on this GPU, this TMA + `mma.sync` path is slower than `cp.async` pipelining in general. That's a property of the path, not of the tail mask.

**So, for tail masks with Tensor Cores on the GB10:** there's real room, about 1.8× between ragged and aligned shapes, and it comes from keeping the asynchronous-copy pipeline. A better way to get it back would be to version outside the kernel: launch the plain, pipelined, unmasked kernel for the interior tiles and the K range that is a multiple of `BLOCK_K`, and a masked kernel only for the edges. That's untested.
