# CPU backend pass pipeline (tail-mask view)

The pass order in `third_party/cpu/backend/compiler.py`, with the tail-mask-related steps marked ★.

`ScalarizeInterface` isn't a pass. It's an op interface: a set of `computeScalarValue` / `canComputeScalarValue` methods attached to ops. It's used inside two passes, the scalarize pass (`ScalarizeUsingForOps`) and `ConvertMemoryOps` (`extractScalarPointer`).

## Stage 1: `make_ttcir` (TTIR → TTCIR)

```python
if _vsetvl_enabled():                                   # TRITON_VSETVL_MINE / _LANE / _REDUCE
    cpu.passes.ttcpuir.add_analyze_tail_masks(pm)       # ★ creates triton_cpu.tail_mask
cpu.passes.ttcpuir.add_scalarize(pm, True)              # ScalarizeUsingForOps: element loops for
                                                        #   non-contiguous ops (uses ScalarizeInterface)
cpu.passes.ttcpuir.add_convert_memory_ops(pm, ...)      # ★ getTailRowMask: per-row 1-D create_mask;
                                                        #   extractScalarPointer (uses ScalarizeInterface)
cpu.passes.ttcpuir.add_convert_ptr_ops(pm)
cpu.passes.ttcpuir.add_convert_elementwise_ops(pm)      # ★ TailMaskOpConversion: tail_mask -> n-D create_mask
cpu.passes.ttcpuir.add_convert_reduction_op(pm, ...)    # ★ ConvertReductionOp: mapToMaskedReduction
                                                        #   -> vector.mask { vector.reduction } (TRITON_VSETVL_REDUCE)
passes.common.add_cse(pm)
passes.common.add_symbol_dce(pm)
passes.common.add_canonicalizer(pm)                     # removes unused ops (e.g. the dead n-D create_mask)
```

## Stage 2: `make_tttcir` (TTCIR optimizations)

```python
cpu.passes.ttcpuir.add_triton_cpu_canonicalizer(pm)
cpu.passes.ttcpuir.add_optimize_masks(pm)
passes.common.add_canonicalizer(pm)
...                                                     # dot lowerings, add_convert_unsupported_ops,
                                                        #   add_decompose_fp_conversions (target-dependent)
if _vsetvl_enabled():
    cpu.passes.ttcpuir.add_tail_mask_to_evl(pm)         # ★ TailMaskToEVL: create_mask for other tail masks,
                                                        #   poison pass-through
passes.common.add_cse(pm)
passes.common.add_symbol_dce(pm)
passes.common.add_canonicalizer(pm)
```

## Stage 3: `make_llir` (TTCIR → LLVM dialect)

```python
cpu.passes.ttcpuir.add_vector_to_scf(pm, True, 1, False)
cpu.passes.ttcpuir.add_lower_affine(pm)
passes.convert.add_scf_to_cf(pm)
passes.convert.add_index_to_llvmir(pm)
cpu.passes.ttcpuir.add_func_op_to_llvmir(pm)
cpu.passes.ttcpuir.add_program_id_to_llvmir(pm)
cpu.passes.ttcpuir.add_memory_op_to_llvmir(pm)          # ★ MemoryOpToLLVM: maskedload/store with a 1-D create_mask
                                                        #   -> vp.load/vp.store or vsetvli loop (TRITON_VSETVL_MINE / _LANE)
...                                                     # math lowerings
cpu.passes.ttcpuir.add_vector_to_llvmir(pm, ...)        # ★ upstream VectorToLLVM: vector.mask { vector.reduction }
                                                        #   -> llvm.vp.reduce.* (MaskedReductionOpConversion)
cpu.passes.ttcpuir.add_memref_to_llvmir(pm)
passes.convert.add_reconcile_unrealized_casts(pm)
passes.convert.add_arith_to_llvmir(pm)
cpu.passes.ttcpuir.add_func_to_llvmir(pm)
cpu.passes.ttcpuir.add_ub_to_llvmir(pm)                 # ub.poison (from TailMaskToEVL) -> LLVM poison
passes.common.add_canonicalizer(pm)
passes.common.add_cse(pm)
passes.common.add_symbol_dce(pm)
```

## The tail-mask path in one line

```
AnalyzeTailMasks (tail_mask) -> ConvertMemoryOps (row create_mask) -> ConvertElementwiseOps (n-D create_mask)
  -> ConvertReductionOp (vector.mask reduction) -> TailMaskToEVL (rebuild other masks, poison)
  -> MemoryOpToLLVM (vp.load / vsetvli loop) -> VectorToLLVM (vp.reduce)
```

Two order details worth noting:
- **`MemoryOpToLLVM` runs before `VectorToLLVM`.** So the custom `vp.load`/`vp.store` lowering claims the masked loads and stores first. The upstream pass then handles what's left, including the masked reductions.
- **`add_ub_to_llvmir`** turns the `ub.poison` pass-throughs from `TailMaskToEVL` into LLVM `poison` values.
