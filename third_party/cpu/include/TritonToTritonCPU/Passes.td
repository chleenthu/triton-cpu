#ifndef TRITONTOTRITONCPU_CONVERSION_PASSES
#define TRITONTOTRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMemoryOps : Pass<"triton-cpu-convert-memory-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory ops.";
    let description = [{

    }];

    let options = [
        Option<"useGatherScatter", "use-gather-scatter",
               "bool", /*default*/"false",
               "Use Gather or Scatter to lower memory ops.">,
        Option<"assumeInBounds", "assume-in-bounds",
               "bool", /*default*/"false",
               "Assume all tensor descriptor memory accesses are in bounds.">
    ];

    let constructor = "mlir::triton::cpu::createConvertMemoryOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def AnalyzeTailMasks : Pass<"triton-cpu-analyze-tail-masks", "mlir::ModuleOp"> {
    let summary = "Make box-shaped (tail) masks explicit as triton_cpu.tail_mask.";
    let description = [{
        For every mask of tt.load / tt.store and every tensor condition of
        arith.select, trace how the mask is formed (arith.cmpi, arith.andi,
        arith.ori, arith.addi, arith.subi, tt.make_range, tt.splat,
        tt.expand_dims, tt.broadcast, tt.trans, arith.constant). If element
        (i_0, ..., i_{n-1}) is on exactly when i_d < B_d for every d, replace
        the mask operand with `triton_cpu.tail_mask B_0, ..., B_{n-1}`, so
        the TTIR -> TTCIR conversions can carry the bounds forward
        (vector.create_mask, per-row masks, masked reductions).
    }];
    let constructor = "mlir::triton::cpu::createAnalyzeTailMasks()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertElementwiseOps : Pass<"triton-cpu-convert-elementwise-ops", "mlir::ModuleOp"> {
    let summary = "Convert elementwise ops.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertElementwiseOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertElemManipOps : Pass<"triton-cpu-convert-elem-manip-ops", "mlir::ModuleOp"> {
    let summary = "Convert elements manipulation ops (transpose, shuffle, etc.).";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertElemManipOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertPtrOps : Pass<"triton-cpu-convert-ptr-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton ops related to pointer arithmetics.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertPtrOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDotOp : Pass<"triton-cpu-convert-dot-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton DotOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertDotOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertControlFlowOps : Pass<"triton-cpu-convert-control-flow-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton DotOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertControlFlowOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertHistogramOp : Pass<"triton-cpu-convert-histogram-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton HistogramOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertHistogramOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect"];
}

def ConvertReductionOp : Pass<"triton-cpu-convert-reduction", "mlir::ModuleOp"> {
    let summary = "Convert Triton ReduceOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertReductionOp()";

    let options = [
        Option<"useMultiDimReductionOp", "use-multidim-reduction-op",
               "bool", /*default*/"false",
               "Use vector::MultiDimReductionOp and its default lowering when possible.">,

        Option<"useReductionOp", "use-reduction-op",
               "bool", /*default*/"false",
                "Use vector::ReductionOp and its default lowering when possible.">,

        Option<"useMaskedReduction", "use-masked-reduction",
               "bool", /*default*/"false",
                "Reduce select(tail mask, x, identity) as a masked vector.reduction over x.">,
    ];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertScanOp : Pass<"triton-cpu-convert-scan", "mlir::ModuleOp"> {
    let summary = "Convert Triton ScanOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertScanOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertAtomicOps : Pass<"triton-cpu-convert-atomic-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton atomic operations.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertAtomicOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ConvertDebugOps : Pass<"triton-cpu-convert-debug-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton debug operations.";
    let description = [{

    }];
    let constructor = "mlir::triton::cpu::createConvertDebugOps()";

    let dependentDialects = ["mlir::vector::VectorDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}

def ScalarizeUsingForOp : Pass<"triton-cpu-scalarize", "mlir::ModuleOp"> {
    let summary = "Insert Loops for ops, that are not vectorizable";
    let description = [{
        This pass is used to reduce compile time by generating loops for
        operations that cannot be handled as vectors, and simply increases
        the amount of IR without any further optimization.
    }];

    let options = [
        Option<"skipGatherScatter", "skip-gather-scatter",
               "bool", /*default*/"false",
               "Skip scalarizing gather/scatter ops.">,
    ];

    let constructor = "mlir::triton::cpu::createScalarizeUsingForOpPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::cpu::TritonCPUDialect"];
}


#endif
