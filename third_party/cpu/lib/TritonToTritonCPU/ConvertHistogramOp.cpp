#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTHISTOGRAMOP
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class HistogramConversionTarget : public ConversionTarget {
public:
  explicit HistogramConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::BuiltinDialect>();
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<math::MathDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();

    addIllegalOp<triton::HistogramOp>();
  }
};

struct HistogramOpConversion : public OpConversionPattern<triton::HistogramOp> {
  using OpConversionPattern::OpConversionPattern;

  // Number of leading lanes a tail mask keeps on: the operand of a 1-D
  // vector.create_mask (an index Value) or the size of a 1-D
  // vector.constant_mask (an int64_t). Looks through the casts the type
  // conversion leaves between the tensor and vector worlds. AnalyzeTailMasks +
  // ConvertElementwiseOps turn `offs < n` into such a create_mask.
  static std::optional<OpFoldResult> getTailLength(Value mask) {
    while (auto cast = mask.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1)
        return std::nullopt;
      mask = cast.getOperand(0);
    }
    if (auto cm = mask.getDefiningOp<vector::CreateMaskOp>()) {
      if (cm.getNumOperands() == 1)
        return OpFoldResult(cm.getOperand(0));
      return std::nullopt;
    }
    if (auto cm = mask.getDefiningOp<vector::ConstantMaskOp>()) {
      if (cm.getMaskDimSizes().size() == 1) {
        Builder b(mask.getContext());
        return OpFoldResult(b.getIndexAttr(cm.getMaskDimSizes()[0]));
      }
    }
    return std::nullopt;
  }

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = rewriter.getRemappedValue(op.getSrc());
    auto srcTy = dyn_cast<VectorType>(src.getType());
    auto resTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(op.getType()));

    if (srcTy.getRank() != 1)
      llvm_unreachable("unsupported input for histogram op (rank != 1)");

    Value zero = arith::ConstantOp::create(rewriter, loc, resTy,
                                           rewriter.getZeroAttr(resTy));
    Value one = arith::ConstantOp::create(rewriter, loc, resTy,
                                          rewriter.getOneAttr(resTy));
    VectorType cmpVecTy =
        VectorType::get(resTy.getShape(), srcTy.getElementType());
    Value rangeVec = arith::ConstantOp::create(
        rewriter, loc, resTy, makeRangeAttr(cmpVecTy, rewriter));
    Value srcMask;
    if (op.getMask())
      srcMask = rewriter.getRemappedValue(op.getMask());

    // Tail mask of length L: only source lanes [0, L) count, so stop there
    // instead of masking every one of the M steps.
    if (std::optional<OpFoldResult> tailLen =
            srcMask ? getTailLength(srcMask) : std::nullopt) {
      int64_t numSrc = srcTy.getShape()[0];
      // res += (x[i] == range) ? 1 : 0
      auto step = [&](OpBuilder &b, Location l, Value res, Value elem) {
        Value elemVec = vector::BroadcastOp::create(b, l, cmpVecTy, elem);
        Value hit = arith::CmpIOp::create(b, l, arith::CmpIPredicate::eq,
                                          elemVec, rangeVec);
        Value delta = arith::SelectOp::create(b, l, hit, one, zero);
        return arith::AddIOp::create(b, l, res, delta).getResult();
      };
      if (auto attr = dyn_cast<Attribute>(*tailLen)) {
        // Constant L: unrolled, cut at L.
        int64_t len =
            std::clamp<int64_t>(cast<IntegerAttr>(attr).getInt(), 0, numSrc);
        Value res = zero;
        for (int64_t i = 0; i < len; ++i)
          res = step(rewriter, loc, res,
                     vector::ExtractOp::create(rewriter, loc, src, i));
        rewriter.replaceOp(op, res);
        return success();
      }
      // Dynamic L: scf.for i in [0, min(L, M)). A negative upper bound runs
      // zero iterations, so no lower clamp is needed.
      Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value cn = arith::ConstantIndexOp::create(rewriter, loc, numSrc);
      Value ub =
          arith::MinSIOp::create(rewriter, loc, cast<Value>(*tailLen), cn);
      auto loop = scf::ForOp::create(
          rewriter, loc, c0, ub, c1, ValueRange{zero},
          [&](OpBuilder &b, Location l, Value iv, ValueRange iters) {
            Value elem = vector::ExtractOp::create(b, l, src, OpFoldResult(iv));
            scf::YieldOp::create(b, l, step(b, l, iters[0], elem));
          });
      rewriter.replaceOp(op, loop.getResult(0));
      return success();
    }

    VectorType maskVecTy =
        VectorType::get(resTy.getShape(), rewriter.getI1Type());
    Value res = zero;
    for (int64_t i = 0; i < srcTy.getShape()[0]; ++i) {
      Value idx = arith::ConstantOp::create(
          rewriter, loc, rewriter.getIndexType(), rewriter.getIndexAttr(i));
      Value elem = vector::ExtractOp::create(rewriter, loc, src, idx);
      Value elemVec =
          vector::BroadcastOp::create(rewriter, loc, cmpVecTy, elem);
      Value mask = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, elemVec, rangeVec);
      // Masked-out source elements do not contribute to any bin.
      if (srcMask) {
        Value maskElem = vector::ExtractOp::create(rewriter, loc, srcMask, idx);
        Value maskElemVec =
            vector::BroadcastOp::create(rewriter, loc, maskVecTy, maskElem);
        mask = arith::AndIOp::create(rewriter, loc, mask, maskElemVec);
      }
      Value delta = vector::selectPassthru(rewriter, mask, one, zero);
      res = arith::AddIOp::create(rewriter, loc, res, delta);
    }

    rewriter.replaceOp(op, res);

    return success();
  }

  TypedAttr makeRangeAttr(VectorType resTy,
                          ConversionPatternRewriter &rewriter) const {
    Type elemTy = resTy.getElementType();
    if (elemTy.isInteger(32)) {
      SmallVector<int32_t> range(resTy.getShape()[0]);
      std::iota(range.begin(), range.end(), 0);
      return rewriter.getI32VectorAttr(range);
    } else if (elemTy.isInteger(64)) {
      SmallVector<int64_t> range(resTy.getShape()[0]);
      std::iota(range.begin(), range.end(), 0);
      return rewriter.getI64VectorAttr(range);
    } else {
      llvm_unreachable(
          "unsupported src elem type for histogram (expected i32 or i64)");
    }
  }
};

struct ConvertHistogramOp
    : public triton::impl::ConvertHistogramOpBase<ConvertHistogramOp> {
  using ConvertHistogramOpBase::ConvertHistogramOpBase;

  ConvertHistogramOp() : ConvertHistogramOpBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    HistogramConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<HistogramOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertHistogramOp() {
  return std::make_unique<ConvertHistogramOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
