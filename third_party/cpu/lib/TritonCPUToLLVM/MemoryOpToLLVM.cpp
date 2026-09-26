#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Pass/Pass.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <cstdlib>
#include <algorithm>
#include <optional>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MEMORYOPTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct PtrToMemRefOpConversion : public OpConversionPattern<PtrToMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PtrToMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value ptr = rewriter.getRemappedValue(op.getSrc());
    auto memRefStructTy = getTypeConverter()->convertType(op.getType());

    Value res = b.undef(memRefStructTy);
    res =
        LLVM::InsertValueOp::create(rewriter, loc, memRefStructTy, res, ptr, 1);
    rewriter.replaceOp(op, res);

    return success();
  }
};

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resTy, ptr, 0,
                                              op.getIsVolatile());
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Value val = rewriter.getRemappedValue(op.getValue());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, val, ptr);
    return success();
  }
};


// Explicit vector length of a masked access whose mask is a tail mask
// (lanes [0, EVL) on, the rest off): a constant prefix mask, a 1-D
// vector.constant_mask, or a 1-D vector.create_mask (TailMaskToEVL rewrites
// tail masks into this form). `constant` is set when EVL is known at compile
// time; otherwise `value` is an i32 already clamped to [0, N].
struct MaskEVL {
  std::optional<int64_t> constant;
  Value value;
};

static std::optional<MaskEVL> getMaskEVL(Value mask, int64_t n,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc) {
  DenseIntElementsAttr attr;
  if (matchPattern(mask, m_Constant(&attr))) {
    int64_t len = 0;
    bool seenFalse = false;
    for (bool bit : attr.getValues<bool>()) {
      if (bit && seenFalse)
        return std::nullopt;
      if (bit)
        ++len;
      else
        seenFalse = true;
    }
    return MaskEVL{len, Value()};
  }
  if (auto cm = mask.getDefiningOp<vector::ConstantMaskOp>()) {
    if (cm.getMaskDimSizes().size() != 1)
      return std::nullopt;
    return MaskEVL{std::clamp<int64_t>(cm.getMaskDimSizes()[0], 0, n), Value()};
  }
  if (auto cm = mask.getDefiningOp<vector::CreateMaskOp>()) {
    if (cm.getNumOperands() != 1)
      return std::nullopt;
    auto i64Ty = rewriter.getI64Type();
    Value len = rewriter.getRemappedValue(cm.getOperand(0));
    if (len.getType() != i64Ty)
      len = UnrealizedConversionCastOp::create(rewriter, loc, i64Ty, len)
                .getResult(0);
    // vector.create_mask clamps its operand to [0, n]; EVL must be in range.
    Value zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(0));
    Value nVal = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(n));
    len = LLVM::SMaxOp::create(rewriter, loc, len, zero);
    len = LLVM::SMinOp::create(rewriter, loc, len, nVal);
    return MaskEVL{std::nullopt,
                   LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(),
                                         len)};
  }
  return std::nullopt;
}

// Name of a vp intrinsic overloaded on a fixed vector type, e.g.
// "llvm.vp.load.v1024bf16.p0" or "llvm.vp.merge.v1024bf16".
static std::string getVPIntrinsicName(StringRef base, VectorType vecTy,
                                      bool hasPtr = true) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "llvm.vp." << base << ".v" << vecTy.getNumElements()
     << vecTy.getElementType() << (hasPtr ? ".p0" : "");
  return name;
}

static Value getAllTrueMask(ConversionPatternRewriter &rewriter, Location loc,
                            VectorType vecTy) {
  auto maskTy = VectorType::get(vecTy.getShape(), rewriter.getI1Type());
  return LLVM::ConstantOp::create(
      rewriter, loc, maskTy,
      SplatElementsAttr::get(maskTy, rewriter.getBoolAttr(true)));
}

static Value getEVLValue(ConversionPatternRewriter &rewriter, Location loc,
                         const MaskEVL &evl) {
  if (evl.value)
    return evl.value;
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(*evl.constant));
}

// Address of base[indices] for a rank-1 memref lowered by
// PtrToMemRefOpConversion (only the aligned pointer, field 1, is set).
static Value getRank1ElementPtr(ConversionPatternRewriter &rewriter,
                                Location loc, Value memrefDesc,
                                ValueRange indices, Type elemTy) {
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value ptr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, memrefDesc,
                                           ArrayRef<int64_t>{1});
  return LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, ptr,
                             ArrayRef<LLVM::GEPArg>{indices[0]});
}

// vector.maskedload with a tail mask -> llvm.vp.load (all-true mask,
// EVL = number of active lanes); lanes >= EVL take the pass-through value via
// llvm.vp.merge (vp.load leaves them poison), unless the pass-through is
// ub.poison (TailMaskToEVL proved those lanes unused).
struct VectorMaskedLoadOpConversion : public OpConversionPattern<vector::MaskedLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType vecTy = op.getVectorType();
    if (vecTy.getRank() != 1 || vecTy.isScalable() ||
        op.getMemRefType().getRank() != 1)
      return failure();
    int64_t n = vecTy.getNumElements();
    std::optional<MaskEVL> evl = getMaskEVL(op.getMask(), n, rewriter, loc);
    if (!evl)
      return failure();

    Value passThru = adaptor.getPassThru();
    if (evl->constant == 0) {
      rewriter.replaceOp(op, passThru);
      return success();
    }

    Type elemTy = getTypeConverter()->convertType(vecTy.getElementType());
    Value loadPtr = getRank1ElementPtr(rewriter, loc, adaptor.getBase(),
                                       adaptor.getIndices(), elemTy);
    Value allTrueMask = getAllTrueMask(rewriter, loc, vecTy);
    Value evlVal = getEVLValue(rewriter, loc, *evl);
    Value vpLoad =
        LLVM::CallIntrinsicOp::create(
            rewriter, loc, vecTy,
            rewriter.getStringAttr(getVPIntrinsicName("load", vecTy)),
            ValueRange({loadPtr, allTrueMask, evlVal}))
            .getResult(0);

    if (evl->constant == n || op.getPassThru().getDefiningOp<ub::PoisonOp>()) {
      rewriter.replaceOp(op, vpLoad);
      return success();
    }
    Value merged =
        LLVM::CallIntrinsicOp::create(
            rewriter, loc, vecTy,
            rewriter.getStringAttr(
                getVPIntrinsicName("merge", vecTy, /*hasPtr=*/false)),
            ValueRange({allTrueMask, vpLoad, passThru, evlVal}))
            .getResult(0);
    rewriter.replaceOp(op, merged);
    return success();
  }
};

// vector.maskedstore with a tail mask -> llvm.vp.store (all-true mask,
// EVL = number of active lanes). Lanes >= EVL are not written, exactly like
// the masked-off lanes of the original store.
struct VectorMaskedStoreOpConversion : public OpConversionPattern<vector::MaskedStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType vecTy = op.getVectorType();
    if (vecTy.getRank() != 1 || vecTy.isScalable() ||
        op.getMemRefType().getRank() != 1)
      return failure();
    std::optional<MaskEVL> evl =
        getMaskEVL(op.getMask(), vecTy.getNumElements(), rewriter, loc);
    if (!evl)
      return failure();

    if (evl->constant == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    Type elemTy = getTypeConverter()->convertType(vecTy.getElementType());
    Value storePtr = getRank1ElementPtr(rewriter, loc, adaptor.getBase(),
                                        adaptor.getIndices(), elemTy);
    Value allTrueMask = getAllTrueMask(rewriter, loc, vecTy);
    Value evlVal = getEVLValue(rewriter, loc, *evl);
    LLVM::CallIntrinsicOp::create(
        rewriter, loc, rewriter.getStringAttr(getVPIntrinsicName("store", vecTy)),
        ValueRange({adaptor.getValueToStore(), storePtr, allTrueMask, evlVal}));
    rewriter.eraseOp(op);
    return success();
  }
};


// Strip-mined data movement for TRITON_VSETVL_MINE: copy `evl` elements of
// `bitWidth` bits from `src` to `dst` with the RVV idiom
//   for (i = 0; i < evl; i += vl) {
//     vl = vsetvli(evl - i, SEW, m8);
//     dst[i : i+vl] = vp.load(src + i, vl);
//   }
// Data is moved as same-width integers (<vscale x K x iN>), so bf16/f16/f32
// all use plain integer loads/stores. The ops after the insertion point are
// moved into a new continuation block; the insertion point is left at its
// start.
static void emitVsetvlCopyLoop(ConversionPatternRewriter &rewriter,
                               Location loc, Value src, Value dst,
                               unsigned bitWidth, Value evl64) {
  MLIRContext *ctx = rewriter.getContext();
  auto i64Ty = rewriter.getI64Type();
  auto i32Ty = rewriter.getI32Type();
  Type ptrTy = LLVM::LLVMPointerType::get(ctx);
  Type elemTy = rewriter.getIntegerType(bitWidth);
  // vsetvli immediates: SEW = log2(bits / 8), LMUL m8 = 3. An m8 register
  // group holds vscale * 64 * 8 / bits elements.
  int64_t sew = llvm::Log2_64(bitWidth / 8);
  const int64_t lmulM8 = 3;
  auto chunkTy = VectorType::get({64 * 8 / bitWidth}, elemTy, {true});
  auto chunkMaskTy = VectorType::get({64 * 8 / bitWidth}, rewriter.getI1Type(),
                                     {true});
  std::string suffix = (".nxv" + Twine(64 * 8 / bitWidth) + "i" +
                        Twine(bitWidth))
                           .str();

  Block *currentBlock = rewriter.getBlock();
  Block *continueBlock =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
  Block *loopBlock = rewriter.createBlock(currentBlock->getParent(),
                                          continueBlock->getIterator(), {i64Ty},
                                          {loc});
  rewriter.setInsertionPointToEnd(currentBlock);
  Value zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                        rewriter.getI64IntegerAttr(0));
  LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, loopBlock);

  rewriter.setInsertionPointToStart(loopBlock);
  Value iv = loopBlock->getArgument(0);
  Value remain = LLVM::SubOp::create(rewriter, loc, evl64, iv);
  Value sewVal = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                          rewriter.getI64IntegerAttr(sew));
  Value lmulVal = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                           rewriter.getI64IntegerAttr(lmulM8));
  Value vl = LLVM::CallIntrinsicOp::create(
                 rewriter, loc, i64Ty,
                 rewriter.getStringAttr("llvm.riscv.vsetvli.i64"),
                 ValueRange({remain, sewVal, lmulVal}))
                 .getResult(0);
  Value vl32 = LLVM::TruncOp::create(rewriter, loc, i32Ty, vl);
  Value allTrue = LLVM::ConstantOp::create(
      rewriter, loc, chunkMaskTy,
      SplatElementsAttr::get(chunkMaskTy, rewriter.getBoolAttr(true)));
  Value srcPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, src,
                                     ArrayRef<LLVM::GEPArg>{iv});
  Value dstPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, dst,
                                     ArrayRef<LLVM::GEPArg>{iv});
  Value chunk = LLVM::CallIntrinsicOp::create(
                    rewriter, loc, chunkTy,
                    rewriter.getStringAttr("llvm.vp.load" + suffix + ".p0"),
                    ValueRange({srcPtr, allTrue, vl32}))
                    .getResult(0);
  LLVM::CallIntrinsicOp::create(
      rewriter, loc, rewriter.getStringAttr("llvm.vp.store" + suffix + ".p0"),
      ValueRange({chunk, dstPtr, allTrue, vl32}));
  Value next = LLVM::AddOp::create(rewriter, loc, iv, vl);
  // vsetvli(0) == 0, so evl == 0 leaves after one empty iteration.
  Value done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge,
                                    next, evl64);
  LLVM::CondBrOp::create(rewriter, loc, done, continueBlock, ValueRange{},
                         loopBlock, ValueRange{next});
  rewriter.setInsertionPointToStart(continueBlock);
}

static Value getEVL64(ConversionPatternRewriter &rewriter, Location loc,
                      const MaskEVL &evl) {
  auto i64Ty = rewriter.getI64Type();
  if (evl.value)
    return LLVM::ZExtOp::create(rewriter, loc, i64Ty, evl.value);
  return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                  rewriter.getI64IntegerAttr(*evl.constant));
}

// A stack slot for one whole vector, placed in the function's entry block so
// it is allocated once even when the access is inside a loop.
static Value createEntryAlloca(ConversionPatternRewriter &rewriter,
                               Operation *op, Type vecTy) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto func = op->getParentOfType<FunctionOpInterface>();
  rewriter.setInsertionPointToStart(&func.getFunctionBody().front());
  Location loc = op->getLoc();
  Value one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(1));
  return LLVM::AllocaOp::create(
      rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()), vecTy,
      one, /*alignment=*/16);
}

static std::optional<unsigned> getCopyBitWidth(VectorType vecTy) {
  unsigned bits = vecTy.getElementTypeBitWidth();
  if (bits != 8 && bits != 16 && bits != 32 && bits != 64)
    return std::nullopt;
  return bits;
}

// TRITON_VSETVL_MINE: vector.maskedload with a tail mask -> the vector's
// stack slot is filled with the pass-through (skipped for ub.poison), the
// first EVL elements are copied into it by a vsetvli loop, then the whole
// vector is loaded from it.
struct VsetvlMaskedLoadOpConversion : public OpConversionPattern<vector::MaskedLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType vecTy = op.getVectorType();
    if (vecTy.getRank() != 1 || vecTy.isScalable() ||
        op.getMemRefType().getRank() != 1)
      return failure();
    std::optional<unsigned> bits = getCopyBitWidth(vecTy);
    if (!bits)
      return failure();
    int64_t n = vecTy.getNumElements();
    std::optional<MaskEVL> evl = getMaskEVL(op.getMask(), n, rewriter, loc);
    if (!evl)
      return failure();

    Value passThru = adaptor.getPassThru();
    if (evl->constant == 0) {
      rewriter.replaceOp(op, passThru);
      return success();
    }

    Type resTy = getTypeConverter()->convertType(vecTy);
    Type elemTy = getTypeConverter()->convertType(vecTy.getElementType());
    Value src = getRank1ElementPtr(rewriter, loc, adaptor.getBase(),
                                   adaptor.getIndices(), elemTy);
    Value slot = createEntryAlloca(rewriter, op, resTy);
    if (!op.getPassThru().getDefiningOp<ub::PoisonOp>() &&
        evl->constant != n)
      LLVM::StoreOp::create(rewriter, loc, passThru, slot, /*alignment=*/16);
    emitVsetvlCopyLoop(rewriter, loc, src, slot, *bits,
                       getEVL64(rewriter, loc, *evl));
    Value result = LLVM::LoadOp::create(rewriter, loc, resTy, slot,
                                        /*alignment=*/16);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// TRITON_VSETVL_MINE: vector.maskedstore with a tail mask -> the value is
// stored to a stack slot and its first EVL elements are copied to memory by a
// vsetvli loop.
struct VsetvlMaskedStoreOpConversion : public OpConversionPattern<vector::MaskedStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    VectorType vecTy = op.getVectorType();
    if (vecTy.getRank() != 1 || vecTy.isScalable() ||
        op.getMemRefType().getRank() != 1)
      return failure();
    std::optional<unsigned> bits = getCopyBitWidth(vecTy);
    if (!bits)
      return failure();
    std::optional<MaskEVL> evl =
        getMaskEVL(op.getMask(), vecTy.getNumElements(), rewriter, loc);
    if (!evl)
      return failure();

    if (evl->constant == 0) {
      rewriter.eraseOp(op);
      return success();
    }

    Type valTy = getTypeConverter()->convertType(vecTy);
    Type elemTy = getTypeConverter()->convertType(vecTy.getElementType());
    Value dst = getRank1ElementPtr(rewriter, loc, adaptor.getBase(),
                                   adaptor.getIndices(), elemTy);
    Value slot = createEntryAlloca(rewriter, op, valTy);
    LLVM::StoreOp::create(rewriter, loc, adaptor.getValueToStore(), slot,
                          /*alignment=*/16);
    Value evl64 = getEVL64(rewriter, loc, *evl);
    rewriter.eraseOp(op);
    emitVsetvlCopyLoop(rewriter, loc, slot, dst, *bits, evl64);
    return success();
  }
};


struct PtrToIntOpConversion : public OpConversionPattern<triton::PtrToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = rewriter.getRemappedValue(op.getSrc());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resTy, src);
    return success();
  }
};

struct IntToPtrOpConversion : public OpConversionPattern<triton::IntToPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = rewriter.getRemappedValue(op.getSrc());
    Type resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, resTy, src);
    return success();
  }
};

struct AddPtrOpConversion : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expect only scalar pointers here.
    assert(isa<PointerType>(op.getType()));
    auto ptrTy = cast<PointerType>(op.getPtr().getType());
    Type elemTy = getTypeConverter()->convertType(ptrTy.getPointeeType());
    Type resTy = getTypeConverter()->convertType(ptrTy);
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Value offset = rewriter.getRemappedValue(op.getOffset());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, resTy, elemTy, ptr, offset);
    return success();
  }
};

struct PtrBitcastConversion : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // By this moment we expect tt.bitcast used only for scalar pointer casts.
    // This cast becomes NOP for LLVM dialect, so simply return the source arg.
    assert(isa<PointerType>(op.getType()));
    assert(isa<PointerType>(op.getSrc().getType()));
    Value src = rewriter.getRemappedValue(op.getSrc());
    rewriter.replaceOp(op, src);
    return success();
  }
};

struct PtrSelectConversion : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // By this moment we expect tt.bitcast used only for scalar pointer casts.
    // This cast becomes NOP for LLVM dialect, so simply return the source arg.
    if (!isa<PointerType>(op.getType()))
      return failure();

    Value trueVal = rewriter.getRemappedValue(op.getTrueValue());
    Value falseVal = rewriter.getRemappedValue(op.getFalseValue());
    Value cond = rewriter.getRemappedValue(op.getCondition());
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(op, cond, trueVal, falseVal);
    return success();
  }
};

struct MakeTensorDescOpConversion
    : public OpConversionPattern<MakeTensorDescOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto structTy = getTypeConverter()->convertType(op.getType());
    auto i64Ty = IntegerType::get(getContext(), 64);

    auto d = MemRefDescriptor::poison(rewriter, loc, structTy);
    d.setAlignedPtr(rewriter, loc, adaptor.getBase());
    d.setConstantOffset(rewriter, loc, 0);
    for (auto [i, v] : llvm::enumerate(op.getShape()))
      d.setSize(rewriter, loc, i,
                LLVM::ZExtOp::create(rewriter, loc, i64Ty, v));
    for (auto [i, v] : llvm::enumerate(op.getStrides()))
      d.setStride(rewriter, loc, i, v);

    rewriter.replaceOp(op, static_cast<Value>(d));

    return success();
  }
};

struct ExtractMemRefOpConversion : public OpConversionPattern<ExtractMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractMemRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getDesc());
    return success();
  }
};

struct MemoryOpToLLVM
    : public triton::impl::MemoryOpToLLVMBase<MemoryOpToLLVM> {
  using MemoryOpToLLVMBase::MemoryOpToLLVMBase;

  MemoryOpToLLVM() : MemoryOpToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<LoadOpConversion>(typeConverter, context);
    patterns.add<StoreOpConversion>(typeConverter, context);

    // Experimental RISC-V lowerings of tail-masked loads/stores (the masks
    // come from TailMaskToEVL): TRITON_VSETVL_MINE copies through a stack
    // slot with a vsetvli loop, TRITON_VSETVL_LANE uses vp.load / vp.store.
    if (getenv("TRITON_VSETVL_MINE")) {
      llvm::outs() << "!TRITON_VSETVL_MINE is set\n";
      patterns.add<VsetvlMaskedLoadOpConversion>(typeConverter, context);
      patterns.add<VsetvlMaskedStoreOpConversion>(typeConverter, context);
    } else if (getenv("TRITON_VSETVL_LANE")) {
      llvm::outs() << "!TRITON_VSETVL_LANE is set\n";
      patterns.add<VectorMaskedLoadOpConversion>(typeConverter, context);
      patterns.add<VectorMaskedStoreOpConversion>(typeConverter, context);
    }

    patterns.add<PtrToIntOpConversion>(typeConverter, context);
    patterns.add<IntToPtrOpConversion>(typeConverter, context);
    patterns.add<PtrToMemRefOpConversion>(typeConverter, context);
    patterns.add<AddPtrOpConversion>(typeConverter, context);
    patterns.add<PtrBitcastConversion>(typeConverter, context);
    patterns.add<PtrSelectConversion>(typeConverter, context);
    patterns.add<MakeTensorDescOpConversion>(typeConverter, context);
    patterns.add<ExtractMemRefOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createMemoryOpToLLVMPass() {
  return std::make_unique<MemoryOpToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
