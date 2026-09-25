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


struct VsetvlMaskedLoadOpConversion : public OpConversionPattern<vector::MaskedLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Type resTy = getTypeConverter()->convertType(op.getType());
    Value memref = adaptor.getBase();
    Value ptr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, memref, ArrayRef<int64_t>{1});

    auto vecTy = mlir::dyn_cast<VectorType>(resTy);
    if (!vecTy) {
      return failure();
    }
    auto i64Ty = IntegerType::get(getContext(), 64);
    auto i32Ty = IntegerType::get(getContext(), 32);
    auto i1Ty = IntegerType::get(getContext(), 1);
    auto one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, resTy, one, /*alignment=*/16);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));

    Block *currentBlock = rewriter.getBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *headerBlock = rewriter.createBlock(currentBlock->getParent(), continueBlock->getIterator());
    headerBlock->addArgument(i64Ty, loc);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, headerBlock);
    rewriter.setInsertionPointToStart(headerBlock);

    int64_t vecSize = vecTy.getNumElements();
    auto constVecSize = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(vecSize));
    auto constVecSizeExt = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(vecSize));
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    Value nElements = funcOp.getArgument(3);
    Value pidX = funcOp.getArgument(4);
    Value blockStart = LLVM::MulOp::create(rewriter, loc, i32Ty, pidX, constVecSize);
    auto blockStartExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, blockStart);
    auto nElementsExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, nElements);
    Value remaining = LLVM::SubOp::create(rewriter, loc, i64Ty, nElementsExt, blockStartExt);
    auto cmpVecSize = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::slt, remaining, constVecSizeExt);
    Value clampElements = LLVM::SelectOp::create(rewriter, loc, i64Ty, cmpVecSize, remaining, constVecSizeExt);
    auto two = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(2));
    auto three = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(3));
    Value iv = headerBlock->getArgument(0);
    auto remain = LLVM::SubOp::create(rewriter, loc, i64Ty, clampElements, iv);
    auto vl = LLVM::CallIntrinsicOp::create(rewriter, loc, i64Ty,
                rewriter.getStringAttr("llvm.riscv.vsetvli.i64"),
                ArrayRef<Value>{remain, two, three}).getResult(0);
    auto vlTrunc = LLVM::TruncOp::create(rewriter, loc, i32Ty, vl);

    Type elemTy = vecTy.getElementType();
    auto loadPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, ptr,
                    ArrayRef<LLVM::GEPArg>{iv});
    VectorType nxv16f32Ty = VectorType::get({16}, rewriter.getF32Type(), /*scalable=*/true);
    VectorType nxv16i1Ty = VectorType::get({16}, i1Ty, /*scalable=*/true);
    auto splatAttr = SplatElementsAttr::get(nxv16i1Ty, rewriter.getBoolAttr(true));
    auto allTrueMask = LLVM::ConstantOp::create(rewriter, loc, nxv16i1Ty, splatAttr);
    auto emptyDict = rewriter.getDictionaryAttr({});
    auto alignAttr = rewriter.getNamedAttr("llvm.align", rewriter.getI64IntegerAttr(4));
    auto ptrAttrDict = DictionaryAttr::get(rewriter.getContext(), {alignAttr});
    auto vpLoadOp = LLVM::CallIntrinsicOp::create(rewriter, loc, nxv16f32Ty,
                    rewriter.getStringAttr("llvm.vp.load.nxv16f32.p0"),
                    ArrayRef<Value>{loadPtr, allTrueMask, vlTrunc});
    Value vpLoad = vpLoadOp.getResult(0);

    Value destPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, alloca,
                    ArrayRef<LLVM::GEPArg>{iv});
    auto vpStoreOp = LLVM::CallIntrinsicOp::create(rewriter, loc,
                    rewriter.getStringAttr("llvm.vp.store.nxv16f32.p0"),
                    ValueRange({vpLoad, destPtr, allTrueMask, vlTrunc}));

    auto iNext = LLVM::AddOp::create(rewriter, loc, i64Ty, iv, vl);
    auto done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge, iNext, clampElements);
    LLVM::CondBrOp::create(rewriter, loc, done,
                                    continueBlock, ValueRange{},
                                    headerBlock, ValueRange{iNext});

    rewriter.setInsertionPointToStart(continueBlock);
    auto finalLoad = LLVM::LoadOp::create(rewriter, loc, resTy, alloca, /*alignment=*/16);

    rewriter.replaceOp(op, finalLoad);
    return success();
  }
};

struct VetvlMaskedStoreOpConversion : public OpConversionPattern<vector::MaskedStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::MaskedStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *typeConverter = getTypeConverter();
    Type ptrTy = LLVM::LLVMPointerType::get(getContext());
    Type valTy = typeConverter->convertType(op.getValueToStore().getType());
    Value memref = adaptor.getBase();
    Value valueToStore = adaptor.getValueToStore();

    Value ptr = LLVM::ExtractValueOp::create(rewriter, loc, ptrTy, memref, ArrayRef<int64_t>{1});

    auto vecTy = mlir::dyn_cast<VectorType>(valTy);
    if (!vecTy) {
      return failure();
    }
    auto i64Ty = IntegerType::get(getContext(), 64);
    auto i32Ty = IntegerType::get(getContext(), 32);
    auto i1Ty = IntegerType::get(getContext(), 1);
    auto one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1));
    Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrTy, valTy, one, /*alignment=*/16);
    auto zero = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(0));
    LLVM::StoreOp::create(rewriter, loc, valueToStore, alloca, /*alignment=*/16);

    Block *currentBlock = rewriter.getBlock();
    Block *continueBlock = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *headerBlock = rewriter.createBlock(currentBlock->getParent(), continueBlock->getIterator());
    headerBlock->addArgument(i64Ty, loc);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange{zero}, headerBlock);
    rewriter.setInsertionPointToStart(headerBlock);

    int64_t vecSize = vecTy.getNumElements();
    auto constVecSize = LLVM::ConstantOp::create(rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(vecSize));
    auto constVecSizeExt = LLVM::ConstantOp::create(rewriter, loc, i64Ty, rewriter.getI64IntegerAttr(vecSize));
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    Value nElements = funcOp.getArgument(3);
    Value pidX = funcOp.getArgument(4);
    Value blockStart = LLVM::MulOp::create(rewriter, loc, i32Ty, pidX, constVecSize);
    auto blockStartExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, blockStart);
    auto nElementsExt = LLVM::SExtOp::create(rewriter, loc, i64Ty, nElements);
    Value remaining = LLVM::SubOp::create(rewriter, loc, i64Ty, nElementsExt, blockStartExt);
    auto cmpVecSize = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::slt, remaining, constVecSizeExt);
    Value clampElements = LLVM::SelectOp::create(rewriter, loc, i64Ty, cmpVecSize, remaining, constVecSizeExt);
    auto two = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(2));
    auto three = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                    rewriter.getI64IntegerAttr(3));
    Value iv = headerBlock->getArgument(0);
    auto remain = LLVM::SubOp::create(rewriter, loc, i64Ty, clampElements, iv);
    auto vl = LLVM::CallIntrinsicOp::create(rewriter, loc, i64Ty,
                rewriter.getStringAttr("llvm.riscv.vsetvli.i64"),
                ValueRange({remain, two, three})).getResult(0);
    auto vlTrunc = LLVM::TruncOp::create(rewriter, loc, i32Ty, vl);

    Type elemTy = vecTy.getElementType();
    auto loadPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, alloca,
                    ArrayRef<LLVM::GEPArg>{iv});
    VectorType nxv16f32Ty = VectorType::get({16}, rewriter.getF32Type(), /*scalable=*/true);
    VectorType nxv16i1Ty = VectorType::get({16}, i1Ty, /*scalable=*/true);
    auto splatAttr = SplatElementsAttr::get(nxv16i1Ty, rewriter.getBoolAttr(true));
    auto allTrueMask = LLVM::ConstantOp::create(rewriter, loc, nxv16i1Ty, splatAttr);
    auto emptyDict = rewriter.getDictionaryAttr({});
    auto alignAttr = rewriter.getNamedAttr("llvm.align", rewriter.getI64IntegerAttr(4));
    auto ptrAttrDict = DictionaryAttr::get(rewriter.getContext(), {alignAttr});
    auto vpLoadOp = LLVM::CallIntrinsicOp::create(rewriter, loc, nxv16f32Ty,
                    rewriter.getStringAttr("llvm.vp.load.nxv16f32.p0"),
                    ValueRange({loadPtr, allTrueMask, vlTrunc}));
    Value vpLoad = vpLoadOp.getResult(0);

    auto destPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, ptr,
                    ArrayRef<LLVM::GEPArg>{iv});//,
    auto vpStoreOp = LLVM::CallIntrinsicOp::create(rewriter, loc,
                    rewriter.getStringAttr("llvm.vp.store.nxv16f32.p0"),
                    ValueRange({vpLoad, destPtr, allTrueMask, vlTrunc}));

    auto iNext = LLVM::AddOp::create(rewriter, loc, i64Ty, iv, vl);
    auto done = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::uge, iNext, clampElements);
    LLVM::CondBrOp::create(rewriter, loc, done,
                                    continueBlock, ValueRange{},
                                    headerBlock, ValueRange{iNext});

    rewriter.setInsertionPointToStart(continueBlock);
    rewriter.eraseOp(op);

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

    char* vsetvl_env = getenv("TRITON_VSETVL_MINE");
    if (vsetvl_env) {
      llvm::outs()<<"!TRITON_VSETVL_MINE is set\n";
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
