// ConvertFlipToShuffle (TTIR): turn tl.flip back into a lane shuffle.
//
// Triton has no flip op. tl.flip (python/triton/language/standard.py) bitcasts
// to int, reshapes the flipped dim of size 2^k to (2, ..., 2) and runs k steps
//   %r = tt.reduce(%y) {axis = a} (xori)    ; a ^ b of the two halves
//   %e = tt.expand_dims %r {axis = a}
//   %b = tt.broadcast %e                    ; back to the shape of %y
//   %z = arith.xori %y, %b                  ; a ^ (a^b) = b: swaps the halves
// Each step swaps the two entries of one size-2 axis, i.e. flips one bit of
// the lane index, so the whole chain is a constant permutation of %y. Lowered
// literally it becomes extract / xori / broadcast chains (vxor + slides on
// RVV); as one vector.shuffle LLVM emits a single gather instead (vid / vrsub
// / vrgather.vv for a lane reverse).
//
// The permutation only touches the axes in [amin, amax] (the outermost and
// innermost flipped axes), so %y is viewed as O x M x B (O = dims before amin,
// M = dims in [amin, amax], B = dims after amax) and only M is shuffled, once
// per O, moving whole B-element blocks:
//   %v = unrealized_conversion_cast %y : tensor<S> to vector<S>
//   %m = vector.shape_cast %v : vector<S> to vector<OxMxB>
//   %s_o = vector.shuffle %m[o], %m[o] [perm]     ; for o in 0..O-1
//   %z' = shape_cast + unrealized_conversion_cast back to tensor<S>
// The reshapes around the chain are left alone; they lower to shape_casts.

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <memory>
#include <optional>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTFLIPTOSHUFFLE
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

struct FlipStep {
  Value y;   // the step's input
  int axis;  // the size-2 axis it swaps
  SmallVector<Operation *, 4> ops; // xori, broadcast, expand_dims, reduce
};

// A reduce over one operand whose combiner is a single xori of its two
// arguments.
static bool isXorReduce(ReduceOp reduce) {
  if (reduce.getNumOperands() != 1 || reduce->getNumResults() != 1)
    return false;
  Block &body = reduce.getCombineOp().front();
  if (body.getNumArguments() != 2 ||
      !llvm::hasSingleElement(body.without_terminator()))
    return false;
  auto xorOp = dyn_cast<arith::XOrIOp>(body.front());
  auto ret = dyn_cast<ReduceReturnOp>(body.getTerminator());
  if (!xorOp || !ret || ret.getNumOperands() != 1 ||
      ret.getOperand(0) != xorOp.getResult())
    return false;
  Value a = body.getArgument(0), b = body.getArgument(1);
  return (xorOp.getLhs() == a && xorOp.getRhs() == b) ||
         (xorOp.getLhs() == b && xorOp.getRhs() == a);
}

// %z = arith.xori %y, broadcast(expand_dims(xor_reduce(%y, a), a)) with
// %y.shape[a] == 2, in either operand order.
static std::optional<FlipStep> matchStep(Value z) {
  auto xorOp = z.getDefiningOp<arith::XOrIOp>();
  if (!xorOp || !isa<RankedTensorType>(z.getType()))
    return std::nullopt;
  for (int i = 0; i < 2; ++i) {
    Value y = xorOp->getOperand(i), other = xorOp->getOperand(1 - i);
    auto bcast = other.getDefiningOp<BroadcastOp>();
    if (!bcast || bcast.getType() != y.getType())
      continue;
    auto expand = bcast.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expand)
      continue;
    auto reduce = expand.getSrc().getDefiningOp<ReduceOp>();
    if (!reduce || !isXorReduce(reduce) || reduce.getOperand(0) != y)
      continue;
    int axis = reduce.getAxis();
    if (expand.getAxis() != axis ||
        cast<RankedTensorType>(y.getType()).getShape()[axis] != 2)
      continue;
    return FlipStep{y, axis, {xorOp, bcast, expand, reduce}};
  }
  return std::nullopt;
}

struct ConvertFlipToShuffle
    : public triton::impl::ConvertFlipToShuffleBase<ConvertFlipToShuffle> {
  using ConvertFlipToShuffleBase::ConvertFlipToShuffleBase;

  void runOnOperation() override {
    // Chain ends: steps whose result is not the input of another step.
    SmallVector<Value> ends;
    getOperation()->walk([&](arith::XOrIOp op) {
      if (!matchStep(op.getResult()))
        return;
      bool feedsStep = llvm::any_of(op->getUsers(), [&](Operation *user) {
        auto next = user->getNumResults() == 1
                        ? matchStep(user->getResult(0))
                        : std::nullopt;
        return next && next->y == op.getResult();
      });
      if (!feedsStep)
        ends.push_back(op.getResult());
    });

    for (Value end : ends)
      rewriteChain(end);
  }

  void rewriteChain(Value end) {
    // Walk back to the chain's input, collecting the swapped axes.
    SmallVector<int> axes;
    SmallVector<Operation *> ops; // from the end towards the input
    Value base = end;
    while (auto step = matchStep(base)) {
      axes.push_back(step->axis);
      ops.append(step->ops.begin(), step->ops.end());
      base = step->y;
    }

    auto tensorTy = cast<RankedTensorType>(base.getType());
    ArrayRef<int64_t> shape = tensorTy.getShape();
    int rank = shape.size();

    // Swapping an axis twice cancels out.
    SmallVector<bool> flipped(rank, false);
    for (int a : axes)
      flipped[a] = !flipped[a];
    int amin = -1, amax = -1;
    for (int d = 0; d < rank; ++d) {
      if (!flipped[d])
        continue;
      if (amin < 0)
        amin = d;
      amax = d;
    }

    OpBuilder b(end.getDefiningOp());
    Location loc = end.getLoc();
    Value result = base;
    if (amin >= 0) {
      int64_t outer = 1, inner = 1;
      for (int d = 0; d < amin; ++d)
        outer *= shape[d];
      for (int d = amax + 1; d < rank; ++d)
        inner *= shape[d];
      ArrayRef<int64_t> midShape = shape.slice(amin, amax - amin + 1);
      int64_t mid = 1;
      for (int64_t s : midShape)
        mid *= s;

      // perm[m] = source position of mid position m: flip every swapped axis
      // in the (row-major) coordinates of m.
      SmallVector<int64_t> perm(mid);
      for (int64_t m = 0; m < mid; ++m) {
        int64_t rest = m, src = 0, stride = 1;
        for (int d = amax; d >= amin; --d) {
          int64_t size = shape[d], c = rest % size;
          rest /= size;
          if (flipped[d])
            c = size - 1 - c;
          src += c * stride;
          stride *= size;
        }
        perm[m] = src;
      }

      Type elemTy = tensorTy.getElementType();
      auto vecTy = VectorType::get(shape, elemTy);
      SmallVector<int64_t> innerShape{mid};
      if (inner != 1)
        innerShape.push_back(inner);
      auto innerTy = VectorType::get(innerShape, elemTy);

      Value vec =
          UnrealizedConversionCastOp::create(b, loc, vecTy, base).getResult(0);
      Value shuffled;
      if (outer == 1) {
        Value v = vector::ShapeCastOp::create(b, loc, innerTy, vec);
        shuffled = vector::ShuffleOp::create(b, loc, v, v, perm);
      } else {
        SmallVector<int64_t> blockShape{outer};
        blockShape.append(innerShape);
        auto blockTy = VectorType::get(blockShape, elemTy);
        Value v = vector::ShapeCastOp::create(b, loc, blockTy, vec);
        Value acc = arith::ConstantOp::create(
            b, loc, blockTy, cast<TypedAttr>(b.getZeroAttr(blockTy)));
        for (int64_t o = 0; o < outer; ++o) {
          Value row = vector::ExtractOp::create(b, loc, v, o);
          Value s = vector::ShuffleOp::create(b, loc, row, row, perm);
          acc = vector::InsertOp::create(b, loc, s, acc, o);
        }
        shuffled = acc;
      }
      Value back = vector::ShapeCastOp::create(b, loc, vecTy, shuffled);
      result = UnrealizedConversionCastOp::create(b, loc, tensorTy, back)
                   .getResult(0);
    }

    end.replaceAllUsesWith(result);
    for (Operation *op : ops)
      if (op->use_empty())
        op->erase();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertFlipToShuffle() {
  return std::make_unique<ConvertFlipToShuffle>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
