// AnalyzeTailMasks (TTIR): recognize box-shaped ("tail") masks of Triton
// memory ops and selects, and make their per-dimension bounds explicit as
// triton_cpu.tail_mask. Later layers read the bounds directly instead of
// re-deriving them from the i1 tensor:
//   TTIR   tail_mask %b0, %b1 : tensor<1x1024xi1>
//   TTCIR  vector.create_mask (ConvertElementwiseOps), per-row 1-D
//          create_mask (ConvertMemoryOps), masked reduction (ConvertReductionOp)
//   LLVM   explicit vector length (vp.load / vp.store / vsetvli loops).
//
// A mask is a box if element (i_0, ..., i_{n-1}) is on iff i_d < B_d for all
// d. The analysis builds B_d as a small expression over scalars, from:
//   arith.cmpi slt/sle/sgt/sge  of (range + offset) against a splat bound
//   arith.andi                  per-dimension min
//   arith.ori                   per-dimension max where the other dims agree
//   tt.splat (i1)               all-on or all-off
//   tt.expand_dims, tt.broadcast, tt.trans, arith.constant
// with index tensors described through tt.make_range, tt.splat,
// tt.expand_dims, tt.broadcast, tt.trans, arith.addi / arith.subi and
// arith.constant. The analysis itself creates no IR; the bounds are only
// materialized for masks that are boxes.

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <memory>
#include <optional>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ANALYZETAILMASKS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

constexpr int kMaxDepth = 32;

// Scalar i32 expression.
struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;
struct Expr {
  enum Kind { Const, Leaf, Add, Sub, Min, Max, Select, AtLeastOne } kind;
  int64_t cst = 0;  // Const
  Value leaf;       // Leaf: an i32 scalar, or i1 for a Select condition
  ExprPtr a, b;     // operands (Select: a = then, b = else)
  int64_t size = 0; // AtLeastOne: a >= 1 ? size : 0
};

ExprPtr mkConst(int64_t c) {
  auto e = std::make_shared<Expr>();
  e->kind = Expr::Const;
  e->cst = c;
  return e;
}
ExprPtr mkLeaf(Value v) {
  auto e = std::make_shared<Expr>();
  e->kind = Expr::Leaf;
  e->leaf = v;
  return e;
}
ExprPtr mkBin(Expr::Kind k, ExprPtr a, ExprPtr b) {
  if (a->kind == Expr::Const && b->kind == Expr::Const) {
    switch (k) {
    case Expr::Add:
      return mkConst(a->cst + b->cst);
    case Expr::Sub:
      return mkConst(a->cst - b->cst);
    case Expr::Min:
      return mkConst(std::min(a->cst, b->cst));
    case Expr::Max:
      return mkConst(std::max(a->cst, b->cst));
    default:
      break;
    }
  }
  if (k == Expr::Add && b->kind == Expr::Const && b->cst == 0)
    return a;
  if (k == Expr::Sub && b->kind == Expr::Const && b->cst == 0)
    return a;
  auto e = std::make_shared<Expr>();
  e->kind = k;
  e->a = a;
  e->b = b;
  return e;
}
// cond (an i1 scalar) ? a : b
ExprPtr mkSelect(Value cond, ExprPtr a, ExprPtr b) {
  auto e = std::make_shared<Expr>();
  e->kind = Expr::Select;
  e->leaf = cond;
  e->a = a;
  e->b = b;
  return e;
}
// a >= 1 ? size : 0 (a size-1 dimension broadcast to `size`)
ExprPtr mkAtLeastOne(ExprPtr a, int64_t size) {
  if (a->kind == Expr::Const)
    return mkConst(a->cst >= 1 ? size : 0);
  auto e = std::make_shared<Expr>();
  e->kind = Expr::AtLeastOne;
  e->a = a;
  e->size = size;
  return e;
}

bool sameExpr(const ExprPtr &x, const ExprPtr &y) {
  if (x == y)
    return true;
  if (x->kind != y->kind || x->cst != y->cst || x->leaf != y->leaf ||
      x->size != y->size)
    return false;
  if (!x->a != !y->a || !x->b != !y->b)
    return false;
  return (!x->a || sameExpr(x->a, y->a)) && (!x->b || sameExpr(x->b, y->b));
}

// Integer tensor description.
//  - Splat: every element is `scalar`.
//  - Range: element (i_0, ..., i_{n-1}) is `scalar + i_axis`.
struct IntInfo {
  bool isRange = false;
  int64_t axis = 0;
  ExprPtr scalar;
};

std::optional<SmallVector<int64_t>> getShape(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty)
    return std::nullopt;
  return SmallVector<int64_t>(ty.getShape());
}

std::optional<IntInfo> analyzeInt(Value v, int depth) {
  if (depth > kMaxDepth)
    return std::nullopt;
  auto shape = getShape(v);
  if (!shape || !getElementTypeOrSelf(v.getType()).isInteger(32))
    return std::nullopt;

  DenseIntElementsAttr dense;
  if (matchPattern(v, m_Constant(&dense))) {
    if (dense.isSplat())
      return IntInfo{false, 0,
                     mkConst(dense.getSplatValue<APInt>().getSExtValue())};
    // A step-1 progression along exactly one axis.
    SmallVector<int64_t> vals;
    for (const APInt &x : dense.getValues<APInt>())
      vals.push_back(x.getSExtValue());
    SmallVector<int64_t> strides(shape->size(), 1);
    for (int64_t d = shape->size() - 2; d >= 0; --d)
      strides[d] = strides[d + 1] * (*shape)[d + 1];
    for (int64_t axis = 0; axis < (int64_t)shape->size(); ++axis) {
      bool ok = true;
      for (int64_t i = 0; i < (int64_t)vals.size() && ok; ++i)
        ok = vals[i] == vals[0] + (i / strides[axis]) % (*shape)[axis];
      if (ok)
        return IntInfo{true, axis, mkConst(vals[0])};
    }
    return std::nullopt;
  }

  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto range = dyn_cast<MakeRangeOp>(def))
    return IntInfo{true, 0, mkConst(range.getStart())};
  if (auto splat = dyn_cast<SplatOp>(def)) {
    Value src = splat.getSrc();
    if (!src.getType().isInteger(32))
      return std::nullopt;
    return IntInfo{false, 0, mkLeaf(src)};
  }
  if (auto expand = dyn_cast<ExpandDimsOp>(def)) {
    auto src = analyzeInt(expand.getSrc(), depth + 1);
    if (src && src->isRange && src->axis >= (int64_t)expand.getAxis())
      ++src->axis;
    return src;
  }
  if (auto bc = dyn_cast<BroadcastOp>(def)) {
    auto src = analyzeInt(bc.getSrc(), depth + 1);
    // A size-1 axis holds a single value: the broadcast result is a splat
    // along it.
    if (src && src->isRange &&
        cast<RankedTensorType>(bc.getSrc().getType()).getDimSize(src->axis) == 1)
      src->isRange = false;
    return src;
  }
  if (auto trans = dyn_cast<TransOp>(def)) {
    auto src = analyzeInt(trans.getSrc(), depth + 1);
    if (src && src->isRange) {
      ArrayRef<int32_t> order = trans.getOrder();
      auto it = llvm::find(order, src->axis);
      src->axis = it - order.begin();
    }
    return src;
  }
  if (isa<arith::AddIOp, arith::SubIOp>(def)) {
    auto lhs = analyzeInt(def->getOperand(0), depth + 1);
    auto rhs = lhs ? analyzeInt(def->getOperand(1), depth + 1) : std::nullopt;
    if (!rhs)
      return std::nullopt;
    bool isSub = isa<arith::SubIOp>(def);
    if (lhs->isRange && rhs->isRange)
      return std::nullopt;
    if (isSub && rhs->isRange)
      return std::nullopt;
    IntInfo res = lhs->isRange ? *lhs : *rhs;
    res.scalar = mkBin(isSub ? Expr::Sub : Expr::Add, lhs->scalar, rhs->scalar);
    return res;
  }
  return std::nullopt;
}

using Box = SmallVector<ExprPtr>;

Box fullBox(ArrayRef<int64_t> shape) {
  Box box;
  for (int64_t d : shape)
    box.push_back(mkConst(d));
  return box;
}

std::optional<Box> analyzeMask(Value mask, int depth) {
  if (depth > kMaxDepth)
    return std::nullopt;
  auto shape = getShape(mask);
  if (!shape || shape->empty() ||
      !getElementTypeOrSelf(mask.getType()).isInteger(1))
    return std::nullopt;
  int64_t rank = shape->size();

  DenseIntElementsAttr dense;
  if (matchPattern(mask, m_Constant(&dense))) {
    SmallVector<bool> bits(dense.getValues<bool>());
    SmallVector<int64_t> strides(rank, 1);
    for (int64_t d = rank - 2; d >= 0; --d)
      strides[d] = strides[d + 1] * (*shape)[d + 1];
    // Candidate box: along each axis, the run of ones from the origin.
    Box box;
    SmallVector<int64_t> bounds;
    for (int64_t d = 0; d < rank; ++d) {
      int64_t b = 0;
      while (b < (*shape)[d] && bits[b * strides[d]])
        ++b;
      bounds.push_back(b);
      box.push_back(mkConst(b));
    }
    for (int64_t i = 0; i < (int64_t)bits.size(); ++i) {
      bool inBox = true;
      for (int64_t d = 0; d < rank; ++d)
        inBox &= (i / strides[d]) % (*shape)[d] < bounds[d];
      if (bits[i] != inBox)
        return std::nullopt;
    }
    return box;
  }

  Operation *def = mask.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto tm = dyn_cast<triton::cpu::TailMaskOp>(def)) {
    Box box;
    for (Value b : tm.getBounds())
      box.push_back(mkLeaf(b));
    return box;
  }
  if (auto splat = dyn_cast<SplatOp>(def)) {
    // One i1 for every element: all on or all off. Turning one axis off
    // turns the whole box off.
    Box box = fullBox(*shape);
    box[0] = mkSelect(splat.getSrc(), mkConst((*shape)[0]), mkConst(0));
    return box;
  }
  if (auto expand = dyn_cast<ExpandDimsOp>(def)) {
    auto src = analyzeMask(expand.getSrc(), depth + 1);
    if (src)
      src->insert(src->begin() + expand.getAxis(), mkConst(1));
    return src;
  }
  if (auto bc = dyn_cast<BroadcastOp>(def)) {
    auto src = analyzeMask(bc.getSrc(), depth + 1);
    if (!src)
      return std::nullopt;
    auto srcShape = cast<RankedTensorType>(bc.getSrc().getType()).getShape();
    for (int64_t d = 0; d < rank; ++d)
      if (srcShape[d] == 1 && (*shape)[d] != 1)
        (*src)[d] = mkAtLeastOne((*src)[d], (*shape)[d]);
    return src;
  }
  if (auto trans = dyn_cast<TransOp>(def)) {
    auto src = analyzeMask(trans.getSrc(), depth + 1);
    if (!src)
      return std::nullopt;
    Box box;
    for (int32_t d : trans.getOrder())
      box.push_back((*src)[d]);
    return box;
  }
  if (isa<arith::AndIOp, arith::OrIOp>(def)) {
    auto lhs = analyzeMask(def->getOperand(0), depth + 1);
    auto rhs = lhs ? analyzeMask(def->getOperand(1), depth + 1) : std::nullopt;
    if (!rhs)
      return std::nullopt;
    // A bound equal to the dimension size means the whole dimension.
    auto isFull = [&](const ExprPtr &e, int64_t d) {
      return e->kind == Expr::Const && e->cst == (*shape)[d];
    };
    Box box;
    if (isa<arith::AndIOp>(def)) {
      // The intersection of two boxes is the box of per-axis minimums.
      for (int64_t d = 0; d < rank; ++d) {
        const ExprPtr &l = (*lhs)[d], &r = (*rhs)[d];
        box.push_back(isFull(l, d)   ? r
                      : isFull(r, d) ? l
                                     : mkBin(Expr::Min, l, r));
      }
      return box;
    }
    // The union of two boxes is a box only if they differ in at most one
    // axis (then it is the per-axis maximum there).
    int64_t differ = -1;
    for (int64_t d = 0; d < rank; ++d) {
      if (sameExpr((*lhs)[d], (*rhs)[d]))
        continue;
      if (differ >= 0)
        return std::nullopt;
      differ = d;
    }
    box = *lhs;
    if (differ >= 0) {
      const ExprPtr &l = (*lhs)[differ], &r = (*rhs)[differ];
      box[differ] = isFull(l, differ) || isFull(r, differ)
                        ? mkConst((*shape)[differ])
                        : mkBin(Expr::Max, l, r);
    }
    return box;
  }
  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    // range + off <  n   <=>  i_axis < n - off
    // range + off <= n   <=>  i_axis < n - off + 1
    // (and the mirrored sgt / sge). Signed only: an unsigned compare stops
    // being a prefix once n - off wraps.
    Value rangeSide, boundSide;
    int64_t extra = 0;
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::slt:
      rangeSide = cmp.getLhs(), boundSide = cmp.getRhs();
      break;
    case arith::CmpIPredicate::sle:
      rangeSide = cmp.getLhs(), boundSide = cmp.getRhs(), extra = 1;
      break;
    case arith::CmpIPredicate::sgt:
      rangeSide = cmp.getRhs(), boundSide = cmp.getLhs();
      break;
    case arith::CmpIPredicate::sge:
      rangeSide = cmp.getRhs(), boundSide = cmp.getLhs(), extra = 1;
      break;
    default:
      return std::nullopt;
    }
    auto range = analyzeInt(rangeSide, depth + 1);
    auto bound = range ? analyzeInt(boundSide, depth + 1) : std::nullopt;
    if (!bound || bound->isRange)
      return std::nullopt;
    Box box = fullBox(*shape);
    if (!range->isRange) {
      // Scalar compare broadcast to every element: all on or all off.
      return std::nullopt;
    }
    box[range->axis] =
        mkBin(Expr::Add, mkBin(Expr::Sub, bound->scalar, range->scalar),
              mkConst(extra));
    return box;
  }
  return std::nullopt;
}

Value materialize(OpBuilder &b, Location loc, const ExprPtr &e) {
  auto i32 = b.getI32Type();
  switch (e->kind) {
  case Expr::Const:
    return arith::ConstantIntOp::create(b, loc, i32, e->cst);
  case Expr::Leaf:
    return e->leaf;
  case Expr::Add:
    return arith::AddIOp::create(b, loc, materialize(b, loc, e->a),
                                 materialize(b, loc, e->b));
  case Expr::Sub:
    return arith::SubIOp::create(b, loc, materialize(b, loc, e->a),
                                 materialize(b, loc, e->b));
  case Expr::Min:
    return arith::MinSIOp::create(b, loc, materialize(b, loc, e->a),
                                  materialize(b, loc, e->b));
  case Expr::Max:
    return arith::MaxSIOp::create(b, loc, materialize(b, loc, e->a),
                                  materialize(b, loc, e->b));
  case Expr::Select:
    return arith::SelectOp::create(b, loc, e->leaf, materialize(b, loc, e->a),
                                   materialize(b, loc, e->b));
  case Expr::AtLeastOne: {
    Value a = materialize(b, loc, e->a);
    Value one = arith::ConstantIntOp::create(b, loc, i32, 1);
    Value on = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, a, one);
    return arith::SelectOp::create(
        b, loc, on, arith::ConstantIntOp::create(b, loc, i32, e->size),
        arith::ConstantIntOp::create(b, loc, i32, 0));
  }
  }
  llvm_unreachable("unknown Expr kind");
}

struct AnalyzeTailMasks
    : public triton::impl::AnalyzeTailMasksBase<AnalyzeTailMasks> {
  void runOnOperation() override {
    // Mask operands of loads, stores and selects (tl.where).
    SmallVector<OpOperand *> uses;
    getOperation()->walk([&](Operation *op) {
      if (auto load = dyn_cast<LoadOp>(op)) {
        if (load.getMask())
          uses.push_back(&load.getMaskMutable()[0]);
      } else if (auto store = dyn_cast<StoreOp>(op)) {
        if (store.getMask())
          uses.push_back(&store.getMaskMutable()[0]);
      } else if (auto sel = dyn_cast<arith::SelectOp>(op)) {
        if (isa<RankedTensorType>(sel.getCondition().getType()))
          uses.push_back(&sel->getOpOperand(0));
      }
    });

    for (OpOperand *use : uses) {
      Value mask = use->get();
      if (mask.getDefiningOp<triton::cpu::TailMaskOp>())
        continue;
      std::optional<Box> box = analyzeMask(mask, 0);
      if (!box)
        continue;
      Operation *user = use->getOwner();
      OpBuilder b(user);
      Location loc = user->getLoc();
      SmallVector<Value> bounds;
      for (const ExprPtr &e : *box)
        bounds.push_back(materialize(b, loc, e));
      Value tail = triton::cpu::TailMaskOp::create(b, loc, mask.getType(),
                                                   bounds);
      use->set(tail);
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeTailMasks() {
  return std::make_unique<AnalyzeTailMasks>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
