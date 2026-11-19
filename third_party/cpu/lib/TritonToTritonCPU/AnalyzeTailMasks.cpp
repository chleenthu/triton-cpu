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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <memory>
#include <optional>

#define DEBUG_TYPE "analyze-tail-masks"

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

const char *kindName(Expr::Kind k) {
  static constexpr const char *names[] = {
      "Const", "Leaf", "Add", "Sub", "Min", "Max", "Select", "AtLeastOne"};
  static_assert(std::size(names) == Expr::AtLeastOne + 1,
               "kindName out of sync with Expr::Kind");
  return unsigned(k) < std::size(names) ? names[k] : "<unknown>";
}

// Printing flags for the debug dump: name SSA values after their NameLoc
// (%a_16, %rm_7, ...), like the TRITON_KERNEL_DUMP files, instead of %0, %1.
OpPrintingFlags debugPrintingFlags() {
  return OpPrintingFlags().printNameLocAsPrefix(true);
}

// Debug dump of an expression tree, one node per line, children indented:
//   Sub
//     Leaf %arg3
//     Const 128
LLVM_ATTRIBUTE_UNUSED void printExpr(const ExprPtr &e,
                                     llvm::raw_ostream &os = llvm::dbgs(),
                                     unsigned indent = 0) {
  os.indent(indent);
  if (!e) {
    os << "<null>\n";
    return;
  }
  os << kindName(e->kind);
  switch (e->kind) {
  case Expr::Const:
    os << " " << e->cst;
    break;
  case Expr::Leaf:
  case Expr::Select: // leaf holds the i1 condition
    os << " ";
    e->leaf.printAsOperand(os, debugPrintingFlags());
    break;
  case Expr::AtLeastOne:
    os << " size=" << e->size;
    break;
  default:
    break;
  }
  os << "\n";
  if (e->a)
    printExpr(e->a, os, indent + 2);
  if (e->b)
    printExpr(e->b, os, indent + 2);
}

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

// Splits e into base + c, where c is a constant: Add(x, Const c) -> (x, c),
// Sub(x, Const c) -> (x, -c), Const c -> (null, c), anything else -> (e, 0).
std::pair<ExprPtr, int64_t> splitConstOffset(const ExprPtr &e) {
  if (e->kind == Expr::Const)
    return {nullptr, e->cst};
  if ((e->kind == Expr::Add || e->kind == Expr::Sub) &&
      e->b->kind == Expr::Const)
    return {e->a, e->kind == Expr::Add ? e->b->cst : -e->b->cst};
  return {e, 0};
}

// Proves that bound x never covers more of a dimension of `size` elements
// than bound y. Bounds are read like vector.create_mask operands, clamped to
// [0, size], so this proves clamp(x) <= clamp(y). False means "not proven".
bool provablyLE(const ExprPtr &x, const ExprPtr &y, int64_t size) {
  if (sameExpr(x, y))
    return true;
  if (x->kind == Expr::Const && x->cst <= 0) // x covers nothing
    return true;
  if (y->kind == Expr::Const && y->cst >= size) // y covers everything
    return true;
  // Same base, constant offsets: (e + c1) <= (e + c2) iff c1 <= c2. With a
  // null base this also compares two constants.
  auto [xb, xc] = splitConstOffset(x);
  auto [yb, yc] = splitConstOffset(y);
  if (xc <= yc && (xb == yb || (xb && yb && sameExpr(xb, yb))))
    return true;
  // min(a, b) <= a, b; y <= max(a, b) if y <= a or y <= b.
  if (x->kind == Expr::Min &&
      (provablyLE(x->a, y, size) || provablyLE(x->b, y, size)))
    return true;
  if (y->kind == Expr::Max &&
      (provablyLE(x, y->a, size) || provablyLE(x, y->b, size)))
    return true;
  // max(a, b) <= y needs both; x <= min(a, b) needs both; a select needs
  // both of its arms.
  if ((x->kind == Expr::Max || x->kind == Expr::Select) &&
      provablyLE(x->a, y, size) && provablyLE(x->b, y, size))
    return true;
  if ((y->kind == Expr::Min || y->kind == Expr::Select) &&
      provablyLE(x, y->a, size) && provablyLE(x, y->b, size))
    return true;
  return false;
}

// Integer tensor description.
//  - Splat: every element is `scalar`.
//  - Range: element (i_0, ..., i_{n-1}) is `scalar + i_axis`.
struct IntInfo {
  bool isRange = false;
  int64_t axis = 0;
  ExprPtr scalar;
};

LLVM_ATTRIBUTE_UNUSED void printIntInfo(const IntInfo &x,
                                        llvm::raw_ostream &os = llvm::dbgs(),
                                        unsigned indent = 0) {
  os.indent(indent);
  if (x.isRange)
    os << "IntInfo Range axis=" << x.axis << " (scalar + i_" << x.axis
       << ")\n";
  else
    os << "IntInfo Splat\n";
  printExpr(x.scalar, os, indent + 2);
}

std::optional<SmallVector<int64_t>> getShape(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty)
    return std::nullopt;
  return SmallVector<int64_t>(ty.getShape());
}

// Index tensors whose IntInfo is already known: loop-carried index tensors
// (scf.for iter_args and results) whose scalar is carried by an extra i32
// iter_arg, and, while a loop is being checked, placeholders for them. Set by
// AnalyzeTailMasks::runOnOperation for the duration of the pass.
thread_local llvm::DenseMap<Value, IntInfo> *knownInts = nullptr;

// The expression of an i32 scalar: constants and arith.addi / arith.subi are
// looked through, anything else is a leaf. Building the expression from its
// parts keeps the bound usable where the scalar itself is not visible, e.g.
// `base + 3` computed inside an scf.if branch, used after the scf.if.
ExprPtr scalarExpr(Value v, int depth = 0) {
  APInt c;
  if (matchPattern(v, m_ConstantInt(&c)))
    return mkConst(c.getSExtValue());
  Operation *def = v.getDefiningOp();
  if (depth < kMaxDepth && isa_and_nonnull<arith::AddIOp, arith::SubIOp>(def))
    return mkBin(isa<arith::AddIOp>(def) ? Expr::Add : Expr::Sub,
                 scalarExpr(def->getOperand(0), depth + 1),
                 scalarExpr(def->getOperand(1), depth + 1));
  return mkLeaf(v);
}

std::optional<IntInfo> analyzeInt(Value v, int depth);

// The body of analyzeInt; analyzeInt wraps it with the debug dump.
std::optional<IntInfo> analyzeIntImpl(Value v, int depth) {
  // Under -debug-only=analyze-tail-masks: which case matched, inside the
  // analyzeInt block of this depth.
  auto note = [&](StringRef what) {
    (void)what;
    LLVM_DEBUG(llvm::dbgs().indent(2 * depth + 2) << what << "\n");
  };
  if (depth > kMaxDepth)
    return std::nullopt;
  auto shape = getShape(v);
  if (!shape || !getElementTypeOrSelf(v.getType()).isInteger(32))
    return std::nullopt;

  if (knownInts) {
    auto it = knownInts->find(v);
    if (it != knownInts->end()) {
      note("is loop-carried index");
      return it->second;
    }
  }

  DenseIntElementsAttr dense;
  if (matchPattern(v, m_Constant(&dense))) {
    note("is dense constant");
    if (dense.isSplat()) {
      return IntInfo{false, 0,
                     mkConst(dense.getSplatValue<APInt>().getSExtValue())};
    }
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
  // An index tensor yielded by both branches of an scf.if: both must have the
  // same shape (range along the same axis, or splat); the scalar is
  // cond ? then-scalar : else-scalar (or the common scalar if they agree).
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    note("is scf.if result");
    if (ifOp.getElseRegion().empty())
      return std::nullopt;
    unsigned idx = cast<OpResult>(v).getResultNumber();
    auto t = analyzeInt(ifOp.thenYield().getOperand(idx), depth + 1);
    auto e = t ? analyzeInt(ifOp.elseYield().getOperand(idx), depth + 1)
               : std::nullopt;
    if (!e || t->isRange != e->isRange ||
        (t->isRange && t->axis != e->axis))
      return std::nullopt;
    IntInfo res = *t;
    if (!sameExpr(t->scalar, e->scalar))
      res.scalar = mkSelect(ifOp.getCondition(), t->scalar, e->scalar);
    return res;
  }
  if (auto range = dyn_cast<MakeRangeOp>(def)) {
    note("is MakeRangeOp");
    return IntInfo{true, 0, mkConst(range.getStart())};
  }
  if (auto splat = dyn_cast<SplatOp>(def)) {
    note("is SplatOp");
    Value src = splat.getSrc();
    if (!src.getType().isInteger(32))
      return std::nullopt;
    return IntInfo{false, 0, scalarExpr(src)};
  }
  if (auto expand = dyn_cast<ExpandDimsOp>(def)) {
    note("is ExpandDimsOp");
    auto src = analyzeInt(expand.getSrc(), depth + 1);
    if (src && src->isRange && src->axis >= (int64_t)expand.getAxis())
      ++src->axis;
    return src;
  }
  if (auto bc = dyn_cast<BroadcastOp>(def)) {
    note("is BroadcastOp");
    auto src = analyzeInt(bc.getSrc(), depth + 1);
    // A size-1 axis holds a single value: the broadcast result is a splat
    // along it.
    if (src && src->isRange &&
        cast<RankedTensorType>(bc.getSrc().getType()).getDimSize(src->axis) == 1)
      src->isRange = false;
    return src;
  }
  if (auto trans = dyn_cast<TransOp>(def)) {
    note("is TransOp");
    auto src = analyzeInt(trans.getSrc(), depth + 1);
    if (src && src->isRange) {
      ArrayRef<int32_t> order = trans.getOrder();
      auto it = llvm::find(order, src->axis);
      src->axis = it - order.begin();
    }
    return src;
  }
  if (isa<arith::AddIOp, arith::SubIOp>(def)) {
    note(isa<arith::SubIOp>(def) ? "is SubIOp" : "is AddIOp");
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

// Under -debug-only=analyze-tail-masks, each call is dumped as a block nested
// by depth, with the case that matched, the calls it makes and its result:
//   analyzeInt depth=3 {
//     is ExpandDimsOp
//     analyzeInt depth=4 {
//       ...
//     }
//     IntInfo Range axis=1 (scalar + i_1)
//       Leaf %6
//   }
std::optional<IntInfo> analyzeInt(Value v, int depth) {
  unsigned indent = 2 * depth;
  LLVM_DEBUG(llvm::dbgs().indent(indent)
             << "analyzeInt depth=" << depth << " {\n");
  std::optional<IntInfo> res = analyzeIntImpl(v, depth);
  LLVM_DEBUG({
    if (res)
      printIntInfo(*res, llvm::dbgs(), indent + 2);
    else
      llvm::dbgs().indent(indent + 2) << "failed\n";
    llvm::dbgs().indent(indent) << "}\n";
  });
  return res;
}

using Box = SmallVector<ExprPtr>;

Box fullBox(ArrayRef<int64_t> shape) {
  Box box;
  for (int64_t d : shape)
    box.push_back(mkConst(d));
  return box;
}

// Debug dump of a box: one expression tree per axis.
LLVM_ATTRIBUTE_UNUSED void printBox(const Box &box,
                                    llvm::raw_ostream &os = llvm::dbgs(),
                                    unsigned indent = 0) {
  os.indent(indent) << "Box rank=" << box.size() << "\n";
  for (auto [d, e] : llvm::enumerate(box)) {
    os.indent(indent + 2) << "axis " << d << ":\n";
    printExpr(e, os, indent + 4);
  }
}

// Masks whose box is already known: loop-carried masks (scf.for iter_args
// and results) whose bounds are carried by extra i32 iter_args, and, while a
// loop is being checked, placeholder boxes for its iter_args. Set by
// AnalyzeTailMasks::runOnOperation for the duration of the pass.
thread_local llvm::DenseMap<Value, Box> *knownBoxes = nullptr;

std::optional<Box> analyzeMask(Value mask, int depth);

// The body of analyzeMask; analyzeMask wraps it with the debug dump.
std::optional<Box> analyzeMaskImpl(Value mask, int depth) {
  // Under -debug-only=analyze-tail-masks: which case matched, inside the
  // analyzeMask block of this depth.
  auto note = [&](StringRef what) {
    (void)what;
    LLVM_DEBUG(llvm::dbgs().indent(2 * depth + 2) << what << "\n");
  };
  if (depth > kMaxDepth)
    return std::nullopt;
  auto shape = getShape(mask);
  if (!shape || shape->empty() ||
      !getElementTypeOrSelf(mask.getType()).isInteger(1))
    return std::nullopt;
  int64_t rank = shape->size();

  if (knownBoxes) {
    auto it = knownBoxes->find(mask);
    if (it != knownBoxes->end()) {
      note("is loop-carried box");
      return it->second;
    }
  }

  DenseIntElementsAttr dense;
  if (matchPattern(mask, m_Constant(&dense))) {
    note("is dense constant");
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

  // A mask yielded by both branches of an scf.if: if both yielded masks are
  // boxes, the result is the box whose bound in each axis is
  // cond ? then-bound : else-bound (or the common bound if they agree).
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    note("is scf.if result");
    if (ifOp.getElseRegion().empty())
      return std::nullopt;
    unsigned idx = cast<OpResult>(mask).getResultNumber();
    auto thenBox = analyzeMask(ifOp.thenYield().getOperand(idx), depth + 1);
    auto elseBox = thenBox
                       ? analyzeMask(ifOp.elseYield().getOperand(idx), depth + 1)
                       : std::nullopt;
    if (!elseBox)
      return std::nullopt;
    Box box;
    for (int64_t d = 0; d < rank; ++d) {
      const ExprPtr &t = (*thenBox)[d], &e = (*elseBox)[d];
      box.push_back(sameExpr(t, e) ? t : mkSelect(ifOp.getCondition(), t, e));
    }
    return box;
  }

  if (auto tm = dyn_cast<triton::cpu::TailMaskOp>(def)) {
    note("is TailMaskOp");
    Box box;
    for (Value b : tm.getBounds())
      box.push_back(mkLeaf(b));
    return box;
  }
  if (auto splat = dyn_cast<SplatOp>(def)) {
    note("is SplatOp");
    // One i1 for every element: all on or all off. Turning one axis off
    // turns the whole box off.
    Box box = fullBox(*shape);
    box[0] = mkSelect(splat.getSrc(), mkConst((*shape)[0]), mkConst(0));
    return box;
  }
  if (auto expand = dyn_cast<ExpandDimsOp>(def)) {
    note("is ExpandDimsOp");
    auto src = analyzeMask(expand.getSrc(), depth + 1);
    if (src)
      src->insert(src->begin() + expand.getAxis(), mkConst(1));
    return src;
  }
  if (auto bc = dyn_cast<BroadcastOp>(def)) {
    note("is BroadcastOp");
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
    note("is TransOp");
    auto src = analyzeMask(trans.getSrc(), depth + 1);
    if (!src)
      return std::nullopt;
    Box box;
    for (int32_t d : trans.getOrder())
      box.push_back((*src)[d]);
    return box;
  }
  if (isa<arith::AndIOp, arith::OrIOp>(def)) {
    note(isa<arith::AndIOp>(def) ? "is AndIOp" : "is OrIOp");
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
    // If one box provably lies inside the other in every axis, the union is
    // the larger box.
    auto allLE = [&](const Box &a, const Box &b) {
      for (int64_t d = 0; d < rank; ++d)
        if (!provablyLE(a[d], b[d], (*shape)[d]))
          return false;
      return true;
    };
    if (allLE(*lhs, *rhs)) {
      note("lhs inside rhs");
      return *rhs;
    }
    if (allLE(*rhs, *lhs)) {
      note("rhs inside lhs");
      return *lhs;
    }
    // Otherwise the union of two boxes is a box only if they differ in at
    // most one axis (then it is the per-axis maximum there).
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
    note(("is CmpIOp " + arith::stringifyCmpIPredicate(cmp.getPredicate()))
             .str());
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

// Under -debug-only=analyze-tail-masks, each call is dumped as a block nested
// by depth, like analyzeInt, with the box it found:
//   analyzeMask depth=0 {
//     is AndIOp
//     analyzeMask depth=1 {
//       ...
//     }
//     Box rank=2
//       axis 0:
//         Sub
//           Leaf %arg3
//           Leaf %2
//       axis 1:
//         ...
//   }
std::optional<Box> analyzeMask(Value mask, int depth) {
  unsigned indent = 2 * depth;
  LLVM_DEBUG(llvm::dbgs().indent(indent)
             << "analyzeMask depth=" << depth << " {\n");
  std::optional<Box> res = analyzeMaskImpl(mask, depth);
  LLVM_DEBUG({
    if (res)
      printBox(*res, llvm::dbgs(), indent + 2);
    else
      llvm::dbgs().indent(indent + 2) << "failed\n";
    llvm::dbgs().indent(indent) << "}\n";
  });
  return res;
}

// True if every SSA value e refers to (leaves and select conditions) is
// visible at `user`. Bounds are materialized in front of the mask's user; a
// box that looked through an scf.if can refer to values defined inside a
// branch, which do not dominate a user after the if.
bool valuesDominate(const ExprPtr &e, Operation *user, DominanceInfo &dom) {
  if (!e)
    return true;
  if ((e->kind == Expr::Leaf || e->kind == Expr::Select) &&
      !dom.properlyDominates(e->leaf, user))
    return false;
  return valuesDominate(e->a, user, dom) && valuesDominate(e->b, user, dom);
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

bool isTensorOf(Value v, unsigned width) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && ty.getRank() > 0 && ty.getElementType().isInteger(width);
}

bool isKnown(Value v) { return knownBoxes->count(v) || knownInts->count(v); }

// Where a loop-carried value lives:
//  - ForIter:     scf.for iter_arg `idx` and result `idx`. Sources: the init
//                 (scalars materialized before the loop) and the yielded
//                 value (before the scf.yield).
//  - WhileBefore: scf.while before-region argument `idx`. Sources: the init
//                 (before the loop) and the after-region yielded value
//                 (before its scf.yield).
//  - WhileAfter:  scf.while after-region argument `idx` and result `idx`.
//                 Source: scf.condition argument `idx` (before the
//                 scf.condition), which the condition forwards to both.
enum class CarryKind { ForIter, WhileBefore, WhileAfter };

// A loop-carried value whose scalars get carried as extra i32 values:
//  - a mask (i1 tensor): its box, one bound per axis;
//  - an index tensor (i32 tensor): its IntInfo scalar. The rest of the
//    IntInfo (range or splat, axis) is fixed, taken from the first source.
struct Carried {
  Operation *loop;
  CarryKind kind;
  unsigned idx;
  bool isMask;
  int64_t rank = 0;              // mask: rank
  std::optional<IntInfo> shape;  // index: isRange / axis (scalar unused)
  unsigned scalar0 = 0;          // index of the first extra scalar

  unsigned numScalars() const { return isMask ? rank : 1; }

  // The values this describes.
  SmallVector<Value, 2> values() const {
    switch (kind) {
    case CarryKind::ForIter: {
      auto loop_ = cast<scf::ForOp>(loop);
      return {loop_.getRegionIterArgs()[idx], loop_.getResult(idx)};
    }
    case CarryKind::WhileBefore:
      return {cast<scf::WhileOp>(loop).getBeforeArguments()[idx]};
    case CarryKind::WhileAfter: {
      auto loop_ = cast<scf::WhileOp>(loop);
      return {loop_.getAfterArguments()[idx], loop_.getResult(idx)};
    }
    }
    llvm_unreachable("unknown CarryKind");
  }

  // The values the described ones are taken from, each with the op in front
  // of which its scalars are materialized.
  SmallVector<std::pair<Value, Operation *>, 2> sources() const {
    switch (kind) {
    case CarryKind::ForIter: {
      auto loop_ = cast<scf::ForOp>(loop);
      Operation *yield = loop_.getBody()->getTerminator();
      return {{loop_.getInitArgs()[idx], loop},
              {yield->getOperand(idx), yield}};
    }
    case CarryKind::WhileBefore: {
      auto loop_ = cast<scf::WhileOp>(loop);
      scf::YieldOp yield = loop_.getYieldOp();
      return {{loop_.getInits()[idx], loop},
              {yield->getOperand(idx), yield}};
    }
    case CarryKind::WhileAfter: {
      scf::ConditionOp cond = cast<scf::WhileOp>(loop).getConditionOp();
      return {{cond.getArgs()[idx], cond}};
    }
    }
    llvm_unreachable("unknown CarryKind");
  }

  // The operand that takes extra scalar k of source s.
  OpOperand &scalarOperand(unsigned s, unsigned k) const {
    unsigned i = scalar0 + k;
    switch (kind) {
    case CarryKind::ForIter: {
      auto loop_ = cast<scf::ForOp>(loop);
      return s == 0 ? loop_.getInitArgsMutable()[i]
                    : loop_.getBody()->getTerminator()->getOpOperand(i);
    }
    case CarryKind::WhileBefore: {
      auto loop_ = cast<scf::WhileOp>(loop);
      return s == 0 ? loop_->getOpOperand(i)
                    : loop_.getYieldOp()->getOpOperand(i);
    }
    case CarryKind::WhileAfter:
      return cast<scf::WhileOp>(loop).getConditionOp().getArgsMutable()[i];
    }
    llvm_unreachable("unknown CarryKind");
  }

  // Records v (one of values()) as described by `scalars`: the box with
  // these bounds, or the IntInfo with this scalar.
  void record(Value v, ArrayRef<ExprPtr> scalars) const {
    if (isMask)
      (*knownBoxes)[v] = Box(scalars.begin(), scalars.end());
    else
      (*knownInts)[v] = IntInfo{shape->isRange, shape->axis, scalars[0]};
  }

  // The scalars describing a source value, or nothing if it is not a box /
  // not an index tensor of the same shape.
  std::optional<SmallVector<ExprPtr>> scalarsOf(Value v) const {
    if (isMask) {
      auto box = analyzeMask(v, 0);
      if (!box)
        return std::nullopt;
      return SmallVector<ExprPtr>(*box);
    }
    auto info = analyzeInt(v, 0);
    if (!info || info->isRange != shape->isRange ||
        (info->isRange && info->axis != shape->axis))
      return std::nullopt;
    return SmallVector<ExprPtr>{info->scalar};
  }
};

void findCarried(Value v, DenseSet<Value> &seen, SmallVector<Carried> &out);

void addCarried(Operation *loop, CarryKind kind, unsigned idx, Value v,
                DenseSet<Value> &seen, SmallVector<Carried> &out) {
  for (const Carried &c : out)
    if (c.loop == loop && c.kind == kind && c.idx == idx)
      return;
  Carried c{loop, kind, idx, isTensorOf(v, 1)};
  c.rank = cast<RankedTensorType>(v.getType()).getRank();
  out.push_back(c);
  for (auto [src, at] : c.sources())
    findCarried(src, seen, out);
}

// Collects the loop-carried values (scf.for iter_args, scf.while arguments
// and results) that the mask v depends on: i1 tensors (carried masks) and
// i32 tensors (carried index tensors). Values that are already known are
// not followed.
void findCarried(Value v, DenseSet<Value> &seen, SmallVector<Carried> &out) {
  if (!(isTensorOf(v, 1) || isTensorOf(v, 32)) || isKnown(v) ||
      !seen.insert(v).second)
    return;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = arg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (arg.getArgNumber() > 0) // argument 0 is the induction var
        addCarried(forOp, CarryKind::ForIter, arg.getArgNumber() - 1, v, seen,
                   out);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      bool before = arg.getOwner() == &whileOp.getBefore().front();
      addCarried(whileOp,
                 before ? CarryKind::WhileBefore : CarryKind::WhileAfter,
                 arg.getArgNumber(), v, seen, out);
    }
    return;
  }
  auto res = cast<OpResult>(v);
  Operation *def = res.getOwner();
  if (auto forOp = dyn_cast<scf::ForOp>(def)) {
    addCarried(forOp, CarryKind::ForIter, res.getResultNumber(), v, seen, out);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
    addCarried(whileOp, CarryKind::WhileAfter, res.getResultNumber(), v, seen,
               out);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    unsigned i = res.getResultNumber();
    findCarried(ifOp.thenYield().getOperand(i), seen, out);
    if (!ifOp.getElseRegion().empty())
      findCarried(ifOp.elseYield().getOperand(i), seen, out);
  } else {
    for (Value operand : def->getOperands())
      findCarried(operand, seen, out);
  }
}

// Placeholder descriptions (the value itself as a leaf; never materialized)
// for the values of `cs`, or their removal.
void setPlaceholders(ArrayRef<Carried> cs, bool set) {
  for (const Carried &c : cs) {
    if (!c.isMask && !c.shape)
      continue;
    for (Value v : c.values()) {
      if (!set) {
        knownBoxes->erase(v);
        knownInts->erase(v);
      } else {
        c.record(v, SmallVector<ExprPtr>(c.numScalars(), mkLeaf(v)));
      }
    }
  }
}

bool scalarsDominate(ArrayRef<ExprPtr> es, Operation *user,
                     DominanceInfo &dom) {
  return llvm::all_of(
      es, [&](const ExprPtr &e) { return valuesDominate(e, user, dom); });
}

// Adds `n` i32 iter_args to an scf.for (dummy init 0, yields itself), sets
// scalar0 of its ForIter entries, and returns the new loop.
Operation *addScalars(scf::ForOp loop, ArrayRef<Carried *> group,
                      IRRewriter &rewriter) {
  OpBuilder b(loop);
  Value zero =
      arith::ConstantIntOp::create(b, loop.getLoc(), b.getI32Type(), 0);
  unsigned next = loop.getNumResults();
  SmallVector<Value> dummies;
  for (Carried *c : group) {
    c->scalar0 = next;
    next += c->numScalars();
    dummies.append(c->numScalars(), zero);
  }
  return loop
      .replaceWithAdditionalYields(
          rewriter, dummies, /*replaceInitOperandUsesInLoop=*/false,
          [](OpBuilder &, Location, ArrayRef<BlockArgument> args) {
            return SmallVector<Value>(args.begin(), args.end());
          })
      ->getOperation();
}

// Rebuilds an scf.while with extra i32 values: WhileBefore entries get
// before-region arguments (init 0, the after-region yields 0), WhileAfter
// entries get after-region arguments and results (the condition forwards 0).
// The regions are moved, so ops and block arguments inside stay the same.
Operation *addScalars(scf::WhileOp loop, ArrayRef<Carried *> group) {
  OpBuilder b(loop);
  Location loc = loop.getLoc();
  Type i32 = b.getI32Type();
  Value zero = arith::ConstantIntOp::create(b, loc, i32, 0);
  SmallVector<Value> inits(loop.getInits());
  SmallVector<Type> resultTypes(loop.getResultTypes());
  unsigned numBefore = 0, numAfter = 0;
  for (Carried *c : group) {
    if (c->kind == CarryKind::WhileBefore) {
      c->scalar0 = inits.size();
      inits.append(c->numScalars(), zero);
      numBefore += c->numScalars();
    } else {
      c->scalar0 = resultTypes.size();
      resultTypes.append(c->numScalars(), i32);
      numAfter += c->numScalars();
    }
  }
  auto newLoop = scf::WhileOp::create(b, loc, resultTypes, inits);
  newLoop->setDiscardableAttrs(loop->getDiscardableAttrDictionary());
  newLoop.getBefore().takeBody(loop.getBefore());
  newLoop.getAfter().takeBody(loop.getAfter());
  Block &before = newLoop.getBefore().front();
  Block &after = newLoop.getAfter().front();
  for (unsigned k = 0; k < numBefore; ++k)
    before.addArgument(i32, loc);
  for (unsigned k = 0; k < numAfter; ++k)
    after.addArgument(i32, loc);
  scf::ConditionOp cond = newLoop.getConditionOp();
  cond.getArgsMutable().append(SmallVector<Value>(numAfter, zero));
  scf::YieldOp yield = newLoop.getYieldOp();
  yield.getResultsMutable().append(SmallVector<Value>(numBefore, zero));
  loop->replaceAllUsesWith(
      newLoop->getResults().take_front(loop->getNumResults()));
  loop->erase();
  return newLoop;
}

// Gives loop-carried masks and index tensors explicit scalars. For each
// loop-carried value (see CarryKind) that a collected mask depends on:
//   0. Index tensors: take the shape (range or splat, axis) from the first
//      source. A source that depends on another carried value is retried
//      once that value has a shape.
//   1. Check, assuming every candidate is what it claims, that each of its
//      sources is a box / an index tensor of the same shape, with scalars
//      visible where they are materialized. Candidates that fail are dropped
//      and the rest are checked again, until none fails.
//   2. Add one i32 value per scalar to each loop (dummy values for now), and
//      record the candidates in knownBoxes / knownInts in terms of the new
//      block arguments and results.
//   3. Set the new operands (inits, yields, condition arguments) to the
//      sources' scalars.
// By induction a carried value is described by its carried scalars in every
// iteration. Step 2 comes before 3 so that loops can depend on each other's
// carried values (an inner loop starting from the outer loop's value, which
// then yields the inner loop's result; a while loop's before and after
// regions).
SmallVector<Carried> carryLoopValues(ArrayRef<OpOperand *> uses,
                                     Operation *root) {
  DenseSet<Value> seen;
  SmallVector<Carried> cs;
  for (OpOperand *use : uses)
    findCarried(use->get(), seen, cs);

  // 0. Shapes of the index tensors.
  for (bool progress = true; progress;) {
    progress = false;
    setPlaceholders(cs, true);
    for (Carried &c : cs) {
      if (c.isMask || c.shape)
        continue;
      if (auto info = analyzeInt(c.sources()[0].first, 0)) {
        c.shape = info;
        progress = true;
      }
    }
    setPlaceholders(cs, false);
  }
  llvm::erase_if(cs, [](const Carried &c) { return !c.isMask && !c.shape; });

  // 1. Check.
  while (!cs.empty()) {
    setPlaceholders(cs, true);
    DominanceInfo dom(root);
    SmallVector<Carried> ok;
    for (const Carried &c : cs) {
      bool good = true;
      for (auto [src, at] : c.sources()) {
        auto scalars = c.scalarsOf(src);
        good &= scalars && scalarsDominate(*scalars, at, dom);
      }
      if (good)
        ok.push_back(c);
    }
    setPlaceholders(cs, false);
    if (ok.size() == cs.size())
      break;
    cs = std::move(ok);
  }
  if (cs.empty())
    return cs;
  LLVM_DEBUG(llvm::dbgs() << "carrying the scalars of " << cs.size()
                          << " loop-carried value(s)\n");

  // 2. Add the scalars, one loop at a time.
  IRRewriter rewriter(root->getContext());
  llvm::MapVector<Operation *, SmallVector<Carried *>> byLoop;
  for (Carried &c : cs)
    byLoop[c.loop].push_back(&c);
  for (auto &[op, group] : byLoop) {
    Operation *newLoop =
        isa<scf::ForOp>(op)
            ? addScalars(cast<scf::ForOp>(op), group, rewriter)
            : addScalars(cast<scf::WhileOp>(op), group);
    for (Carried *c : group) {
      c->loop = newLoop;
      // The extra scalars sit at the same index as the values they come
      // with: iter_arg / result for scf.for, before or after argument (and
      // result) for scf.while.
      for (Value v : c->values()) {
        SmallVector<ExprPtr> scalars;
        for (unsigned k = 0; k < c->numScalars(); ++k) {
          Value s;
          if (auto arg = dyn_cast<BlockArgument>(v)) {
            unsigned base = arg.getArgNumber() - c->idx;
            s = arg.getOwner()->getArgument(base + c->scalar0 + k);
          } else {
            s = newLoop->getResult(c->scalar0 + k);
          }
          scalars.push_back(mkLeaf(s));
        }
        c->record(v, scalars);
      }
    }
  }

  // 3. Set the real operands. Same analysis as in step 1, with the carried
  // scalars in place of the placeholders, so it succeeds again.
  for (Carried &c : cs) {
    for (auto [s, src] : llvm::enumerate(c.sources())) {
      auto scalars = c.scalarsOf(src.first);
      assert(scalars && "carried value stopped being describable");
      OpBuilder b(src.second);
      for (unsigned k = 0; k < c.numScalars(); ++k)
        c.scalarOperand(s, k).assign(
            materialize(b, src.second->getLoc(), (*scalars)[k]));
    }
  }
  return cs;
}

bool isPureNoRegion(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         !op->hasTrait<OpTrait::IsTerminator>();
}

// The values of `root` that something with an effect needs, following
// values through scf.for / scf.while: a loop-carried value is live only if
// its block argument or result is. Plain use counts cannot see that a carried
// mask is dead once its only use is its own update, which feeds back into it
// through the yield.
DenseSet<Value> liveValues(Operation *root) {
  DenseSet<Value> live;
  SmallVector<Value> work;
  auto mark = [&](Value v) {
    if (live.insert(v).second)
      work.push_back(v);
  };
  root->walk([&](Operation *op) {
    // Loop terminators and inits are reached from the values they feed.
    if (isa<scf::YieldOp>(op) &&
        isa<scf::ForOp, scf::WhileOp>(op->getParentOp()))
      return;
    if (auto cond = dyn_cast<scf::ConditionOp>(op)) {
      mark(cond.getCondition());
      return;
    }
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      mark(loop.getLowerBound());
      mark(loop.getUpperBound());
      mark(loop.getStep());
      return;
    }
    if (isa<scf::WhileOp>(op) || isPureNoRegion(op))
      return;
    for (Value v : op->getOperands())
      mark(v);
  });
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      Operation *parent = arg.getOwner()->getParentOp();
      unsigned n = arg.getArgNumber();
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        if (n > 0) {
          mark(loop.getInitArgs()[n - 1]);
          mark(loop.getBody()->getTerminator()->getOperand(n - 1));
        }
      } else if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
        if (arg.getOwner() == &loop.getBefore().front()) {
          mark(loop.getInits()[n]);
          mark(loop.getYieldOp()->getOperand(n));
        } else {
          mark(loop.getConditionOp().getArgs()[n]);
        }
      }
      continue;
    }
    auto res = cast<OpResult>(v);
    Operation *def = res.getOwner();
    unsigned n = res.getResultNumber();
    if (auto loop = dyn_cast<scf::ForOp>(def)) {
      mark(loop.getInitArgs()[n]);
      mark(loop.getBody()->getTerminator()->getOperand(n));
    } else if (auto loop = dyn_cast<scf::WhileOp>(def)) {
      mark(loop.getConditionOp().getArgs()[n]);
    } else if (isPureNoRegion(def)) {
      for (Value operand : def->getOperands())
        mark(operand);
    }
  }
  return live;
}

// Erases the pure ops of `root` whose results are all dead and unused.
void eraseDeadOps(Operation *root, const DenseSet<Value> &live) {
  SmallVector<Operation *> ops;
  root->walk<WalkOrder::PreOrder>([&](Operation *op) { ops.push_back(op); });
  // Users come after their definitions, so erase from the back.
  for (Operation *op : llvm::reverse(ops))
    if (isPureNoRegion(op) && op->getNumResults() > 0 && op->use_empty() &&
        llvm::none_of(op->getResults(),
                      [&](Value v) { return live.contains(v); }))
      op->erase();
}

// Removes the loop-carried masks that are dead once their users read the
// carried bounds instead: the mask is still updated every iteration, but only
// its own update reads it. The cycle through the yield is cut by yielding the
// init value instead (scf.for iter_arg, scf.while before argument), the
// update ops are erased, and the region-branch canonicalization patterns
// then drop the now unused block arguments and results with their operands.
void removeDeadCarriedMasks(ArrayRef<Carried> cs, Operation *root) {
  DenseSet<Value> live = liveValues(root);
  SetVector<Operation *> loops;
  for (const Carried &c : cs) {
    if (!c.isMask || llvm::any_of(c.values(), [&](Value v) {
          return live.contains(v);
        }))
      continue;
    loops.insert(c.loop);
    if (c.kind == CarryKind::WhileAfter)
      continue; // goes away with its scf.condition operand
    auto srcs = c.sources(); // {init, yielded}
    OpOperand &yielded =
        c.kind == CarryKind::ForIter
            ? cast<scf::ForOp>(c.loop).getBody()->getTerminator()->getOpOperand(
                  c.idx)
            : cast<scf::WhileOp>(c.loop).getYieldOp()->getOpOperand(c.idx);
    yielded.assign(srcs[0].first);
  }
  if (loops.empty())
    return;
  LLVM_DEBUG(llvm::dbgs() << "removing dead carried masks from "
                          << loops.size() << " loop(s)\n");
  eraseDeadOps(root, live);
  RewritePatternSet patterns(root->getContext());
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      patterns, scf::ForOp::getOperationName());
  populateRegionBranchOpInterfaceCanonicalizationPatterns(
      patterns, scf::WhileOp::getOperationName());
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  (void)applyOpPatternsGreedily(loops.getArrayRef(), std::move(patterns),
                                config);
  // The inits of the removed masks (computed before the loops) are dead now.
  eraseDeadOps(root, live);
}

struct AnalyzeTailMasks
    : public triton::impl::AnalyzeTailMasksBase<AnalyzeTailMasks> {
  void runOnOperation() override {
    // Mask operands of loads, stores, histograms and selects (tl.where).
    SmallVector<OpOperand *> uses;
    getOperation()->walk([&](Operation *op) {
      if (auto load = dyn_cast<LoadOp>(op)) {
        if (load.getMask())
          uses.push_back(&load.getMaskMutable()[0]);
      } else if (auto store = dyn_cast<StoreOp>(op)) {
        if (store.getMask())
          uses.push_back(&store.getMaskMutable()[0]);
      } else if (auto hist = dyn_cast<HistogramOp>(op)) {
        if (hist.getMask())
          uses.push_back(&hist.getMaskMutable()[0]);
      } else if (auto sel = dyn_cast<arith::SelectOp>(op)) {
        if (isa<RankedTensorType>(sel.getCondition().getType()))
          uses.push_back(&sel->getOpOperand(0));
      }
    });

    // Loop-carried masks and index tensors: give their scf.for explicit
    // scalar iter_args.
    llvm::DenseMap<Value, Box> known;
    llvm::DenseMap<Value, IntInfo> knownIdx;
    knownBoxes = &known;
    knownInts = &knownIdx;
    auto resetKnown = llvm::scope_exit([] {
      knownBoxes = nullptr;
      knownInts = nullptr;
    });
    SmallVector<Carried> carried = carryLoopValues(uses, getOperation());

    DominanceInfo dom(getOperation());
    for (OpOperand *use : uses) {
      Value mask = use->get();
      if (mask.getDefiningOp<triton::cpu::TailMaskOp>())
        continue;
      std::optional<Box> box = analyzeMask(mask, 0);
      if (!box)
        continue;
      Operation *user = use->getOwner();
      if (!llvm::all_of(*box, [&](const ExprPtr &e) {
            return valuesDominate(e, user, dom);
          })) {
        LLVM_DEBUG(llvm::dbgs() << "bounds not visible at the user; skipped\n");
        continue;
      }
      OpBuilder b(user);
      Location loc = user->getLoc();
      SmallVector<Value> bounds;
      for (const ExprPtr &e : *box)
        bounds.push_back(materialize(b, loc, e));
      Value tail = triton::cpu::TailMaskOp::create(b, loc, mask.getType(),
                                                   bounds);
      use->set(tail);
    }

    // Carried masks whose users now all read the carried bounds.
    removeDeadCarriedMasks(carried, getOperation());
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
