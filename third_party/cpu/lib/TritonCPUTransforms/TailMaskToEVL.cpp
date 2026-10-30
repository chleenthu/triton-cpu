#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <optional>

#define DEBUG_TYPE "tail-mask-to-evl"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_TAILMASKTOEVL
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Recursion limit for walking the ops that form a mask.
constexpr int kMaxDepth = 16;

// Number of lanes of a vector whose dims are all 1 except the innermost one,
// i.e. a vector laid out exactly like a 1-D vector of that length.
std::optional<int64_t> getEffective1DLength(Type type) {
  auto vecTy = dyn_cast<VectorType>(type);
  if (!vecTy || vecTy.isScalable() || vecTy.getRank() == 0)
    return std::nullopt;
  for (int64_t dim : vecTy.getShape().drop_back())
    if (dim != 1)
      return std::nullopt;
  return vecTy.getShape().back();
}

// A scalar `value + cst`. `value` may be null (just `cst`), a scalar, or a
// single-element vector holding the scalar.
struct Scalar {
  Value value;
  int64_t cst = 0;
};

// Tail length L of a mask: lanes [0, L) on, the rest off. Not clamped:
// L <= 0 means all lanes off and L >= N all lanes on, as for
// vector.create_mask.
struct TailLen {
  enum Kind {
    Const,   // cst
    Index,   // value (index-typed)
    Positive, // lhs > 0 ? n : 0 (a unit leading dim of an n-D create_mask)
    AllOrNone, // cond (i1 scalar) ? n : 0
    Min,
    Max,
    Compare, // bound - off + cst, compared in intTy
  } kind;
  int64_t cst = 0;
  int64_t n = 0;
  Value value;
  Scalar cond, bound, off;
  Type intTy;
  std::shared_ptr<TailLen> lhs, rhs;
};
using TailLenPtr = std::shared_ptr<TailLen>;

const char *kindName(TailLen::Kind k) {
  static constexpr const char *names[] = {
      "Const", "Index", "Positive", "AllOrNone", "Min", "Max", "Compare"};
  static_assert(std::size(names) == TailLen::Compare + 1,
                "kindName out of sync with TailLen::Kind");
  return unsigned(k) < std::size(names) ? names[k] : "<unknown>";
}

// Printing flags for the debug dump: name SSA values after their NameLoc
// (%a_16, %rm_7, ...), like the TRITON_KERNEL_DUMP files, instead of %0, %1.
OpPrintingFlags debugPrintingFlags() {
  return OpPrintingFlags().printNameLocAsPrefix(true);
}

// Debug dump of a Scalar as `value + cst` (just `cst` without a value).
LLVM_ATTRIBUTE_UNUSED void printScalar(const Scalar &s,
                                       llvm::raw_ostream &os = llvm::dbgs()) {
  if (!s.value) {
    os << s.cst;
    return;
  }
  s.value.printAsOperand(os, debugPrintingFlags());
  if (s.cst)
    os << " + " << s.cst;
}

// Debug dump of a tail length tree, one node per line, children indented:
//   Min
//     Positive n=16
//       Index %5
//     Compare bound=%n off=%3 extra=0 ty=i32
LLVM_ATTRIBUTE_UNUSED void printTailLen(const TailLenPtr &len,
                                        llvm::raw_ostream &os = llvm::dbgs(),
                                        unsigned indent = 0) {
  os.indent(indent);
  if (!len) {
    os << "<null>\n";
    return;
  }
  os << kindName(len->kind);
  switch (len->kind) {
  case TailLen::Const:
    os << " " << len->cst;
    break;
  case TailLen::Index:
    os << " ";
    len->value.printAsOperand(os, debugPrintingFlags());
    break;
  case TailLen::Positive:
    os << " n=" << len->n;
    break;
  case TailLen::AllOrNone:
    os << " n=" << len->n << " cond=";
    printScalar(len->cond, os);
    break;
  case TailLen::Compare:
    os << " bound=";
    printScalar(len->bound, os);
    os << " off=";
    printScalar(len->off, os);
    os << " extra=" << len->cst << " ty=" << len->intTy;
    break;
  default:
    break;
  }
  os << "\n";
  if (len->lhs)
    printTailLen(len->lhs, os, indent + 2);
  if (len->rhs)
    printTailLen(len->rhs, os, indent + 2);
}

// `v` is a scalar broadcast to all lanes (vector.broadcast of a scalar or of
// a single-element vector, or a splat constant).
std::optional<Scalar> matchSplat(Value v, int depth) {
  if (depth > kMaxDepth)
    return std::nullopt;
  DenseIntElementsAttr dense;
  if (matchPattern(v, m_Constant(&dense)) && dense.isSplat())
    return Scalar{Value(), dense.getSplatValue<APInt>().getSExtValue()};
  if (auto bc = v.getDefiningOp<vector::BroadcastOp>()) {
    Value src = bc.getSource();
    auto srcTy = dyn_cast<VectorType>(src.getType());
    if (!srcTy)
      return Scalar{src, 0};
    if (srcTy.isScalable() || srcTy.getNumElements() != 1)
      return std::nullopt;
    if (auto inner = matchSplat(src, depth + 1))
      return inner;
    return Scalar{src, 0};
  }
  if (auto sc = v.getDefiningOp<vector::ShapeCastOp>())
    return matchSplat(sc.getSource(), depth + 1);
  return std::nullopt;
}

// `v` holds `off + i` in lane i (along its effectively 1-D lanes).
std::optional<Scalar> matchIotaPlusOffset(Value v, int depth) {
  if (depth > kMaxDepth || !getEffective1DLength(v.getType()))
    return std::nullopt;
  DenseIntElementsAttr dense;
  if (matchPattern(v, m_Constant(&dense))) {
    auto values = dense.getValues<APInt>();
    int64_t start = (*values.begin()).getSExtValue();
    int64_t i = 0;
    for (const APInt &val : values)
      if (val.getSExtValue() != start + i++)
        return std::nullopt;
    return Scalar{Value(), start};
  }
  if (v.getDefiningOp<vector::StepOp>())
    return Scalar{Value(), 0};
  if (auto sc = v.getDefiningOp<vector::ShapeCastOp>()) {
    if (!getEffective1DLength(sc.getSource().getType()))
      return std::nullopt;
    return matchIotaPlusOffset(sc.getSource(), depth + 1);
  }
  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    for (auto [iotaSide, splatSide] :
         {std::pair(add.getLhs(), add.getRhs()),
          std::pair(add.getRhs(), add.getLhs())}) {
      auto iota = matchIotaPlusOffset(iotaSide, depth + 1);
      auto splat = iota ? matchSplat(splatSide, depth + 1) : std::nullopt;
      // At most one runtime term keeps the offset a single `value + cst`.
      if (splat && !(iota->value && splat->value))
        return Scalar{iota->value ? iota->value : splat->value,
                      iota->cst + splat->cst};
    }
  }
  return std::nullopt;
}

TailLenPtr makeLen(TailLen::Kind kind) {
  auto len = std::make_shared<TailLen>();
  len->kind = kind;
  return len;
}

// Length given as a scalar (the operand of a 1-D vector.create_mask), seen
// through the ops AnalyzeTailMasks / ConvertMemoryOps build it from, so it
// can be compared with other lengths:
//   constant -> Const, select(c, x, 0) -> min(c ? n : 0, x),
//   minsi / maxsi -> Min / Max, index_cast / extsi -> the source.
// Anything else is an opaque Index value.
TailLenPtr analyzeScalarLength(Value v, int64_t n, int depth);

// The body of analyzeScalarLength; analyzeScalarLength wraps it with the
// debug dump.
TailLenPtr analyzeScalarLengthImpl(Value v, int64_t n, int depth) {
  // Under -debug-only=tail-mask-to-evl: which case matched, inside the
  // analyzeScalarLength block of this depth.
  auto note = [&](StringRef what) {
    (void)what;
    LLVM_DEBUG(llvm::dbgs().indent(2 * depth + 2) << what << "\n");
  };
  auto opaque = [&] {
    note("opaque");
    auto len = makeLen(TailLen::Index);
    len->value = v;
    return len;
  };
  if (depth > kMaxDepth)
    return opaque();
  IntegerAttr cstAttr;
  if (matchPattern(v, m_Constant(&cstAttr))) {
    note("is constant");
    auto len = makeLen(TailLen::Const);
    len->cst = cstAttr.getValue().getSExtValue();
    return len;
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return opaque();
  if (isa<arith::IndexCastOp, arith::ExtSIOp>(def)) {
    note(isa<arith::IndexCastOp>(def) ? "is IndexCastOp" : "is ExtSIOp");
    return analyzeScalarLength(def->getOperand(0), n, depth + 1);
  }
  if (isa<arith::MinSIOp, arith::MaxSIOp>(def)) {
    note(isa<arith::MinSIOp>(def) ? "is MinSIOp" : "is MaxSIOp");
    auto len = makeLen(isa<arith::MinSIOp>(def) ? TailLen::Min : TailLen::Max);
    len->lhs = analyzeScalarLength(def->getOperand(0), n, depth + 1);
    len->rhs = analyzeScalarLength(def->getOperand(1), n, depth + 1);
    return len;
  }
  if (auto sel = dyn_cast<arith::SelectOp>(def)) {
    IntegerAttr elseAttr;
    if (!sel.getCondition().getType().isInteger(1) ||
        !matchPattern(sel.getFalseValue(), m_Constant(&elseAttr)) ||
        !elseAttr.getValue().isZero())
      return opaque();
    note("is SelectOp (c ? x : 0)");
    // c ? x : 0 == min(c ? n : 0, x) for the clamped length semantics.
    auto cond = makeLen(TailLen::AllOrNone);
    cond->cond = Scalar{sel.getCondition(), 0};
    cond->n = n;
    auto len = makeLen(TailLen::Min);
    len->lhs = cond;
    len->rhs = analyzeScalarLength(sel.getTrueValue(), n, depth + 1);
    return len;
  }
  return opaque();
}

// Under -debug-only=tail-mask-to-evl, each call is dumped as a block nested
// by depth, like analyzeTailLength, with the length it found.
TailLenPtr analyzeScalarLength(Value v, int64_t n, int depth) {
  unsigned indent = 2 * depth;
  LLVM_DEBUG(llvm::dbgs().indent(indent)
             << "analyzeScalarLength depth=" << depth << " {\n");
  TailLenPtr res = analyzeScalarLengthImpl(v, n, depth);
  LLVM_DEBUG({
    printTailLen(res, llvm::dbgs(), indent + 2);
    llvm::dbgs().indent(indent) << "}\n";
  });
  return res;
}

// Pure analysis: no IR is created.
TailLenPtr analyzeTailLength(Value mask, int depth);

// The body of analyzeTailLength; analyzeTailLength wraps it with the debug
// dump.
TailLenPtr analyzeTailLengthImpl(Value mask, int depth) {
  // Under -debug-only=tail-mask-to-evl: which case matched, inside the
  // analyzeTailLength block of this depth.
  auto note = [&](StringRef what) {
    (void)what;
    LLVM_DEBUG(llvm::dbgs().indent(2 * depth + 2) << what << "\n");
  };
  if (depth > kMaxDepth)
    return nullptr;
  std::optional<int64_t> n = getEffective1DLength(mask.getType());
  if (!n)
    return nullptr;

  DenseIntElementsAttr dense;
  if (matchPattern(mask, m_Constant(&dense))) {
    note("is dense constant");
    int64_t count = 0;
    bool seenFalse = false;
    for (bool bit : dense.getValues<bool>()) {
      if (bit && seenFalse)
        return nullptr;
      if (bit)
        ++count;
      else
        seenFalse = true;
    }
    auto len = makeLen(TailLen::Const);
    len->cst = count;
    return len;
  }

  Operation *def = mask.getDefiningOp();
  if (!def)
    return nullptr;

  if (auto cm = dyn_cast<vector::ConstantMaskOp>(def)) {
    note("is ConstantMaskOp");
    auto len = makeLen(TailLen::Const);
    ArrayRef<int64_t> dims = cm.getMaskDimSizes();
    len->cst = llvm::is_contained(dims.drop_back(), 0) ? 0 : dims.back();
    return len;
  }
  if (auto cm = dyn_cast<vector::CreateMaskOp>(def)) {
    note("is CreateMaskOp");
    // An n-D create_mask with unit leading dims (from triton_cpu.tail_mask
    // of a [1, ..., 1, N] tensor) is identified by the op itself, so masks
    // that share it compare equal.
    // n-D create_mask with unit leading dims (triton_cpu.tail_mask of a
    // [1, ..., 1, N] tensor): lanes are on iff every leading bound is > 0
    // and the lane is below the last bound.
    TailLenPtr len = analyzeScalarLength(cm.getOperands().back(), *n, depth + 1);
    for (Value bound : cm.getOperands().drop_back()) {
      auto lead = makeLen(TailLen::Positive);
      lead->lhs = analyzeScalarLength(bound, *n, depth + 1);
      lead->n = *n;
      auto both = makeLen(TailLen::Min);
      both->lhs = lead;
      both->rhs = len;
      len = both;
    }
    return len;
  }
  if (auto sc = dyn_cast<vector::ShapeCastOp>(def)) {
    note("is ShapeCastOp");
    if (getEffective1DLength(sc.getSource().getType()) != n)
      return nullptr;
    return analyzeTailLength(sc.getSource(), depth + 1);
  }
  if (isa<vector::BroadcastOp>(def)) {
    note("is BroadcastOp");
    auto cond = matchSplat(mask, depth + 1);
    if (!cond || !cond->value)
      return nullptr;
    auto len = makeLen(TailLen::AllOrNone);
    len->cond = *cond;
    len->n = *n;
    return len;
  }
  if (isa<arith::AndIOp, arith::OrIOp>(def)) {
    note(isa<arith::AndIOp>(def) ? "is AndIOp" : "is OrIOp");
    auto lhs = analyzeTailLength(def->getOperand(0), depth + 1);
    auto rhs = lhs ? analyzeTailLength(def->getOperand(1), depth + 1) : nullptr;
    if (!rhs)
      return nullptr;
    // (lane < a) & (lane < b) == lane < min(a, b); | gives max.
    auto len = makeLen(isa<arith::AndIOp>(def) ? TailLen::Min : TailLen::Max);
    len->lhs = lhs;
    len->rhs = rhs;
    return len;
  }
  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    note(("is CmpIOp " + arith::stringifyCmpIPredicate(cmp.getPredicate()))
             .str());
    // off + lane <  bound  <=>  lane < bound - off
    // off + lane <= bound  <=>  lane < bound - off + 1
    // and the mirrored sgt/sge forms. Signed predicates only: Triton offsets
    // and bounds are signed, and an unsigned compare is not a tail mask once
    // bound - off wraps.
    Value iotaSide, boundSide;
    int64_t extra = 0;
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::slt:
      iotaSide = cmp.getLhs(), boundSide = cmp.getRhs();
      break;
    case arith::CmpIPredicate::sle:
      iotaSide = cmp.getLhs(), boundSide = cmp.getRhs(), extra = 1;
      break;
    case arith::CmpIPredicate::sgt:
      iotaSide = cmp.getRhs(), boundSide = cmp.getLhs();
      break;
    case arith::CmpIPredicate::sge:
      iotaSide = cmp.getRhs(), boundSide = cmp.getLhs(), extra = 1;
      break;
    default:
      return nullptr;
    }
    auto off = matchIotaPlusOffset(iotaSide, depth + 1);
    auto bound = off ? matchSplat(boundSide, depth + 1) : std::nullopt;
    if (!bound)
      return nullptr;
    auto len = makeLen(TailLen::Compare);
    len->bound = *bound;
    len->off = *off;
    len->cst = extra;
    len->intTy = getElementTypeOrSelf(iotaSide.getType());
    return len;
  }
  return nullptr;
}

// Under -debug-only=tail-mask-to-evl, each call is dumped as a block nested
// by depth, with the case that matched, the calls it makes and its result:
//   analyzeTailLength depth=0 {
//     is AndIOp
//     analyzeTailLength depth=1 {
//       ...
//     }
//     Min
//       ...
//   }
TailLenPtr analyzeTailLength(Value mask, int depth) {
  unsigned indent = 2 * depth;
  LLVM_DEBUG(llvm::dbgs().indent(indent)
             << "analyzeTailLength depth=" << depth << " {\n");
  TailLenPtr res = analyzeTailLengthImpl(mask, depth);
  LLVM_DEBUG({
    if (res)
      printTailLen(res, llvm::dbgs(), indent + 2);
    else
      llvm::dbgs().indent(indent + 2) << "failed\n";
    llvm::dbgs().indent(indent) << "}\n";
  });
  return res;
}

// Materialize a Scalar as a value of type `ty` (extracting it from a
// single-element vector if needed).
Value buildScalar(OpBuilder &b, Location loc, const Scalar &s, Type ty) {
  auto cst = [&](int64_t c) -> Value {
    return arith::ConstantOp::create(b, loc, ty, b.getIntegerAttr(ty, c));
  };
  if (!s.value)
    return cst(s.cst);
  Value v = s.value;
  if (auto vecTy = dyn_cast<VectorType>(v.getType()))
    v = vector::ExtractOp::create(b, loc, v,
                                  SmallVector<int64_t>(vecTy.getRank(), 0));
  if (v.getType() != ty) {
    if (isa<IndexType>(ty) || isa<IndexType>(v.getType()))
      v = arith::IndexCastOp::create(b, loc, ty, v);
    else
      v = arith::ExtSIOp::create(b, loc, ty, v);
  }
  return s.cst ? arith::AddIOp::create(b, loc, v, cst(s.cst)).getResult() : v;
}

// Build the index-typed tail length.
Value buildTailLength(OpBuilder &b, Location loc, const TailLen &len,
                      int depth = 0);

// The body of buildTailLength; buildTailLength wraps it with the debug dump.
Value buildTailLengthImpl(OpBuilder &b, Location loc, const TailLen &len,
                          int depth) {
  // Under -debug-only=tail-mask-to-evl: which node is built, inside the
  // buildTailLength block of this depth.
  auto note = [&](StringRef what) {
    (void)what;
    LLVM_DEBUG(llvm::dbgs().indent(2 * depth + 2) << what << "\n");
  };
  auto indexCst = [&](int64_t c) -> Value {
    return arith::ConstantIndexOp::create(b, loc, c);
  };
  switch (len.kind) {
  case TailLen::Const:
    note("is Const: arith.constant");
    return indexCst(len.cst);
  case TailLen::Index:
    // analyzeScalarLength looks through index_cast / extsi (both
    // sign-preserving), so the value may be an integer of another width.
    if (!isa<IndexType>(len.value.getType())) {
      note("is Index: arith.index_cast to index");
      return arith::IndexCastOp::create(b, loc, b.getIndexType(), len.value);
    }
    note("is Index: reuse the value");
    return len.value;
  case TailLen::Positive: {
    note("is Positive: v > 0 ? n : 0");
    Value v = buildTailLength(b, loc, *len.lhs, depth + 1);
    Value zero = indexCst(0);
    Value on = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, v, zero);
    return arith::SelectOp::create(b, loc, on, indexCst(len.n), zero);
  }
  case TailLen::AllOrNone:
    note("is AllOrNone: cond ? n : 0");
    return arith::SelectOp::create(b, loc,
                                   buildScalar(b, loc, len.cond, b.getI1Type()),
                                   indexCst(len.n), indexCst(0));
  case TailLen::Min:
  case TailLen::Max: {
    note(len.kind == TailLen::Min ? "is Min: arith.minsi"
                                  : "is Max: arith.maxsi");
    Value lhs = buildTailLength(b, loc, *len.lhs, depth + 1);
    Value rhs = buildTailLength(b, loc, *len.rhs, depth + 1);
    if (len.kind == TailLen::Min)
      return arith::MinSIOp::create(b, loc, lhs, rhs);
    return arith::MaxSIOp::create(b, loc, lhs, rhs);
  }
  case TailLen::Compare: {
    note("is Compare: bound - off + extra");
    // Operands are sign-extended to i64 first so bound - off (+1) cannot
    // wrap (index is 64-bit on the targets this runs on).
    Type wideTy = isa<IndexType>(len.intTy) ? len.intTy : b.getI64Type();
    Value bound = buildScalar(b, loc, len.bound, wideTy);
    Value off = buildScalar(b, loc, len.off, wideTy);
    Value diff = arith::SubIOp::create(b, loc, bound, off);
    if (len.cst)
      diff = arith::AddIOp::create(
          b, loc, diff,
          arith::ConstantOp::create(b, loc, wideTy,
                                    b.getIntegerAttr(wideTy, len.cst)));
    if (isa<IndexType>(wideTy))
      return diff;
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), diff);
  }
  }
  llvm_unreachable("unknown TailLen kind");
}

// Under -debug-only=tail-mask-to-evl, each call is dumped as a block nested
// by depth, like analyzeTailLength, ending with the value built for the node
// (its defining op, or the reused value):
//   buildTailLength depth=0 {
//     is Min: arith.minsi
//     buildTailLength depth=1 {
//       ...
//     }
//     built: %3 = arith.minsi %1, %2 : index
//   }
Value buildTailLength(OpBuilder &b, Location loc, const TailLen &len,
                      int depth) {
  unsigned indent = 2 * depth;
  LLVM_DEBUG(llvm::dbgs().indent(indent)
             << "buildTailLength depth=" << depth << " {\n");
  Value res = buildTailLengthImpl(b, loc, len, depth);
  LLVM_DEBUG({
    llvm::dbgs().indent(indent + 2) << "built: ";
    if (Operation *def = res.getDefiningOp())
      def->print(llvm::dbgs(), debugPrintingFlags());
    else
      res.printAsOperand(llvm::dbgs(), debugPrintingFlags());
    llvm::dbgs() << "\n";
    llvm::dbgs().indent(indent) << "}\n";
  });
  return res;
}

bool isAlreadyEVLMask(Value mask) {
  return matchPattern(mask, m_Constant()) ||
         mask.getDefiningOp<vector::CreateMaskOp>() ||
         mask.getDefiningOp<vector::ConstantMaskOp>();
}

bool sameScalar(const Scalar &a, const Scalar &b) {
  return a.value == b.value && a.cst == b.cst;
}

// Structural equality of two tail sizes.
bool sameLen(const TailLen &a, const TailLen &b) {
  if (a.kind != b.kind)
    return false;
  switch (a.kind) {
  case TailLen::Const:
    return a.cst == b.cst;
  case TailLen::Index:
    return a.value == b.value;
  case TailLen::Positive:
    return a.n == b.n && sameLen(*a.lhs, *b.lhs);
  case TailLen::AllOrNone:
    return a.n == b.n && sameScalar(a.cond, b.cond);
  case TailLen::Min:
  case TailLen::Max:
    return (sameLen(*a.lhs, *b.lhs) && sameLen(*a.rhs, *b.rhs)) ||
           (sameLen(*a.lhs, *b.rhs) && sameLen(*a.rhs, *b.lhs));
  case TailLen::Compare:
    return a.cst == b.cst && a.intTy == b.intTy &&
           sameScalar(a.bound, b.bound) && sameScalar(a.off, b.off);
  }
  return false;
}

// Whether size `a` <= size `b` for every run-time value (so a mask of size
// `a` never enables a lane that a mask of size `b` disables).
bool provablyLE(const TailLen &a, const TailLen &b) {
  if (sameLen(a, b))
    return true;
  if (a.kind == TailLen::Const && a.cst <= 0)
    return true;
  if (a.kind == TailLen::Const && b.kind == TailLen::Const)
    return a.cst <= b.cst;
  if ((a.kind == TailLen::AllOrNone || a.kind == TailLen::Positive) &&
      b.kind == TailLen::Const)
    return a.n <= b.cst;
  // min(x, y) <= b if either x <= b or y <= b.
  if (a.kind == TailLen::Min &&
      (provablyLE(*a.lhs, b) || provablyLE(*a.rhs, b)))
    return true;
  // a <= max(x, y) if a <= x or a <= y.
  if (b.kind == TailLen::Max &&
      (provablyLE(a, *b.lhs) || provablyLE(a, *b.rhs)))
    return true;
  // a <= min(x, y) if a <= x and a <= y.
  if (b.kind == TailLen::Min && provablyLE(a, *b.lhs) &&
      provablyLE(a, *b.rhs))
    return true;
  return false;
}

// `mask` (a use's own mask) never enables a lane >= `len`.
bool maskWithin(Value mask, const TailLen &len) {
  TailLenPtr useLen = analyzeTailLength(mask, 0);
  return useLen && provablyLE(*useLen, len);
}

// Lane-wise ops: result lane i depends only on lane i of each operand, and
// lanes keep their positions (all shapes are effectively 1-D of one length).
bool isLaneWise(Operation *op, int64_t n) {
  if (op->getNumResults() != 1 ||
      getEffective1DLength(op->getResult(0).getType()) != n)
    return false;
  if (isa<vector::ShapeCastOp>(op))
    return true;
  if (!op->hasTrait<OpTrait::Elementwise>())
    return false;
  return llvm::all_of(op->getOperandTypes(), [&](Type t) {
    return getEffective1DLength(t) == n;
  });
}

// Whether no lane >= `len` of `v` can affect the program: every path through
// lane-wise ops ends in an arith.select that picks `v` only where its
// condition (size <= len) is on, in a vector.maskedstore of `v` whose mask
// has size <= len, or in a masked vector.reduction whose mask has size <= len.
bool tailLanesUnused(Value v, const TailLen &len, int64_t n,
                     llvm::SmallPtrSetImpl<Value> &visited, int depth = 0) {
  if (depth > 64)
    return false;
  if (!visited.insert(v).second)
    return true;
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();
    if (auto store = dyn_cast<vector::MaskedStoreOp>(user)) {
      if (use.get() == store.getValueToStore() &&
          use.getOperandNumber() != 0 && maskWithin(store.getMask(), len))
        continue;
      return false;
    }
    // A masked reduction (vector.mask %m { vector.reduction %v }, from
    // ConvertReductionOp with TRITON_VSETVL_REDUCE) only reads lanes of %v
    // that %m enables.
    if (auto red = dyn_cast<vector::ReductionOp>(user)) {
      auto maskOp = dyn_cast<vector::MaskOp>(red->getParentOp());
      if (maskOp && use.get() == red.getVector() &&
          maskWithin(maskOp.getMask(), len))
        continue;
      return false;
    }
    if (auto sel = dyn_cast<arith::SelectOp>(user)) {
      if (use.getOperandNumber() == 1 &&
          isa<VectorType>(sel.getCondition().getType()) &&
          maskWithin(sel.getCondition(), len))
        continue;
    }
    if (!isLaneWise(user, n) ||
        !tailLanesUnused(user->getResult(0), len, n, visited, depth + 1))
      return false;
  }
  return true;
}

// Tail length of a masked access, or null when its mask is not a tail mask.
template <typename OpTy> TailLenPtr getTailLen(OpTy op) {
  auto maskTy = cast<VectorType>(op.getMask().getType());
  if (maskTy.getRank() != 1 || maskTy.isScalable())
    return nullptr;
  return analyzeTailLength(op.getMask(), 0);
}

template <typename OpTy> void replaceMask(OpTy op, const TailLen &len) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  Value evl = buildTailLength(b, loc, len);
  Value newMask =
      vector::CreateMaskOp::create(b, loc, op.getMask().getType(), evl);
  op.getMaskMutable().assign(newMask);
  LLVM_DEBUG({
    llvm::dbgs() << "new mask: ";
    newMask.print(llvm::dbgs(), debugPrintingFlags());
    llvm::dbgs() << "\n  for: ";
    op->print(llvm::dbgs(), debugPrintingFlags());
    llvm::dbgs() << "\n";
  });
}

struct TailMaskToEVL
    : public triton::cpu::impl::TailMaskToEVLBase<TailMaskToEVL> {
  TailMaskToEVL() = default;

  void runOnOperation() override {
    // Analyze everything on the original IR first: whether a load's tail
    // lanes are used is decided from the masks of its users' selects and
    // stores, which the rewrite below replaces.
    SmallVector<std::pair<vector::MaskedLoadOp, TailLenPtr>> loads;
    SmallVector<std::pair<vector::MaskedStoreOp, TailLenPtr>> stores;
    SmallVector<vector::MaskedLoadOp> poisonTail;
    getOperation()->walk([&](Operation *op) {
      if (auto load = dyn_cast<vector::MaskedLoadOp>(op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "masked load: ";
          load->print(llvm::dbgs(), debugPrintingFlags());
          llvm::dbgs() << "\n";
        });
        if (TailLenPtr len = getTailLen(load)) {
          loads.emplace_back(load, len);
          llvm::SmallPtrSet<Value, 16> visited;
          if (!load.getPassThru().getDefiningOp<ub::PoisonOp>() &&
              tailLanesUnused(load.getResult(), *len,
                              load.getVectorType().getNumElements(),
                              visited)) {
            LLVM_DEBUG(llvm::dbgs() << "  tail lanes unused: poison "
                                       "pass-through\n");
            poisonTail.push_back(load);
          }
        }
      } else if (auto store = dyn_cast<vector::MaskedStoreOp>(op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "masked store: ";
          store->print(llvm::dbgs(), debugPrintingFlags());
          llvm::dbgs() << "\n";
        });
        if (TailLenPtr len = getTailLen(store))
          stores.emplace_back(store, len);
      }
    });

    // Masks already in create_mask / constant form are kept as they are.
    for (auto &[load, len] : loads)
      if (!isAlreadyEVLMask(load.getMask()))
        replaceMask(load, *len);
    for (auto &[store, len] : stores)
      if (!isAlreadyEVLMask(store.getMask()))
        replaceMask(store, *len);
    // Lanes past the tail are never observed, so they need not hold the
    // pass-through value; poison lets the lowering skip the merge.
    for (vector::MaskedLoadOp load : poisonTail) {
      OpBuilder b(load);
      Value poison =
          ub::PoisonOp::create(b, load.getLoc(), load.getPassThru().getType());
      load.getPassThruMutable().assign(poison);
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createTailMaskToEVL() {
  return std::make_unique<TailMaskToEVL>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
