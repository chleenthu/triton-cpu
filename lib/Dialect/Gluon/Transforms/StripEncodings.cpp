#include "triton/Dialect/Gluon/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace triton;
namespace ttg = triton::gpu;
namespace gluon = mlir::triton::gluon;

namespace mlir::triton::gluon {
#define GEN_PASS_DEF_GLUONSTRIPENCODINGSPASS
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"
} // namespace mlir::triton::gluon

namespace {

RankedTensorType stripEncoding(RankedTensorType type) {
  return RankedTensorType::get(type.getShape(), type.getElementType());
}

struct StripEncodings
    : public gluon::impl::GluonStripEncodingsPassBase<StripEncodings> {
  void runOnOperation() override;
};

} // namespace

void StripEncodings::runOnOperation() {
  ModuleOp mod = getOperation();

  // Dense constants carry their (encoded) tensor type inside the attribute.
  mod.walk([](arith::ConstantOp op) {
    auto value = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!value)
      return;
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (type && type.getEncoding())
      op.setValueAttr(value.reshape(stripEncoding(type)));
  });

  // Strip the encoding from every tensor type: op results, block arguments,
  // function signatures and type attributes.
  AttrTypeReplacer replacer;
  replacer.addReplacement([](RankedTensorType type) -> std::optional<Type> {
    if (!type.getEncoding())
      return std::nullopt;
    return stripEncoding(type);
  });
  replacer.recursivelyReplaceElementsIn(mod, /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

  // Without encodings, layout conversions are identities.
  mod.walk([](ttg::ConvertLayoutOp op) {
    op.replaceAllUsesWith(op.getSrc());
    op.erase();
  });

  // Anything left outside these dialects has no TTIR counterpart.
  static const llvm::StringSet<> allowedDialects = {
      "builtin", "tt", "arith", "math", "scf", "cf", "ub"};
  WalkResult result = mod.walk([](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (dialect && allowedDialects.contains(dialect->getNamespace()))
      return WalkResult::advance();
    op->emitError() << "'" << op->getName()
                    << "' is not supported when lowering Gluon to TTIR";
    return WalkResult::interrupt();
  });
  if (result.wasInterrupted())
    return signalPassFailure();

  // Drop the ttg.* module attributes (num-warps, threads-per-warp, target...).
  SmallVector<StringAttr> ttgAttrs;
  for (NamedAttribute attr : mod->getAttrs())
    if (attr.getName().strref().starts_with("ttg."))
      ttgAttrs.push_back(attr.getName());
  for (StringAttr name : ttgAttrs)
    mod->removeAttr(name);
}
