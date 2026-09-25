#include "triton/Tools/Sys/GetEnv.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"

#include <sstream>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>

#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

namespace py = nanobind;

namespace {

// The CPU backend cross-compiles for riscv64 by default, whatever the build
// host is. TRITON_CPU_TARGET=native selects the build host instead; it is only
// used to run a model on the host (e.g. scripts/build_qwen_engine.py capture).
bool isNativeTarget() {
  return mlir::triton::tools::getStrEnv("TRITON_CPU_TARGET") == "native";
}

const char *const kRISCV64Triple = "riscv64-unknown-linux-gnu";

std::string getHostTargetTriple() {
  if (isNativeTarget()) {
    std::string triple = llvm::sys::getDefaultTargetTriple();
    if (triple.empty())
      triple = llvm::sys::getProcessTriple();
    return triple;
  }
  return kRISCV64Triple;
}

// The CPU backend always targets riscv64: a fixed RV64GCV baseline with
// fixed-length vectors, since LLVM cannot reliably detect the vector extension
// at runtime.
const char *const kRISCV64CPU = "generic-rv64";
// +zvfbfmin only makes bf16 a legal vector memory type (plain vle16/vse16);
// without it LLVM scalarizes every bf16 vector load/store. The compiler
// decomposes bf16 conversions to integer ops, so no bf16 instructions are emitted.
const char *const kRISCV64Features = "+m,+f,+d,+v,+zvfbfmin";

std::string getHostCPUName() {
  return isNativeTarget() ? llvm::sys::getHostCPUName().str() : kRISCV64CPU;
}

std::string getHostFeatureString() {
  return isNativeTarget() ? "" : kRISCV64Features;
}

// LMUL is taken from TRITON_RISCV_LMUL (1, 2, 4 or 8; default 8).
std::string getRISCVLMUL() {
  std::string lmul = mlir::triton::tools::getStrEnv("TRITON_RISCV_LMUL");
  if (lmul.empty())
    return "8";
  if (lmul != "1" && lmul != "2" && lmul != "4" && lmul != "8")
    throw std::runtime_error("TRITON_RISCV_LMUL must be one of 1, 2, 4, 8; got '" +
                             lmul + "'");
  return lmul;
}

void setRISCV64CodegenOptions() {
  const std::string regLMUL = "-riscv-v-register-bit-width-lmul=" + getRISCVLMUL();
  // Also cap the LMUL of the fixed-length vectors Triton emits; the flag above
  // only affects the autovectorizers.
  const std::string fixedLMUL =
      "-riscv-v-fixed-length-vector-lmul-max=" + getRISCVLMUL();
  const char *const args[] = {
      "triton-cpu",
      "--tail-folding-policy=prefer-fold-tail",
      "-riscv-v-vector-bits-min=256",
      regLMUL.c_str(),
      fixedLMUL.c_str(),
      "-force-tail-folding-style=data-with-evl",
  };
  std::vector<std::string> extra;
  {
    // TRITON_RISCV_LLVM_ARGS: extra whitespace-separated LLVM options, e.g.
    // "-custom-sink -custom-remat -custom-a".
    std::istringstream iss(mlir::triton::tools::getStrEnv("TRITON_RISCV_LLVM_ARGS"));
    for (std::string tok; iss >> tok;)
      extra.push_back(tok);
  }
  std::vector<const char *> argv(std::begin(args), std::end(args));
  for (const auto &e : extra)
    argv.push_back(e.c_str());
  llvm::cl::ParseCommandLineOptions(argv.size(), argv.data());
}

void initializeHostTarget() {
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    if (isNativeTarget()) {
      if (llvm::InitializeNativeTarget())
        throw std::runtime_error("LLVM native target is not available");
      if (llvm::InitializeNativeTargetAsmPrinter())
        throw std::runtime_error("LLVM native target assembly printer is not "
                                 "available");
      return;
    }
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVAsmPrinter();
  });

  // LLVM's global thread pool is not fork-safe. Triton kernels are small, so
  // disabling LLVM's internal parallelism also avoids unnecessary overhead.
  llvm::parallel::strategy = llvm::hardware_concurrency(1);

  if (!isNativeTarget()) {
    static std::once_flag riscvFlag;
    std::call_once(riscvFlag, setRISCV64CodegenOptions);
  }
}

void setLLVMBooleanOption(const std::string &name, bool value) {
  auto options = llvm::cl::getRegisteredOptions();
  auto it = options.find(name);
  if (it == options.end())
    return;
  it->second->addOccurrence(1, name, value ? "true" : "false");
}

std::unique_ptr<llvm::TargetMachine>
createHostTargetMachine(llvm::Module &module, bool enableFpFusion,
                        bool enableFastMath) {
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  if (!target)
    throw std::runtime_error("target lookup error: " + error);

  llvm::TargetOptions options;
  if (enableFpFusion)
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  if (enableFastMath)
    options.NoTrappingFPMath = true;
  options.TrapUnreachable = true;
  options.MCOptions.AsmVerbose = true;
  options.MCOptions.PreserveAsmComments = true;

  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  return std::unique_ptr<llvm::TargetMachine>{target->createTargetMachine(
      module.getTargetTriple(), getHostCPUName(), getHostFeatureString(),
      options,
      llvm::Reloc::PIC_, std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
}

std::string translateHostLLVMIRToASM(llvm::Module &module, bool enableFpFusion,
                                     bool enableFastMath) {
  if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP"))
    setLLVMBooleanOption("print-after-all", true);

  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<llvm::StringRef, 3> flags;
      llvm::StringRef(flagList).split(flags, ',');
      for (llvm::StringRef flag : flags)
        setLLVMBooleanOption(flag.str(), true);
    }
  }

  for (llvm::Function &function : module.functions())
    if (!function.hasFnAttribute(llvm::Attribute::NoInline))
      function.addFnAttr(llvm::Attribute::AlwaysInline);

  llvm::legacy::PassManager modulePasses;
  modulePasses.add(llvm::createAlwaysInlinerLegacyPass());
  modulePasses.add(llvm::createVerifierPass());

  const bool enableTiming =
      mlir::triton::tools::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enableTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  modulePasses.run(module);

  llvm::SmallString<0> timePassesStr;
  llvm::raw_svector_ostream reportStream(timePassesStr);
  if (enableTiming) {
    llvm::reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }

  module.setTargetTriple(llvm::Triple(getHostTargetTriple()));
  auto machine =
      createHostTargetMachine(module, enableFpFusion, enableFastMath);
  module.setDataLayout(machine->createDataLayout());

  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream bufferedStream(stream);
    llvm::legacy::PassManager codegenPasses;
    machine->addPassesToEmitFile(codegenPasses, bufferedStream, nullptr,
                                 llvm::CodeGenFileType::AssemblyFile);
    codegenPasses.run(module);

    if (enableTiming) {
      llvm::reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
  }
  return result;
}

void setHostTarget(llvm::Module &module) {
  initializeHostTarget();
  module.setTargetTriple(llvm::Triple(getHostTargetTriple()));

  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  if (!target)
    throw std::runtime_error("target lookup error: " + error);

  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module.getTargetTriple(), getHostCPUName(), getHostFeatureString(), {},
      llvm::Reloc::PIC_)};
  module.setDataLayout(machine->createDataLayout());

  if (isNativeTarget())
    return;

  // Let IR-level passes (e.g. the vectorizer) see the target CPU/features.
  for (llvm::Function &function : module.functions()) {
    function.addFnAttr("target-cpu", kRISCV64CPU);
    function.addFnAttr("target-features", kRISCV64Features);
  }
}

std::set<std::string> getCPUFeatures() {
  if (!isNativeTarget())
    return {"+m", "+f", "+d", "+v", "+zvfbfmin"};

  auto features = llvm::sys::getHostCPUFeatures();

  std::set<std::string> result;
  for (const auto &feature : features)
    if (feature.second)
      result.insert(feature.first().str());

  // NEON is mandatory on AArch64. Use it as a safe fallback if LLVM feature
  // detection unexpectedly returns an empty set.
  if (result.empty()) {
    std::string triple = llvm::sys::getProcessTriple();
    std::size_t separator = triple.find('-');
    if (separator != std::string::npos) {
      std::string arch = triple.substr(0, separator);
      if (arch == "aarch64" || arch == "arm64")
        result.insert("neon");
    }
  }

  return result;
}

} // namespace

void init_triton_cpu_llvm(py::module_ &m) {
  m.def("get_cpu_triple", []() { return getHostTargetTriple(); });
  m.def("get_cpu_name", []() { return getHostCPUName(); });
  m.def("get_cpu_features", &getCPUFeatures);
  m.def("set_host_target",
        [](llvm::Module *module) { setHostTarget(*module); });
  m.def("translate_to_asm",
        [](std::string llvmIR, bool enableFpFusion,
           bool enableFastMath) -> py::object {
          std::string result;
          {
            py::gil_scoped_release release;
            initializeHostTarget();

            llvm::LLVMContext context;
            std::unique_ptr<llvm::MemoryBuffer> buffer =
                llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
            llvm::SMDiagnostic error;
            std::unique_ptr<llvm::Module> module =
                llvm::parseIR(buffer->getMemBufferRef(), error, context);
            if (!module) {
              llvm::report_fatal_error(
                  "failed to parse IR: " + error.getMessage() +
                  "lineno: " + std::to_string(error.getLineNo()));
            }
            result = translateHostLLVMIRToASM(*module, enableFpFusion,
                                              enableFastMath);
          }
          return py::str(result.c_str(), result.size());
        });
}
