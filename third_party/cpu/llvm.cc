#include "triton/Tools/LLVMOptions.h"
#include "triton/Tools/Sys/GetEnv.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/InstIterator.h"
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
#include <nanobind/stl/vector.h>

#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
#include <sys/sysctl.h>
#endif

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
// No zvfbfmin: the target board (SpacemiT X60) does not implement it, so any
// vfwcvtbf16/vfncvtbf16 would raise SIGILL. bf16 therefore never reaches LLVM
// as a vector type on riscv64: the compiler does bf16 memory ops on i16
// (promote_bf16_to_fp32) and bf16<->f32 conversions with integer ops
// (decompose_bf16_conv).
const char *const kRISCV64Features = "+m,+f,+d,+v";

std::string getHostCPUName() {
  return isNativeTarget() ? llvm::sys::getHostCPUName().str() : kRISCV64CPU;
}

// LMUL is taken from TRITON_RISCV_LMUL (1, 2, 4 or 8; default 1). LMUL=1 was
// ~19% faster than 8 on the vLLM unified-attention 2D kernel on the K1 board.
std::string getRISCVLMUL() {
  std::string lmul = mlir::triton::tools::getStrEnv("TRITON_RISCV_LMUL");
  if (lmul.empty())
    return "1";
  if (lmul != "1" && lmul != "2" && lmul != "4" && lmul != "8")
    throw std::runtime_error("TRITON_RISCV_LMUL must be one of 1, 2, 4, 8; got '" +
                             lmul + "'");
  return lmul;
}

// -debug / -debug-only=<types> in TRITON_RISCV_LLVM_ARGS. These are not
// passed to cl::ParseCommandLineOptions: that would set llvm::DebugFlag for
// the whole process (MLIR passes and the LLVM O3 pipeline included), and
// TRITON_LLVM_DEBUG_ONLY replaces the global debug-type list on every MLIR
// pass-manager run. Instead they are applied around each codegen run only
// (applyCodegenDebugOptions), mirroring the scoping of TRITON_LLVM_DEBUG_ONLY
// in python/src/ir.cc.
bool isDebugOption(llvm::StringRef tok) {
  tok = tok.ltrim('-');
  return tok == "debug" || tok.starts_with("debug-only=");
}

// Called from translateHostLLVMIRToASM; the caller restores llvm::DebugFlag.
void applyCodegenDebugOptions() {
  if (isNativeTarget())
    return;
  std::istringstream iss(
      mlir::triton::tools::getStrEnv("TRITON_RISCV_LLVM_ARGS"));
  for (std::string tok; iss >> tok;) {
    if (!isDebugOption(tok))
      continue;
    llvm::DebugFlag = true;
    llvm::StringRef list = llvm::StringRef(tok).ltrim('-');
    if (!list.consume_front("debug-only="))
      continue; // plain -debug: all debug types
    llvm::SmallVector<llvm::StringRef> parts;
    list.split(parts, ',', -1, /*KeepEmpty=*/false);
    std::vector<std::string> storage(parts.begin(), parts.end());
    std::vector<const char *> types;
    for (const auto &t : storage)
      types.push_back(t.c_str());
    llvm::setCurrentDebugTypes(types.data(), types.size());
  }
}

using mlir::triton::tools::ScopedLLVMOptions;

// The LLVM options the riscv64 target needs, for both the O3 pipeline
// (make_llir, passed as optimize_module flags) and codegen (translate_to_asm).
// They must be applied per run through ScopedLLVMOptions: every
// ScopedLLVMOptions scope (optimize_module, attach_datalayout, ...) calls
// cl::ResetAllOptionOccurrences() on exit, so options parsed once per process
// with cl::ParseCommandLineOptions were silently reset to LLVM defaults after
// the first O3 run (TRITON_RISCV_LMUL, -riscv-v-vector-bits-min, tail folding
// all ignored by codegen).
std::vector<ScopedLLVMOptions::Setting> getTargetLLVMOptionSettings() {
  if (isNativeTarget())
    return {};
  const std::string lmul = getRISCVLMUL();
  std::vector<ScopedLLVMOptions::Setting> settings = {
      {"tail-folding-policy", "prefer-fold-tail"},
      {"riscv-v-vector-bits-min", "256"},
      {"riscv-v-register-bit-width-lmul", lmul},
      // Also cap the LMUL of the fixed-length vectors Triton emits; the flag
      // above only affects the autovectorizers.
      {"riscv-v-fixed-length-vector-lmul-max", lmul},
      {"force-tail-folding-style", "data-with-evl"},
  };
  // TRITON_RISCV_LLVM_ARGS: extra whitespace-separated LLVM options, e.g.
  // "-riscv-foo=1 -enable-bar". -debug/-debug-only are handled separately
  // (applyCodegenDebugOptions), for codegen only.
  std::istringstream iss(
      mlir::triton::tools::getStrEnv("TRITON_RISCV_LLVM_ARGS"));
  for (std::string tok; iss >> tok;) {
    if (isDebugOption(tok))
      continue;
    auto [name, value] = llvm::StringRef(tok).ltrim('-').split('=');
    settings.emplace_back(name.str(),
                          tok.find('=') == std::string::npos ? "true"
                                                             : value.str());
  }
  return settings;
}

// optimize_module takes its option overrides as "name=value" flags.
std::vector<std::string> getTargetLLVMFlags() {
  std::vector<std::string> flags;
  for (const auto &[name, value] : getTargetLLVMOptionSettings())
    flags.push_back(name + "=" + value);
  return flags;
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
}

// Return true if the CPU triple is AArch64.  Used by the NEON fallback below.
static bool isAArch64() {
  std::string triple = llvm::sys::getProcessTriple();
  std::size_t separator = triple.find('-');
  if (separator == std::string::npos)
    return false;
  std::string arch = triple.substr(0, separator);
  return arch == "aarch64" || arch == "arm64";
}

namespace arm {

bool osSupportsBF16() {
#if defined(__linux__) && defined(__aarch64__)
  return (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  int value = 0;
  size_t size = sizeof(value);
  return sysctlbyname("hw.optional.arm.FEAT_BF16", &value, &size, nullptr, 0) ==
             0 &&
         size == sizeof(value) && value == 1;
#else
  return false;
#endif
}

} // namespace arm

// Share Arm feature fallbacks between the Python backend and LLVM target
// machines. LLVM may omit BF16 even when the operating system reports it.
std::set<std::string> getCPUFeatures() {
  if (!isNativeTarget())
    return {"+m", "+f", "+d", "+v", "+zvfbfmin"};

  auto features = llvm::sys::getHostCPUFeatures();

  std::set<std::string> result;
  for (const auto &feature : features)
    if (feature.second)
      result.insert(feature.first().str());

  if (isAArch64()) {
    // Apply this before BF16 so an empty LLVM result still gets mandatory NEON.
    if (result.empty())
      result.insert("neon");
    // Only query the OS when LLVM has not already reported BF16.
    if (!result.count("bf16") && arm::osSupportsBF16())
      result.insert("bf16");
  }

  return result;
}

// Build a comma-separated feature string (e.g. "+sve,+sve2,+neon") suitable
// for passing to TargetMachine::createTargetMachine, derived from
// getCPUFeatures() so both share the same NEON fallback logic.
static std::string getCPUFeaturesString() {
  // riscv64 cross target: the fixed RV64GCV baseline (no zvfbfmin).
  if (!isNativeTarget())
    return kRISCV64Features;
  std::string result;
  for (const auto &feature : getCPUFeatures()) {
    if (!result.empty())
      result += ",";
    result += "+";
    result += feature;
  }
  return result;
}

void enableFPContraction(llvm::Module &module) {
  for (llvm::Function &function : module)
    for (llvm::Instruction &instruction : llvm::instructions(function))
      if (instruction.getOpcode() == llvm::Instruction::FAdd ||
          instruction.getOpcode() == llvm::Instruction::FSub ||
          instruction.getOpcode() == llvm::Instruction::FMul)
        instruction.setHasAllowContract(true);
}

std::unique_ptr<llvm::TargetMachine>
createHostTargetMachine(llvm::Module &module, const std::string &cpuFeatures,
                        bool enableFpFusion) {
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  if (!target)
    throw std::runtime_error("target lookup error: " + error);

  llvm::TargetOptions options;
  if (enableFpFusion)
    enableFPContraction(module);
  options.TrapUnreachable = true;
  options.MCOptions.AsmVerbose = true;
  options.MCOptions.PreserveAsmComments = true;

  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  return std::unique_ptr<llvm::TargetMachine>{target->createTargetMachine(
      module.getTargetTriple(), getHostCPUName(), cpuFeatures,
      options, llvm::Reloc::PIC_, std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
}

// Set the vscale_range function attribute on all functions in the module when
// SVE is available.  This tells LLVM that the function may use scalable
// vectors and that the runtime vscale value is within the given range.
// For SVE, the vector length is vscale * 128 bits, and on common AArch64
// implementations vscale is 1 (128-bit) or 2 (256-bit).  We use a conservative
// range of [1, 16] to cover all SVE implementations (up to 2048-bit).
void setVscaleRangeForSVE(llvm::Module &module, llvm::StringRef cpuFeatures) {
  if (cpuFeatures.find("+sve") == std::string::npos)
    return;

  for (llvm::Function &function : module.functions()) {
    if (function.isIntrinsic())
      continue;
    function.addFnAttr(llvm::Attribute::getWithVScaleRangeArgs(
        function.getContext(), /*Min=*/1, /*Max=*/16));
  }
}

std::string translateHostLLVMIRToASM(llvm::Module &module,
                                     bool enableFpFusion) {
  // All LLVM option overrides for this codegen run, held for its whole
  // duration (see getTargetLLVMOptionSettings).
  std::vector<ScopedLLVMOptions::Setting> settings =
      getTargetLLVMOptionSettings();
  if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP"))
    settings.emplace_back("print-after-all", "true");
  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
      llvm::SmallVector<llvm::StringRef, 3> flags;
      llvm::StringRef(flagList).split(flags, ',');
      for (llvm::StringRef flag : flags)
        settings.emplace_back(flag.str(), "true");
    }
  }
  ScopedLLVMOptions optionScope(settings);

  // Codegen-only -debug/-debug-only from TRITON_RISCV_LLVM_ARGS. Declared
  // after optionScope so the flag is restored before the scope resets the
  // option registry.
  const bool savedDebugFlag = llvm::DebugFlag;
  llvm::scope_exit restoreDebugFlag(
      [savedDebugFlag] { llvm::DebugFlag = savedDebugFlag; });
  applyCodegenDebugOptions();

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
  // Propagate host CPU features to the target machine and set vscale_range
  // attribute on functions when SVE is available so that scalable vector
  // code generation is enabled.
  const std::string cpuFeatures = getCPUFeaturesString();
  setVscaleRangeForSVE(module, cpuFeatures);
  auto machine = createHostTargetMachine(module, cpuFeatures, enableFpFusion);
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

  // Propagate host CPU features to the target machine and set vscale_range
  // attribute on functions when SVE is available so that scalable vector
  // code generation is enabled.
  const std::string cpuFeatures = getCPUFeaturesString();
  setVscaleRangeForSVE(module, cpuFeatures);

  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  if (!target)
    throw std::runtime_error("target lookup error: " + error);

  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module.getTargetTriple(), getHostCPUName(), cpuFeatures, {},
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

} // namespace

void init_triton_cpu_llvm(py::module_ &m) {
  m.def("get_cpu_triple", []() { return getHostTargetTriple(); });
  m.def("get_cpu_name", []() { return getHostCPUName(); });
  m.def("get_cpu_features", &getCPUFeatures);
  // LLVM option overrides ("name=value") the target needs in optimize_module.
  m.def("get_llvm_flags", &getTargetLLVMFlags);
  m.def("set_host_target",
        [](llvm::Module *module) { setHostTarget(*module); });
  m.def("translate_to_asm",
        [](std::string llvmIR, bool enableFpFusion) -> py::object {
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
            result = translateHostLLVMIRToASM(*module, enableFpFusion);
          }
          return py::str(result.c_str(), result.size());
        });
}
