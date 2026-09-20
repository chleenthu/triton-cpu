from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import triton
from triton.compiler import ASTSource

from .driver import ty_to_cpp


@dataclass(frozen=True)
class Toolchain:
    cc: str
    sysroot: str | None
    gcc_toolchain: str | None
    march: str = "rv64gcv"
    mabi: str = "lp64d"

    @classmethod
    def from_env(cls) -> "Toolchain":
        cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("gcc")
        if not cc:
            raise RuntimeError("No C compiler found; set CC to a riscv64-capable clang/gcc")
        toolchain_dir = os.environ.get("TRITON_RISCV_TOOLCHAIN", os.path.expanduser("~/toolchain"))
        sysroot = os.environ.get("TRITON_RISCV_SYSROOT", os.path.join(toolchain_dir, "sysroot"))
        return cls(
            cc=cc,
            sysroot=sysroot if os.path.isdir(sysroot) else None,
            gcc_toolchain=toolchain_dir if os.path.isdir(toolchain_dir) else None,
            march=os.environ.get("TRITON_RISCV_MARCH", "rv64gcv"),
            mabi=os.environ.get("TRITON_RISCV_MABI", "lp64d"),
        )

    def compile_command(self, sources: Sequence[str], output: str, extra_args: Sequence[str] = ()) -> list[str]:
        cmd = [self.cc, *sources, "-O3", "-Wno-psabi", "--target=riscv64-unknown-linux-gnu", f"-march={self.march}",
               f"-mabi={self.mabi}", "-fuse-ld=lld"]
        if self.sysroot:
            cmd.append(f"--sysroot={self.sysroot}")
        if self.gcc_toolchain:
            cmd.append(f"--gcc-toolchain={self.gcc_toolchain}")
        cmd.extend(extra_args)
        cmd.extend(["-o", output])
        return cmd


def _infer_triton_type(value) -> str:
    is_buffer = not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")
    dtype_name = str(getattr(value, "dtype", "")).removeprefix("torch.")
    dtype_map = {
        "bool": "i1", "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
        "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
        "float16": "fp16", "bfloat16": "bf16", "float32": "fp32", "float64": "fp64",
    }
    scalar_type = dtype_map.get(dtype_name)

    if is_buffer:
        values = list(value)
        if scalar_type is None:
            if not values:
                raise ValueError(f"Cannot infer the type of empty buffer; pass signature=... explicitly")
            scalar_type = _infer_triton_type(values[0]).removeprefix("*")
        return f"*{scalar_type}"

    if scalar_type is not None:
        return scalar_type
    if isinstance(value, bool):
        return "i1"
    if isinstance(value, float):
        return "fp32"
    if isinstance(value, int):
        return "i32" if -(2**31) <= value < 2**31 else "i64"
    raise ValueError(f"Cannot infer a Triton type for {type(value).__name__}; pass signature=... explicitly")


_FLOAT_C_TYPES = ("float", "double", "_Float16", "__bf16")


def _c_literal(value, c_type: str) -> str:
    if c_type in _FLOAT_C_TYPES:
        value = float(value)
        # repr() of a non-finite value ("nan"/"inf") is not a valid C literal on its
        # own ("nan" + "f" suffix -> the bare identifier "nanf", not a float literal);
        # uninitialized output buffers can legitimately contain such bit patterns.
        if math.isnan(value):
            return "NAN"
        if math.isinf(value):
            return "INFINITY" if value > 0 else "-INFINITY"
        literal = repr(value)
        if c_type != "double":
            # A plain float literal narrows to _Float16/__bf16 in the initializer.
            literal += "f"
        return literal
    return str(int(value))


def make_standalone_source(kernel, arguments: dict, *, constexprs: dict | None = None,
                            signature: dict | None = None) -> ASTSource:
    constexprs = constexprs or {}
    overrides = signature or {}
    inferred = {}
    for name in kernel.arg_names:
        if name in constexprs:
            inferred[name] = "constexpr"
        elif name in overrides:
            inferred[name] = overrides[name]
        elif name in arguments:
            inferred[name] = _infer_triton_type(arguments[name])
        else:
            raise ValueError(f"Missing standalone argument or constexpr: {name}")
    return ASTSource(fn=kernel, signature=inferred, constexprs=constexprs)


def compile_kernel_to_so(kernel, arguments: dict, *, constexprs: dict | None = None,
                          signature: dict | None = None, num_warps: int = 1, num_stages: int = 0):
    from triton.backends.compiler import GPUTarget

    source = make_standalone_source(kernel, arguments, constexprs=constexprs, signature=signature)
    target = GPUTarget("cpu", 0, 0)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    compiled = triton.compile(source, target=target, options=options.__dict__)
    return compiled, compiled.asm[backend.binary_ext]


def generate_runner(kernel_name: str, signature: dict, arguments: dict, grid: Sequence[int],
                     expected: dict | None = None, atol: float = 1e-5) -> str:
    grid = tuple(int(v) for v in grid)
    if not 1 <= len(grid) <= 3 or any(v < 0 for v in grid):
        raise ValueError("grid must contain one to three non-negative dimensions")
    grid = grid + (1,) * (3 - len(grid))
    expected = expected or {}

    declarations: list[str] = []
    extern_types: list[str] = []
    call_args: list[str] = []
    checks: list[str] = []
    for index, (name, ty) in enumerate(signature.items()):
        if ty == "constexpr":
            continue
        if name not in arguments:
            raise ValueError(f"Missing standalone argument: {name}")

        if ty[0] == "*":
            values = list(arguments[name])
            elem_ty = ty[1:]
            # ty_to_cpp maps bf16/fp16 to "float" (4 bytes) for scalar/ABI purposes, but
            # the buffer the kernel actually indexes into is genuine 2-byte storage;
            # declaring it as a 4-byte array here would corrupt stride/pointer arithmetic.
            # Use clang's native half-precision types instead of bit-pattern plumbing
            # (reference: ~/triton-riscv-workspace/triton-riscv commit 3c28abf "add f16")
            # -- a plain decimal float literal narrows to _Float16/__bf16 in the array
            # initializer, so callers can keep passing ordinary float values.
            if elem_ty == "fp16":
                elem_type = "_Float16"
            elif elem_ty == "bf16":
                elem_type = "__bf16"
            else:
                elem_type = ty_to_cpp(elem_ty)
            storage_size = max(1, len(values))
            initializer = ", ".join(_c_literal(v, elem_type) for v in values) or "0"
            declarations.append(f"  {elem_type} arg_{index}[{storage_size}] = {{{initializer}}};")
            extern_types.append("void*")
            call_args.append(f"arg_{index}")

            if name in expected:
                expected_values = list(expected[name])
                if len(expected_values) != len(values):
                    raise ValueError(f"Expected length for {name} does not match its buffer")
                expected_init = ", ".join(_c_literal(v, elem_type) for v in expected_values) or "0"
                declarations.append(f"  const {elem_type} expected_{index}[{storage_size}] = {{{expected_init}}};")
                if elem_type in _FLOAT_C_TYPES:
                    condition = (f"double diff = (double)arg_{index}[i] - (double)expected_{index}[i]; "
                                 f"if (diff < 0) diff = -diff; if (diff > {atol:.17g})")
                    fail_fmt = (f'      fprintf(stderr, "verification failed: {name}[%d] got=%.9g '
                                f'expected=%.9g\\n", i, (double)arg_{index}[i], (double)expected_{index}[i]);')
                else:
                    condition = f"if (arg_{index}[i] != expected_{index}[i])"
                    fail_fmt = (f'      fprintf(stderr, "verification failed: {name}[%d] got=%lld '
                                f'expected=%lld\\n", i, (long long)arg_{index}[i], (long long)expected_{index}[i]);')
                checks.extend([
                    f"  for (int i = 0; i < {len(values)}; ++i) {{",
                    f"    {condition} {{",
                    fail_fmt,
                    "      return 1;",
                    "    }",
                    "  }",
                ])
        else:
            c_type = ty_to_cpp(ty)
            extern_types.append(c_type)
            call_args.append(_c_literal(arguments[name], c_type))

    extern_types.extend(["uint32_t"] * 6)
    call_prefix = ", ".join(call_args)
    if call_prefix:
        call_prefix += ", "
    gx, gy, gz = grid

    call = f"            {kernel_name}({call_prefix}x, y, z, {gx}, {gy}, {gz});"
    loop_open = [
        "#ifdef _OPENMP",
        "#pragma omp for collapse(3) schedule(static)",
        "#endif",
        f"      for (int x = 0; x < {gx}; ++x)",
        f"        for (int y = 0; y < {gy}; ++y)",
        f"          for (int z = 0; z < {gz}; ++z)",
    ]

    lines = [
        "#include <math.h>",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <time.h>",
        "#ifdef _OPENMP",
        "#include <omp.h>",
        "#endif",
        "",
        f"extern void {kernel_name}({', '.join(extern_types)});",
        "",
        "int main(void) {",
        *declarations,
        "",
        "  long bench_iters = 0, bench_warmup = 0;",
        "  const char *bench_env;",
        '  if ((bench_env = getenv("TRITON_BENCH_ITERS"))) bench_iters = atol(bench_env);',
        '  if ((bench_env = getenv("TRITON_BENCH_WARMUP"))) bench_warmup = atol(bench_env);',
        "  if (bench_iters < 0) bench_iters = 0;",
        "  if (bench_warmup < 0) bench_warmup = 0;",
        "",
        "#ifdef _OPENMP",
        "  #pragma omp parallel",
        "#endif",
        "  {",
        "    for (long bench_it = 0; bench_it < bench_warmup; ++bench_it) {",
        *loop_open,
        call,
        "    }",
        "  }",
        "  struct timespec bench_t0, bench_t1;",
        "  clock_gettime(CLOCK_MONOTONIC, &bench_t0);",
        "#ifdef _OPENMP",
        "  #pragma omp parallel",
        "#endif",
        "  {",
        "    for (long bench_it = 0; bench_it < (bench_iters > 0 ? bench_iters : 1); ++bench_it) {",
        *loop_open,
        call,
        "    }",
        "  }",
        "  clock_gettime(CLOCK_MONOTONIC, &bench_t1);",
        "  if (bench_iters > 0) {",
        "    double bench_ns = (bench_t1.tv_sec - bench_t0.tv_sec) * 1e9 +",
        "                      (double)(bench_t1.tv_nsec - bench_t0.tv_nsec);",
        '    printf("Time: %.4f ms (mean of %ld iters, %ld warmup)\\n",',
        "           bench_ns / 1e6 / bench_iters, bench_iters, bench_warmup);",
        "  }",
        "",
        *checks,
        '  puts("PASS");',
        "  return 0;",
        "}",
        "",
    ]
    return "\n".join(lines)


def build_standalone_executable(kernel_name: str, so_bytes: bytes, runner_source: str, output: str | os.PathLike,
                                 *, toolchain: Toolchain | None = None) -> tuple[Path, Path]:
    toolchain = toolchain or Toolchain.from_env()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    so_path = output.with_name(f"lib{kernel_name}.so")
    runner_path = output.with_suffix(".c")
    so_path.write_bytes(so_bytes)
    runner_path.write_text(runner_source)

    extra_args = ["-fopenmp"]
    if toolchain.sysroot and os.path.isfile(os.path.join(toolchain.sysroot, "usr", "lib", "libomp.a")):
        extra_args += ["-Wl,-Bstatic", "-lomp", "-Wl,-Bdynamic"]
    extra_args.append("-Wl,-rpath,$ORIGIN")
    cmd = toolchain.compile_command(
        [runner_path.name, so_path.name],
        output.name,
        extra_args=extra_args,
    )
    subprocess.run(cmd, check=True, cwd=output.parent)
    return output, so_path


def compile_kernel_to_elf(kernel, arguments: dict, grid: Sequence[int], output: str | os.PathLike, *,
                           constexprs: dict | None = None, signature: dict | None = None,
                           expected: dict | None = None, atol: float = 1e-5,
                           toolchain: Toolchain | None = None) -> tuple[Path, Path]:
    compiled, so_bytes = compile_kernel_to_so(kernel, arguments, constexprs=constexprs, signature=signature)
    runner = generate_runner(compiled.metadata.name, compiled.src.signature, arguments, grid, expected=expected,
                              atol=atol)
    return build_standalone_executable(compiled.metadata.name, so_bytes, runner, output, toolchain=toolchain)


def deploy_and_run(local_files: Sequence[str | os.PathLike], remote_host: str, remote_exe: str, *,
                    remote_dir: str = "~", ssh_opts: Sequence[str] = (), timeout: float = 60) -> subprocess.CompletedProcess:
    subprocess.run(["ssh", *ssh_opts, remote_host, f"mkdir -p {remote_dir}"], check=True, timeout=timeout)
    scp_cmd = ["scp", *ssh_opts, *[str(p) for p in local_files], f"{remote_host}:{remote_dir}"]
    subprocess.run(scp_cmd, check=True, timeout=timeout)
    remote_name = os.path.basename(remote_exe)
    env_prefix = "".join(f"{var}={os.environ[var]} " for var in ("TRITON_BENCH_ITERS", "TRITON_BENCH_WARMUP")
                          if var in os.environ)
    ssh_cmd = [
        "ssh", *ssh_opts, remote_host,
        f"chmod +x {remote_dir}/{remote_name} && cd {remote_dir} && {env_prefix}./{remote_name}"
    ]
    return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)


def compile_deploy_and_run(kernel, arguments: dict, grid: Sequence[int], output: str | os.PathLike,
                            remote_host: str, *, constexprs: dict | None = None, signature: dict | None = None,
                            expected: dict | None = None, atol: float = 1e-5, remote_dir: str = "~",
                            ssh_opts: Sequence[str] = (), toolchain: Toolchain | None = None) -> subprocess.CompletedProcess:
    exe_path, so_path = compile_kernel_to_elf(kernel, arguments, grid, output, constexprs=constexprs,
                                               signature=signature, expected=expected, atol=atol,
                                               toolchain=toolchain)
    return deploy_and_run([exe_path, so_path], remote_host, exe_path.name, remote_dir=remote_dir, ssh_opts=ssh_opts)


def standalone_kernel_cli(kernel, arguments: dict, grid: Sequence[int], *, default_output: str | os.PathLike,
                           default_host: str | None = None, constexprs: dict | None = None,
                           signature: dict | None = None, expected: dict | None = None, atol: float = 1e-5,
                           argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Build {kernel.__name__} for riscv64 and optionally run it remotely")
    parser.add_argument("--output", type=Path, default=Path(default_output))
    parser.add_argument("--host", default=default_host or os.environ.get("TRITON_RISCV_HOST"),
                         help="user@host to scp/ssh the kernel to, e.g. chlee@140.114.78.64")
    parser.add_argument("--remote-dir", default="~")
    parser.add_argument("--compile-only", action="store_true")
    parsed = parser.parse_args(argv)

    exe_path, so_path = compile_kernel_to_elf(kernel, arguments, grid, parsed.output, constexprs=constexprs,
                                               signature=signature, expected=expected, atol=atol)
    print(f"Executable: {exe_path}")
    print(f"Kernel .so: {so_path}")

    if parsed.compile_only:
        return 0
    if not parsed.host:
        raise SystemExit("No --host given (and TRITON_RISCV_HOST unset); pass --compile-only to skip running")

    result = deploy_and_run([exe_path, so_path], parsed.host, exe_path.name, remote_dir=parsed.remote_dir)
    print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode != 0 or "PASS" not in result.stdout:
        print(f"FAIL (exit code {result.returncode})")
        return 1
    print("PASS")
    return 0
