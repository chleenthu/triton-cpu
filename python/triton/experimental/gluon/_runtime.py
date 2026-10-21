from __future__ import annotations
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import Language
from triton.runtime.jit import JITFunction, constexpr_function
from typing import TypeVar, Optional, Callable, Iterable, Union, overload
from triton._C.libtriton import ir, passes, gluon_ir

T = TypeVar("T")

__all__ = ["GluonJITFunction", "constexpr_function", "jit"]


class _CPUGluonOptions:
    """CPU backend options plus the GPU fields the Gluon frontend reads."""

    def __init__(self, options):
        self._options = options
        self.arch = ""
        self.warp_size = 32
        # Layouts are verified against these, so give the usual GPU defaults.
        self.num_warps = options.num_warps or 4
        self.num_ctas = options.num_ctas or 1

    def __getattr__(self, name):
        return getattr(self._options, name)


def _gluon_to_ttir(module, context):
    # Resolve layouts like the GPU backends do, then drop them: the CPU backend
    # only consumes plain TTIR.
    pm = ir.pass_manager(context)
    pm.enable_debug()
    passes.gluon.add_inliner(pm)
    passes.gluon.add_infer_coalesced_encodings(pm)
    passes.gluon.add_resolve_auto_encodings(pm)
    passes.gluon.add_canonicalizer(pm)
    gluon_ir.add_strip_encodings(pm)
    pm.run(module, "gluon_to_ttir")
    return module


class GluonASTSource(ASTSource):

    def __init__(self, fn, signature, constexprs=None, attrs=None, target=None) -> None:
        super().__init__(fn, signature, constexprs, attrs)
        if target is None:
            from triton.runtime import driver
            target = driver.active.get_current_target()
        # The CPU backend has no Gluon stages: hand it TTIR instead of GLIR.
        self.is_cpu = target.backend == "cpu"
        self.language = Language.TRITON if self.is_cpu else Language.GLUON
        self.ext = "ttir" if self.is_cpu else "glir"

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from triton.compiler.compiler import make_backend
        from triton.compiler.code_generator import ast_to_ttir

        if target.backend == "cpu":
            assert self.is_cpu, "GluonASTSource was created for a GPU target; pass target= explicitly"
            return self._make_cpu_ir(options, codegen_fns, module_map, context)

        builder = ir.builder(context)
        module = builder.create_module()

        # Assign module attributes eagerly, as they are needed to verify layouts
        backend = make_backend(target)
        target = backend.get_target_name(options)

        module.set_attr("ttg.target", builder.get_string_attr(target))
        module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
        module.set_attr("ttg.num-ctas", builder.get_int32_attr(options.num_ctas))
        module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(options.warp_size))

        is_cuda = options.backend_name == "cuda"
        if is_cuda and options.maxnreg is not None:
            module.set_attr("ttg.maxnreg", builder.get_int32_attr(options.maxnreg))

        module = ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                             module_map=module_map, module=module)
        return module

    def _make_cpu_ir(self, options, codegen_fns, module_map, context):
        from triton.compiler.code_generator import ast_to_ttir

        options = _CPUGluonOptions(options)
        builder = ir.builder(context)
        module = builder.create_module()
        module.set_attr("ttg.target", builder.get_string_attr("cpu"))
        module.set_attr("ttg.num-warps", builder.get_int32_attr(options.num_warps))
        module.set_attr("ttg.num-ctas", builder.get_int32_attr(options.num_ctas))
        module.set_attr("ttg.threads-per-warp", builder.get_int32_attr(options.warp_size))
        module = ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                             module_map=module_map, module=module)
        return _gluon_to_ttir(module, context)


class GluonJITFunction(JITFunction[T]):

    def create_binder(self):
        result = super().create_binder()
        self.ASTSource = GluonASTSource
        return result

    def is_gluon(self):
        return True


@overload
def jit(fn: T) -> GluonJITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], GluonJITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[GluonJITFunction[T], Callable[[T], JITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> JITFunction[T]:
        return GluonJITFunction(
            fn,
            version=version,
            do_not_specialize=do_not_specialize,
            do_not_specialize_on_alignment=do_not_specialize_on_alignment,
            debug=debug,
            noinline=noinline,
            repr=repr,
            launch_metadata=launch_metadata,
        )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
