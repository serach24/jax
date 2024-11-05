
import ctypes

from jax._src import core
from jax._src.typing import (Array, ArrayLike, DeprecatedArg, DuckTypedArray,
                             Shape)
from jax._src.callback import _check_shape_dtype

from jax._src.interpreters import mlir

from jax._src.lib.mlir import ir

from jax._src.layout import DeviceLocalLayout

import numpy as np

from typing import Any

from collections.abc import Mapping, Sequence

from jax._src import dispatch
from jax._src import effects

ResultMetadata = DuckTypedArray | core.AbstractToken


def _result_avals(results: Sequence[ResultMetadata]) -> tuple[core.AbstractValue, ...]:
  avals: list[core.AbstractValue] = []
  for result in results:
    if isinstance(result, core.AbstractToken):
      avals.append(result)
    else:
      _check_shape_dtype(result)
      avals.append(core.ShapedArray(result.shape, result.dtype))
  return tuple(avals)



class HashableArray:
  __slots__ = ["val"]

  def __init__(self, val):
    assert isinstance(val, np.ndarray)
    self.val = np.copy(val)
    self.val.setflags(write=False)

  def __repr__(self):
    return f"HashableArray({self.val})"

  def __hash__(self):
    return hash((self.val.shape, self.val.dtype, self.val.tobytes()))

  def __eq__(self, other):
    return isinstance(other, HashableArray) and np.array_equal(self.val, other.val)


class HashableDict:
  __slots__ = ["val"]

  def __init__(self, val):
    assert isinstance(val, dict)
    self.val = tuple(sorted(val.items()))

  def __repr__(self):
    return f"HashableDict({dict(self.val)})"

  def __hash__(self):
    return hash(self.val)

  def __eq__(self, other):
    return isinstance(other, HashableDict) and self.val == other.val



# ffi_call must support some small non-hashable input arguments, like np.arrays
# and dicts, to support calling FFI targets with array inputs or user defined
# structs. Since these arguments will eventually be embedded in the HLO as
# dense attributes, we assume that they are small and hash by making an
# immutable copy and hashing by value.
def _wrap_kwargs_hashable(kwargs: dict[str, Any]) -> dict[str, Any]:
  hashable_kwargs: dict[str, Any] = {}
  for k, v in kwargs.items():
    if isinstance(v, np.ndarray):
      hashable_kwargs[k] = HashableArray(v)
    elif isinstance(v, dict):
      hashable_kwargs[k] = HashableDict(v)
    else:
      try:
        hash(v)
      except TypeError as e:
        raise TypeError(
            f"Non-hashable keyword argument to ffi_call {k}: {v}") from e
      else:
        hashable_kwargs[k] = v
  return hashable_kwargs

def ptx_call(
    ptx_code: str,
    kernel_name: str,
    result_shape_dtypes: ResultMetadata | Sequence[ResultMetadata],
    *args: ArrayLike,
    has_side_effect: bool = False,
    vmap_method: str | None = None,
    vectorized: bool | DeprecatedArg = DeprecatedArg(),
    **kwargs: Any,
) -> Array | list[Array]: # type: ignore
  print("args", args)
  if isinstance(result_shape_dtypes, Sequence):
    multiple_results = True
    result_avals = _result_avals(result_shape_dtypes)
  else:
    multiple_results = False
    result_avals = _result_avals((result_shape_dtypes,))
  results = ptx_call_p.bind(
      *args,
      result_avals=result_avals,
      vectorized=vectorized,
      vmap_method=vmap_method,
      kernel_name=kernel_name,
      ptx_code=ptx_code,
      has_side_effect=has_side_effect,
      **_wrap_kwargs_hashable(kwargs),
  )
  if multiple_results:
    return results
  else:
    return results[0]

def _unwrap_kwargs_hashable(kwargs: dict[str, Any]) -> dict[str, Any]:
  unwrapped_kwargs: dict[str, Any] = {}
  for k, v in kwargs.items():
    if isinstance(v, HashableArray):
      unwrapped_kwargs[k] = v.val
    elif isinstance(v, HashableDict):
      unwrapped_kwargs[k] = dict(v.val)
    else:
      unwrapped_kwargs[k] = v
  return unwrapped_kwargs

def _aval_shape(aval: core.AbstractValue) -> Shape:
  return () if aval is core.abstract_token else aval.shape  # pytype: disable=attribute-error

def _convert_layout(aval: core.AbstractValue) -> Sequence[int]:
  """Convert a layout to the minor-to-major order used by the custom call API."""
  return list(reversed(range(len(_aval_shape(aval)))))

def pycapsule(funcptr):
  """Wrap a ctypes function pointer in a PyCapsule.

  The primary use of this function, and the reason why it lives with in the
  ``jax.extend.ffi`` submodule, is to wrap function calls from external
  compiled libraries to be registered as XLA custom calls.

  Example usage::

    import ctypes
    import jax
    from jax.lib import xla_client

    libfoo = ctypes.cdll.LoadLibrary('./foo.so')
    xla_client.register_custom_call_target(
        name="bar",
        fn=jax.extend.ffi.pycapsule(libfoo.bar),
        platform=PLATFORM,
        api_version=API_VERSION
    )

  Args:
    funcptr: A function pointer loaded from a dynamic library using ``ctypes``.

  Returns:
    An opaque ``PyCapsule`` object wrapping ``funcptr``.
  """
  destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
  builder = ctypes.pythonapi.PyCapsule_New
  builder.restype = ctypes.py_object
  builder.argtypes = (ctypes.c_void_p, ctypes.c_char_p, destructor)
  return builder(funcptr, None, destructor(0))


PtxLayoutOptions = Sequence[int] | DeviceLocalLayout | None
# TODO(chenhao): add args
def ptx_lowering(
    ptx_code: str,
    kernel_name: str,
    *,
    operand_layouts: None = None,
    result_layouts: Sequence[PtxLayoutOptions] | None = None,
    backend_config: Mapping[str, ir.Attribute] | None = None,
    **lowering_args: Any
) -> mlir.LoweringRule:
    def _lowering(ctx: mlir.LoweringRuleContext, *operands: ir.Value, **params: Any) -> Sequence[ir.Value | Sequence[ir.Value]]:
        kwargs = dict()
        kwargs.setdefault("api_version", 4)
        # We currently only support PTX
        # params["name"] = kernel_name
        # params["device_kernel_type"] = "ptx"
        # params["ptx_code"] = ptx_code
        # kwargs["device_kernel_type"] = "ptx"
        # kwargs["ptx_code"] = ptx_code
        backend_config = dict(
          name=kernel_name,
          source=ptx_code,
        )
        backend_config = {k: mlir.ir_attribute(v) for k, v in backend_config.items()}

        result_types = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
        # kwargs["backend_config"] = {k: mlir.ir_attribute(v) for k, v in params.items()}

        if "result_types" not in kwargs:
            kwargs["result_types"] = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]
        # kwargs["operand_layouts"] = map(_convert_layout, ctx.avals_in)
        # kwargs["result_layouts"] = map(_convert_layout, ctx.avals_out)
        if "result_shapes" not in kwargs and not all(
            core.is_constant_shape(_aval_shape(aval)) for aval in ctx.avals_out):
            kwargs["result_shapes"] = [
            mlir.shape_tensor(mlir.eval_dynamic_shape_as_ivals(ctx, _aval_shape(aval)))
            for aval in ctx.avals_out]

        return mlir.custom_call("__gpu$xla.gpu.ptx", operands=operands, result_types=result_types, backend_config=backend_config).results  # type: ignore

    return _lowering

class PtxEffect(effects.Effect):
  def __str__(self):
    return "PTX"

_PtxEffect = PtxEffect()
effects.lowerable_effects.add_type(PtxEffect)
effects.control_flow_allowed_effects.add_type(PtxEffect)



def ptx_call_abstract_eval(
    *avals_in,
    result_avals: tuple[core.AbstractValue, ...],
    ptx_code: str,
    kernel_name: str,
    vectorized: bool | DeprecatedArg,
    vmap_method: str | None,
    has_side_effect: bool,
    **kwargs: Any,
):
  del avals_in, kernel_name, ptx_code, vectorized, vmap_method, kwargs
  # effects = {_PtxEffect} if has_side_effect else core.no_effects
  effects = core.no_effects
  return result_avals, effects


def ptx_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    result_avals: tuple[core.AbstractValue, ...],
    kernel_name: str,
    ptx_code: str,
    vectorized: bool | DeprecatedArg,
    vmap_method: str | None,
    has_side_effect: bool,
    **kwargs: Any,
) -> Sequence[ir.Value]:
    del result_avals, vectorized, vmap_method
    rule = ptx_lowering(ptx_code, kernel_name, has_side_effect=has_side_effect)
    return rule(ctx, *operands, **_unwrap_kwargs_hashable(kwargs))

ptx_call_p = core.Primitive("ptx_call")
ptx_call_p.multiple_results = True
dispatch.simple_impl(ptx_call_p)
ptx_call_p.def_effectful_abstract_eval(ptx_call_abstract_eval)
# ffi_call_p.def_effectful_abstract_eval(ffi_call_abstract_eval)
# ad.primitive_jvps[ffi_call_p] = ffi_call_jvp
# ad.primitive_transposes[ffi_call_p] = ffi_call_transpose
# batching.primitive_batchers[ffi_call_p] = functools.partial(
#     callback_batching_rule, ffi_call_p)
mlir.register_lowering(ptx_call_p, ptx_call_lowering)

