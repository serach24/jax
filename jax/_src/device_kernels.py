# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ctypes

from jax._src import core
from jax._src.typing import (Array, ArrayLike, DeprecatedArg, DuckTypedArray,
                             Shape)
from jax._src.callback import _check_shape_dtype

from jax._src.interpreters import mlir

from jax._src.lib.mlir import ir

from jax._src.layout import Layout

import numpy as np

from typing import Any

from collections.abc import Sequence

from jax._src import dispatch
from jax._src import effects

# Import existing implementations from ffi.py
from jax._src.ffi import (
    _result_avals, HashableDict, _aval_shape
)
from jax._src.hashable_array import HashableArray

# Create wrapper functions to maintain dict interface
def _wrap_kwargs_hashable(kwargs: dict[str, Any]) -> dict[str, Any]:
  """Wrapper to maintain dict interface while using ffi implementation."""
  from jax._src.ffi import _wrap_kwargs_hashable as ffi_wrap
  wrapped = ffi_wrap(kwargs)
  return dict(wrapped)

def _unwrap_kwargs_hashable(kwargs: dict[str, Any]) -> dict[str, Any]:
  """Wrapper to maintain dict interface while using ffi implementation."""
  from jax._src.ffi import _unwrap_kwargs_hashable as ffi_unwrap
  return ffi_unwrap(tuple(kwargs.items()))

ResultMetadata = DuckTypedArray | core.AbstractToken

KERNEL_TYPE_TO_CALL_TARGET: dict[str, str] = {
    "ptx": "__gpu$xla.gpu.ptx",
}
SUPPORTED_KERNEL_TYPES: list[str] = list(KERNEL_TYPE_TO_CALL_TARGET.keys())

def _normalize_grid_block_dims(grid_dims: int | tuple[int, ...] | list[int], block_dims: int | tuple[int, ...] | list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
  if isinstance(grid_dims, int):
    grid_dims = (grid_dims, 1, 1)
  elif len(grid_dims) == 1:
    grid_dims = (grid_dims[0], 1, 1)
  elif len(grid_dims) == 2:
    grid_dims = (*grid_dims, 1)
  else:
    raise ValueError(f"Invalid grid dimensions: {grid_dims}")

  if isinstance(block_dims, int):
    block_dims = (block_dims, 1, 1)
  elif len(block_dims) == 1:
    block_dims = (block_dims[0], 1, 1)
  elif len(block_dims) == 2:
    block_dims = (*block_dims, 1)
  else:
    raise ValueError(f"Invalid block dimensions: {block_dims}")

  return grid_dims, block_dims

def kernel_call(
    kernel_content: str,
    kernel_name: str,
    result_shape_dtypes: ResultMetadata | Sequence[ResultMetadata],
    *args: ArrayLike,
    kernel_type: str = "ptx",
    grid_dims: int | tuple[int, ...] | list[int] = (1, 1, 1),
    block_dims: int | tuple[int, ...] | list[int] = (256, 1, 1),
    shared_mem_bytes: int = 0,
    has_side_effect: bool = False,
    output_indices: Sequence[int] | None = None,
    vmap_method: str | None = None,
    vectorized: bool | DeprecatedArg = DeprecatedArg(),
    **kwargs: Any,
) -> Array | list[Array]:  # type: ignore
    """Call a device kernel with the specified kernel type.
    
    Currently supported kernel types:
    - ptx: NVIDIA PTX kernel code for CUDA GPUs
    
    Args:
        kernel_content: Source code for the kernel
        kernel_name: Name of the kernel function to call
        result_shape_dtypes: Shape and dtype of the result(s)
        *args: Input arrays
        kernel_type: Type of kernel code (e.g., "ptx")
        grid_dims: Grid dimensions for kernel launch
        block_dims: Block dimensions for kernel launch
        shared_mem_bytes: Bytes of shared memory to allocate
        has_side_effect: Whether the kernel has side effects
        output_indices: Indices of outputs in argument list
        vmap_method: Method for vmapping the kernel
        vectorized: Whether the kernel is vectorized
        **kwargs: Additional arguments for specific kernel types
        
    Returns:
        Result array(s) from kernel execution
    """
    # Validate kernel_type
    if kernel_type not in SUPPORTED_KERNEL_TYPES:
        raise ValueError(f"Unsupported kernel type: {kernel_type}. Supported types are: {SUPPORTED_KERNEL_TYPES}")

    # Kernel-specific validation
    if kernel_type == "ptx" and ".entry" not in kernel_content:
        raise ValueError("PTX code must contain an .entry point")

    if isinstance(result_shape_dtypes, Sequence):
        multiple_results = True
        result_avals = _result_avals(result_shape_dtypes)
    else:
        multiple_results = False
        result_avals = _result_avals((result_shape_dtypes,))

    if output_indices is not None:
        expected_num_outputs = len(result_avals)
        if not isinstance(output_indices, (list, tuple)):
            raise ValueError("output_indices must be a sequence")
        if len(output_indices) != expected_num_outputs:
            raise ValueError(
                f"Expected {expected_num_outputs} output indices but got {len(output_indices)}"
            )
        if not all(isinstance(idx, int) and 0 <= idx < len(args) + expected_num_outputs for idx in output_indices):
            raise ValueError(
                f"Output indices must be integers in range [0, {len(args)}), got {output_indices}"
            )

    output_indices = np.array([] if output_indices is None else output_indices)

    grid_dims, block_dims = _normalize_grid_block_dims(grid_dims, block_dims)
    call_target = KERNEL_TYPE_TO_CALL_TARGET[kernel_type]
  
    kernel_kwargs = {
        "grid_x": grid_dims[0],
        "grid_y": grid_dims[1],
        "grid_z": grid_dims[2],
        "block_x": block_dims[0],
        "block_y": block_dims[1],
        "block_z": block_dims[2],
        "shared_mem_bytes": shared_mem_bytes,
        "output_indices": output_indices,
        "call_target": call_target,
        **kwargs,
    }
    
    results = kernel_call_p.bind(
        *args,
        result_avals=result_avals,
        vectorized=vectorized,
        vmap_method=vmap_method,
        kernel_name=kernel_name,
        kernel_content=kernel_content,  
        has_side_effect=has_side_effect,
        **_wrap_kwargs_hashable(kernel_kwargs),
    )
    return results if multiple_results else results[0]

def kernel_lowering(
    kernel_content: str,
    kernel_name: str,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    block_x: int,
    block_y: int,
    block_z: int,
    shared_mem_bytes: int,
    call_target: str,
    output_indices: Sequence[int] | None = None,
    has_side_effect: bool = False,
    **lowering_args: Any
) -> mlir.LoweringRule:
    def _lowering(
        ctx: mlir.LoweringRuleContext, 
        *operands: ir.Value, 
        **params: Any
    ) -> Sequence[ir.Value | Sequence[ir.Value]]:
        kwargs = {"api_version": 4}
        
        if isinstance(output_indices, HashableArray):
            output_indices_val = list(output_indices.val)
        elif output_indices is not None:
            output_indices_val = list(output_indices)
        else:
            output_indices_val = []

        backend_config = {
            "name": kernel_name,
            "source": kernel_content,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "block_x": block_x,
            "block_y": block_y,
            "block_z": block_z,
            "shared_mem_bytes": shared_mem_bytes,
            "output_indices": output_indices_val,
        }
        
        backend_config = {k: mlir.ir_attribute(v) for k, v in backend_config.items()}

        result_types = [mlir.aval_to_ir_type(aval) for aval in ctx.avals_out]

        if "result_types" not in kwargs:
            kwargs["result_types"] = result_types

        if "result_shapes" not in kwargs and not all(
            core.is_constant_shape(_aval_shape(aval)) for aval in ctx.avals_out
        ):
            kwargs["result_shapes"] = [
                mlir.shape_tensor(mlir.eval_dynamic_shape_as_ivals(ctx, _aval_shape(aval)))
                for aval in ctx.avals_out
            ]

        if has_side_effect:
            kwargs["has_side_effect"] = True

        return mlir.custom_call(
            call_target,
            operands=operands,
            result_types=result_types,
            backend_config=backend_config,
            has_side_effect=has_side_effect
        ).results

    return _lowering

class KernelEffect(effects.Effect):
  def __str__(self):
    return "Kernel"

_KernelEffect = KernelEffect()
effects.lowerable_effects.add_type(KernelEffect)
effects.control_flow_allowed_effects.add_type(KernelEffect)

def kernel_call_abstract_eval(
    *avals_in,
    result_avals: tuple[core.AbstractValue, ...],
    kernel_content: str,
    kernel_name: str,
    vectorized: bool | DeprecatedArg,
    vmap_method: str | None,
    has_side_effect: bool,
    **kwargs: Any,
):
    del avals_in, kernel_name, kernel_content, vectorized, vmap_method, kwargs
    if has_side_effect:
        effects = {_KernelEffect}  # Use the defined KernelEffect when has_side_effect is True
    else:
        effects = core.no_effects
    return result_avals, effects


def kernel_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    result_avals: tuple[core.AbstractValue, ...],
    kernel_name: str,
    kernel_content: str,
    vectorized: bool | DeprecatedArg,
    vmap_method: str | None,
    has_side_effect: bool,
    **kwargs: Any,
) -> Sequence[ir.Value]:
    del result_avals, vectorized, vmap_method
    
    call_target = kwargs.get("call_target")
    if call_target is None:
        raise ValueError("call_target must be provided")
    
    rule = kernel_lowering(
        kernel_content,
        kernel_name,
        kwargs["grid_x"],
        kwargs["grid_y"], 
        kwargs["grid_z"],
        kwargs["block_x"],
        kwargs["block_y"],
        kwargs["block_z"],
        kwargs["shared_mem_bytes"],
        call_target,
        kwargs["output_indices"],
        has_side_effect=has_side_effect,
    )
    
    return rule(ctx, *operands)

kernel_call_p = core.Primitive("kernel_call")
kernel_call_p.multiple_results = True
dispatch.simple_impl(kernel_call_p)
kernel_call_p.def_effectful_abstract_eval(kernel_call_abstract_eval)
mlir.register_lowering(kernel_call_p, kernel_call_lowering)

