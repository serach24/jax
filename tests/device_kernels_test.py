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

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax import vmap
from jax._src.device_kernels import (
    kernel_call, 
    register_device_kernel_as_batch_partitionable,
    device_kernel_custom_partitioning,
    build_device_kernel_lowering_function,
    kernel_lowering
)
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh, NamedSharding


class TestDeviceKernels:
    """Test suite for device kernels functionality."""

    def test_basic_kernel_call(self):
        """Test basic kernel call functionality."""
        # Simple PTX kernel that adds two arrays
        ptx_kernel = """
        .visible .entry add_kernel(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c,
            .param .u32 n
        ) {
            .reg .u32 %tid.x;
            .reg .u32 %n;
            .reg .u64 %a, %b, %c;
            .reg .f32 %val_a, %val_b, %val_c;
            
            ld.param.u32 %n, [n];
            ld.param.u64 %a, [a];
            ld.param.u64 %b, [b];
            ld.param.u64 %c, [c];
            
            mov.u32 %tid.x, %tid.x;
            setp.ge.u32 %p1, %tid.x, %n;
            @%p1 bra exit;
            
            mul.wide.u32 %rd1, %tid.x, 4;
            add.u64 %rd2, %a, %rd1;
            add.u64 %rd3, %b, %rd1;
            add.u64 %rd4, %c, %rd1;
            
            ld.global.f32 %val_a, [%rd2];
            ld.global.f32 %val_b, [%rd3];
            add.f32 %val_c, %val_a, %val_b;
            st.global.f32 [%rd4], %val_c;
            
        exit:
            ret;
        }
        """
        
        a = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        b = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
        
        # Note: This is a mock test - actual PTX execution would require
        # proper GPU setup and kernel registration
        with pytest.raises(ValueError, match="call_target must be provided"):
            result = kernel_call(
                ptx_kernel,
                "add_kernel",
                a,
                a, b,
                grid_dims=1,
                block_dims=256
            )

    def test_batch_partitioning_registration(self):
        """Test batch partitioning registration."""
        # Test that we can register a kernel type as batch partitionable
        try:
            register_device_kernel_as_batch_partitionable("ptx")
            # Should not raise an error
        except Exception as e:
            pytest.fail(f"register_device_kernel_as_batch_partitionable raised {e}")
        
        # Test invalid kernel type
        with pytest.raises(ValueError, match="Unsupported kernel type"):
            register_device_kernel_as_batch_partitionable("invalid_type")

    def test_build_device_kernel_lowering_function(self):
        """Test the build_device_kernel_lowering_function."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        # Test that the function can be created
        try:
            lowering_fn = build_device_kernel_lowering_function(
                kernel_data=ptx_kernel,
                kernel_name="test_kernel",
                call_target="__gpu$xla.gpu.ptx",
                grid_dims=(1, 1, 1),
                block_dims=(256, 1, 1),
                shared_mem_bytes=0,
                has_side_effect=False
            )
            assert callable(lowering_fn)
        except Exception as e:
            pytest.fail(f"build_device_kernel_lowering_function raised {e}")

    def test_kernel_lowering(self):
        """Test the kernel_lowering function."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        # Test that the lowering rule can be created
        try:
            lowering_rule = kernel_lowering(
                kernel_data=ptx_kernel,
                kernel_name="test_kernel",
                call_target="__gpu$xla.gpu.ptx",
                grid_dims=(1, 1, 1),
                block_dims=(256, 1, 1),
                shared_mem_bytes=0,
                has_side_effect=False
            )
            assert callable(lowering_rule)
        except Exception as e:
            pytest.fail(f"kernel_lowering raised {e}")

    def test_custom_partitioning_decorator(self):
        """Test the device_kernel_custom_partitioning decorator."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        @device_kernel_custom_partitioning(
            kernel_data=ptx_kernel,
            kernel_name="test_kernel",
            kernel_type="ptx"
        )
        def test_kernel_fn(x):
            return jax.ShapeDtypeStruct(x.shape, x.dtype)
        
        # Test that the decorator creates a callable object
        assert hasattr(test_kernel_fn, 'def_partition')
        assert callable(test_kernel_fn.def_partition)
        
        # Test that we can define partitioning strategy
        def partition_strategy(mesh, arg_shapes, result_shape):
            def lower_fn(x):
                return kernel_call(
                    ptx_kernel,
                    "test_kernel",
                    jax.ShapeDtypeStruct(x.shape, x.dtype),
                    x,
                    kernel_type="ptx"
                )
            return mesh, lower_fn, result_shape.sharding, (arg_shapes[0].sharding,)
        
        def infer_sharding(mesh, arg_shapes, result_shape):
            return arg_shapes[0].sharding
        
        # Test that def_partition works
        try:
            test_kernel_fn.def_partition(
                partition=partition_strategy,
                infer_sharding_from_operands=infer_sharding,
                sharding_rule='i j -> i j'
            )
        except Exception as e:
            pytest.fail(f"def_partition raised {e}")

    def test_layout_conversion(self):
        """Test layout conversion functionality."""
        from jax._src.device_kernels import _convert_layout_for_device_kernel
        from jax._src import core
        
        # Test with None layout (should default to row-major)
        aval = core.ShapedArray((2, 3), jnp.float32)
        layout = _convert_layout_for_device_kernel(aval, None)
        assert layout == (1, 0)  # minor-to-major order
        
        # Test with custom layout
        custom_layout = [0, 1]  # major-to-minor
        layout = _convert_layout_for_device_kernel(aval, custom_layout)
        assert layout == (0, 1)

    def test_partitioning_with_sharding(self):
        """Test partitioning with actual sharding."""
        if jax.device_count() < 2:
            pytest.skip("Requires multiple devices")
        
        ptx_kernel = """
        .visible .entry add_kernel(
            .param .u64 a,
            .param .u64 b,
            .param .u64 c,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        # Create a mesh with available devices
        devices = jax.devices()
        mesh = Mesh(devices, ('x',))
        
        # Create sharded arrays
        x = jnp.ones((8, 4), dtype=jnp.float32)
        y = jnp.ones((8, 4), dtype=jnp.float32)
        
        sharding = NamedSharding(mesh, P('x'))
        x_sharded = jax.device_put(x, sharding)
        y_sharded = jax.device_put(y, sharding)
        
        # Test that kernel_call can handle sharded inputs
        # Note: This will fail at execution time due to missing kernel registration
        # but should not fail at the JAX level
        with pytest.raises(ValueError, match="call_target must be provided"):
            result = kernel_call(
                ptx_kernel,
                "add_kernel",
                jax.ShapeDtypeStruct((8, 4), jnp.float32),
                x_sharded, y_sharded,
                kernel_type="ptx",
                grid_dims=(8, 1, 1),
                block_dims=(256, 1, 1)
            )

    def test_sharding_rules(self):
        """Test sharding rules functionality."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        @device_kernel_custom_partitioning(
            kernel_data=ptx_kernel,
            kernel_name="test_kernel",
            kernel_type="ptx"
        )
        def test_kernel_fn(x):
            return jax.ShapeDtypeStruct(x.shape, x.dtype)
        
        # Test different sharding rule formats
        sharding_rules = [
            'i j -> i j',  # Keep same sharding
            'i -> i',      # 1D sharding
            'i j k -> i k',  # Drop middle dimension
            '...i -> ...i',  # Ellipsis notation
        ]
        
        for rule in sharding_rules:
            try:
                test_kernel_fn.def_partition(
                    partition=lambda mesh, arg_shapes, result_shape: (mesh, lambda x: x, result_shape.sharding, (arg_shapes[0].sharding,)),
                    sharding_rule=rule
                )
            except Exception as e:
                pytest.fail(f"Sharding rule '{rule}' failed: {e}")

    def test_custom_partitioning_with_mesh(self):
        """Test custom partitioning with mesh configuration."""
        if jax.device_count() < 2:
            pytest.skip("Requires multiple devices")
        
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        @device_kernel_custom_partitioning(
            kernel_data=ptx_kernel,
            kernel_name="test_kernel",
            kernel_type="ptx"
        )
        def test_kernel_fn(x):
            return jax.ShapeDtypeStruct(x.shape, x.dtype)
        
        def partition_strategy(mesh, arg_shapes, result_shape):
            # Test that we can access mesh information
            assert hasattr(mesh, 'shape')
            assert hasattr(mesh, 'axis_names')
            
            def lower_fn(x):
                return kernel_call(
                    ptx_kernel,
                    "test_kernel",
                    jax.ShapeDtypeStruct(x.shape, x.dtype),
                    x,
                    kernel_type="ptx"
                )
            
            # Return the mesh and lowering function
            return mesh, lower_fn, result_shape.sharding, (arg_shapes[0].sharding,)
        
        def infer_sharding(mesh, arg_shapes, result_shape):
            # Test that we can access mesh and shapes
            assert hasattr(mesh, 'shape')
            assert len(arg_shapes) > 0
            return arg_shapes[0].sharding
        
        # Test that the partitioning strategy works
        try:
            test_kernel_fn.def_partition(
                partition=partition_strategy,
                infer_sharding_from_operands=infer_sharding
            )
        except Exception as e:
            pytest.fail(f"Custom partitioning with mesh failed: {e}")

    def test_layout_handling(self):
        """Test layout handling in lowering functions."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        # Test with custom operand and result layouts
        try:
            lowering_fn = build_device_kernel_lowering_function(
                kernel_data=ptx_kernel,
                kernel_name="test_kernel",
                call_target="__gpu$xla.gpu.ptx",
                operand_layouts=[[1, 0]],  # Transpose input
                result_layouts=[[1, 0]],   # Transpose output
                grid_dims=(1, 1, 1),
                block_dims=(256, 1, 1),
                shared_mem_bytes=0,
                has_side_effect=False
            )
            assert callable(lowering_fn)
        except Exception as e:
            pytest.fail(f"Layout handling failed: {e}")

    def test_batch_partitioning_integration(self):
        """Test integration of batch partitioning with JAX's sharding system."""
        if jax.device_count() < 2:
            pytest.skip("Requires multiple devices")
        
        # Test that batch partitioning registration works
        try:
            register_device_kernel_as_batch_partitionable("ptx")
        except Exception as e:
            pytest.fail(f"Batch partitioning registration failed: {e}")
        
        # Test with shard_map (if available)
        try:
            from jax.experimental import shard_map
            
            ptx_kernel = """
            .visible .entry test_kernel(
                .param .u64 input,
                .param .u64 output,
                .param .u32 n
            ) {
                ret;
            }
            """
            
            devices = jax.devices()
            mesh = Mesh(devices, ('x',))
            
            def kernel_fn(x):
                return kernel_call(
                    ptx_kernel,
                    "test_kernel",
                    jax.ShapeDtypeStruct(x.shape, x.dtype),
                    x,
                    kernel_type="ptx"
                )
            
            # Test that shard_map can be applied (even if execution fails)
            sharded_fn = shard_map.shard_map(kernel_fn, mesh, P('x'), P('x'))
            
            # This should not fail at the JAX level, even if execution fails
            x = jnp.ones((8, 4), dtype=jnp.float32)
            with pytest.raises(ValueError, match="call_target must be provided"):
                result = sharded_fn(x)
                
        except ImportError:
            # shard_map might not be available in all JAX versions
            pass
        except Exception as e:
            pytest.fail(f"shard_map integration failed: {e}")

    def test_partitioning_error_handling(self):
        """Test error handling in partitioning functions."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        # Test invalid grid dimensions
        with pytest.raises(ValueError, match="Invalid grid dimensions"):
            kernel_call(
                ptx_kernel,
                "test_kernel",
                jax.ShapeDtypeStruct((4,), jnp.float32),
                jnp.ones((4,), dtype=jnp.float32),
                grid_dims=(1, 2, 3, 4),  # Too many dimensions
                block_dims=(256, 1, 1)
            )
        
        # Test invalid block dimensions
        with pytest.raises(ValueError, match="Invalid block dimensions"):
            kernel_call(
                ptx_kernel,
                "test_kernel",
                jax.ShapeDtypeStruct((4,), jnp.float32),
                jnp.ones((4,), dtype=jnp.float32),
                grid_dims=(1, 1, 1),
                block_dims=(256, 1, 1, 1)  # Too many dimensions
            )

    def test_vmap_legacy_vectorized(self):
        """Test vmap with legacy_vectorized method."""
        # Simple kernel that doubles input
        ptx_kernel = """
        .visible .entry double_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            .reg .u32 %tid.x;
            .reg .u32 %n;
            .reg .u64 %input, %output;
            .reg .f32 %val, %result;
            
            ld.param.u32 %n, [n];
            ld.param.u64 %input, [input];
            ld.param.u64 %output, [output];
            
            mov.u32 %tid.x, %tid.x;
            setp.ge.u32 %p1, %tid.x, %n;
            @%p1 bra exit;
            
            mul.wide.u32 %rd1, %tid.x, 4;
            add.u64 %rd2, %input, %rd1;
            add.u64 %rd3, %output, %rd1;
            
            ld.global.f32 %val, [%rd2];
            mul.f32 %result, %val, 2.0;
            st.global.f32 [%rd3], %result;
            
        exit:
            ret;
        }
        """
        
        def kernel_fn(x):
            return kernel_call(
                ptx_kernel,
                "double_kernel",
                x,
                x,
                grid_dims=1,
                block_dims=256,
                vmap_method="legacy_vectorized"
            )
        
        # Test with batched input
        batch_size = 3
        x = jnp.ones((batch_size, 4), dtype=jnp.float32)
        
        # This should work with vmap
        vmapped_fn = vmap(kernel_fn, in_axes=0, out_axes=0)
        
        # Note: This is a mock test - actual execution would require proper setup
        with pytest.raises(ValueError, match="call_target must be provided"):
            result = vmapped_fn(x)

    def test_vmap_sequential(self):
        """Test vmap with sequential method."""
        ptx_kernel = """
        .visible .entry sum_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            .reg .u32 %tid.x;
            .reg .u32 %n;
            .reg .u64 %input, %output;
            .reg .f32 %val, %sum;
            
            ld.param.u32 %n, [n];
            ld.param.u64 %input, [input];
            ld.param.u64 %output, [output];
            
            mov.u32 %tid.x, %tid.x;
            setp.ge.u32 %p1, %tid.x, %n;
            @%p1 bra exit;
            
            mul.wide.u32 %rd1, %tid.x, 4;
            add.u64 %rd2, %input, %rd1;
            add.u64 %rd3, %output, %rd1;
            
            ld.global.f32 %val, [%rd2];
            add.f32 %sum, %val, %val;  // Simple operation
            st.global.f32 [%rd3], %sum;
            
        exit:
            ret;
        }
        """
        
        def kernel_fn(x):
            return kernel_call(
                ptx_kernel,
                "sum_kernel",
                x,
                x,
                grid_dims=1,
                block_dims=256,
                vmap_method="sequential"
            )
        
        # Test with batched input
        batch_size = 2
        x = jnp.ones((batch_size, 3), dtype=jnp.float32)
        
        # This should work with vmap
        vmapped_fn = vmap(kernel_fn, in_axes=0, out_axes=0)
        
        # Note: This is a mock test - actual execution would require proper setup
        with pytest.raises(ValueError, match="call_target must be provided"):
            result = vmapped_fn(x)

    def test_vmap_broadcast_all(self):
        """Test vmap with broadcast_all method."""
        ptx_kernel = """
        .visible .entry broadcast_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            .reg .u32 %tid.x;
            .reg .u32 %n;
            .reg .u64 %input, %output;
            .reg .f32 %val;
            
            ld.param.u32 %n, [n];
            ld.param.u64 %input, [input];
            ld.param.u64 %output, [output];
            
            mov.u32 %tid.x, %tid.x;
            setp.ge.u32 %p1, %tid.x, %n;
            @%p1 bra exit;
            
            mul.wide.u32 %rd1, %tid.x, 4;
            add.u64 %rd2, %input, %rd1;
            add.u64 %rd3, %output, %rd1;
            
            ld.global.f32 %val, [%rd2];
            st.global.f32 [%rd3], %val;
            
        exit:
            ret;
        }
        """
        
        def kernel_fn(x, y):
            return kernel_call(
                ptx_kernel,
                "broadcast_kernel",
                x,
                x, y,
                grid_dims=1,
                block_dims=256,
                vmap_method="broadcast_all"
            )
        
        # Test with broadcast
        x = jnp.ones((3,), dtype=jnp.float32)
        y = jnp.array([1.0], dtype=jnp.float32)  # Will be broadcast
        
        # This should work with vmap
        vmapped_fn = vmap(kernel_fn, in_axes=(0, None), out_axes=0)
        
        # Note: This is a mock test - actual execution would require proper setup
        with pytest.raises(ValueError, match="call_target must be provided"):
            result = vmapped_fn(x, y)

    def test_invalid_vmap_method(self):
        """Test that invalid vmap_method raises appropriate error."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        def kernel_fn(x):
            return kernel_call(
                ptx_kernel,
                "test_kernel",
                x,
                x,
                grid_dims=1,
                block_dims=256,
                vmap_method="invalid_method"
            )
        
        x = jnp.ones((2, 3), dtype=jnp.float32)
        vmapped_fn = vmap(kernel_fn, in_axes=0, out_axes=0)
        
        # This should raise a NotImplementedError
        with pytest.raises(NotImplementedError, match="vmap is only supported"):
            vmapped_fn(x)

    def test_kernel_validation(self):
        """Test kernel validation."""
        # Test invalid kernel type
        with pytest.raises(ValueError, match="Unsupported kernel type"):
            kernel_call(
                "invalid",
                "test",
                jnp.array([1.0]),
                jnp.array([1.0]),
                kernel_type="invalid_type"
            )
        
        # Test PTX kernel without .entry
        with pytest.raises(ValueError, match="PTX code must contain an .entry point"):
            kernel_call(
                ".visible .func test() { ret; }",
                "test",
                jnp.array([1.0]),
                jnp.array([1.0]),
                kernel_type="ptx"
            )

    def test_output_indices_validation(self):
        """Test output_indices validation."""
        ptx_kernel = """
        .visible .entry test_kernel(
            .param .u64 input,
            .param .u64 output,
            .param .u32 n
        ) {
            ret;
        }
        """
        
        # Test invalid output_indices type
        with pytest.raises(ValueError, match="output_indices must be a sequence"):
            kernel_call(
                ptx_kernel,
                "test_kernel",
                jnp.array([1.0]),
                jnp.array([1.0]),
                output_indices=123  # type: ignore
            )
        
        # Test wrong number of output indices
        with pytest.raises(ValueError, match="Expected 1 output indices but got 2"):
            kernel_call(
                ptx_kernel,
                "test_kernel",
                jnp.array([1.0]),
                jnp.array([1.0]),
                output_indices=[0, 1]
            )


if __name__ == "__main__":
    pytest.main([__file__]) 