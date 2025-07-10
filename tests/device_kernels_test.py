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
from jax._src.device_kernels import kernel_call


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
                output_indices=123  # Invalid type (not a sequence)
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