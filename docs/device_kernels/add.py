import jax
import jax.numpy as jnp
import jax.extend as jex

# PTX code with multiple kernels
ptx_code = """
.extern .global .align 4 .b8 out[];

.entry add_vectors(
    .param .u64 in1, .param .u64 in2, .param .u64 out
) {
    .reg .s32 r<3>;
    .reg .u64 p<3>;
    
    ld.param.u64 p1, [in1];
    ld.param.u64 p2, [in2];
    ld.param.u64 p3, [out];
    
    mov.u32 r0, %tid.x;

    ld.global.f32 r1, [p1 + r0 * 4];
    ld.global.f32 r2, [p2 + r0 * 4];
    add.f32 r3, r1, r2;

    st.global.f32 [p3 + r0 * 4], r3;
}
"""

# Inputs for vector addition
a = jnp.ones(1024, dtype=jnp.float32)
b = jnp.ones(1024, dtype=jnp.float32)

# Call the 'add_vectors' PTX kernel
# result_add = jax.device_kernels.call(
result_add = jex.device_kernels.ptx_call(
    ptx_code,
    "add_vectors",          # Specify the kernel to invoke
    jax.ShapeDtypeStruct(a.shape, a.dtype), # Output shape and dtype
    a,
    b
    # grid_dims=(4, 1, 1),                # 4 blocks
    # block_dims=(256, 1, 1),             # 256 threads per block
    # shared_mem_bytes=0,                 # No shared memory
    # output_shapes=((1024,),),           # Output shape is a vector of 1024 elements
)
