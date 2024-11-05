import jax
import jax.numpy as jnp
import jax.extend as jex

# PTX code with multiple kernels
# ptx_code = """
# .version 7.0
# .target sm_70
# .address_size 64

# .extern .global .align 4 .b8 out[];

# .entry add_vectors(
#     .param .u64 in1, .param .u64 in2, .param .u64 out
# ) {
#     .reg .s32 r<3>;
#     .reg .u64 p<3>;
    
#     ld.param.u64 p1, [in1];
#     ld.param.u64 p2, [in2];
#     ld.param.u64 p3, [out];
    
#     mov.u32 r0, %tid.x;

#     ld.global.f32 r1, [p1 + r0 * 4];
#     ld.global.f32 r2, [p2 + r0 * 4];
#     add.f32 r3, r1, r2;

#     st.global.f32 [p3 + r0 * 4], r3;
# }
# """

ptx_code = """
.version 8.5
.target sm_90
.address_size 64

.entry add_vectors(
    .param .u64 in1,
    .param .u64 in2,
    .param .u64 out
) {
    .reg .s32 r0;
    .reg .u64 p1, p2, p3, p4;
    .reg .f32 r1, r2, r3;

    // Load parameters into registers
    ld.param.u64 p1, [in1];
    ld.param.u64 p2, [in2];
    ld.param.u64 p3, [out];

    // Calculate offset in bytes using thread index
    mov.u32 r0, %tid.x;
    mul.wide.s32 p4, r0, 4;  // p4 now holds the byte offset as a u64

    // Calculate final addresses for in1, in2, and out
    add.u64 p1, p1, p4;
    add.u64 p2, p2, p4;
    add.u64 p3, p3, p4;

    // Load float values from the computed addresses, perform addition, and store result
    ld.global.f32 r1, [p1];
    ld.global.f32 r2, [p2];
    add.f32 r3, r1, r2;
    st.global.f32 [p3], r3;
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
