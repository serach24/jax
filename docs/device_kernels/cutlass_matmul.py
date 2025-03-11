import os
import jax
import jax.numpy as jnp
import jax.extend as jex

# # Set print options to display the full array
# jnp.set_printoptions(threshold=jnp.inf)
current_dir = os.path.dirname(os.path.abspath(__file__))
ptx_path = os.path.join(current_dir, "cutlass_gemm_kernel_f32xf32_to_f32.nvptx")

with open(ptx_path, "r") as f:
    ptx_code = f.read()


# Inputs for vector addition
a = jnp.ones(1024, dtype=jnp.float32)
b = jnp.ones(1024, dtype=jnp.float32)

M = 128 
N = 128 
K = 8

A = jax.random.normal(jax.random.PRNGKey(0), (M, K))
B = jax.random.normal(jax.random.PRNGKey(1), (K, N))
# C = jnp.zeros((M, N), dtype=jnp.float32)


# Call the 'add_vectors' PTX kernel
device_kernel_result = jex.device_kernels.ptx_call(
    ptx_code,
    "_ZN7cutlass7Kernel2INS_4gemm6kernel20DefaultGemmUniversalIfNS_6layout8RowMajorELNS_16ComplexTransformE0ELi1EfS5_LS6_0ELi1EfS5_fNS_4arch11OpClassSimtENS7_4Sm70ENS1_9GemmShapeILi128ELi128ELi8EEENSA_ILi32ELi64ELi8EEENSA_ILi1ELi1ELi1EEENS_8epilogue6thread17LinearCombinationIfLi1EffLNSF_9ScaleType4KindE0ELNS_15FloatRoundStyleE2EfEENS1_11threadblock30GemmIdentityThreadblockSwizzleILi1EEELi2ENS7_13OpMultiplyAddELNS1_23SharedMemoryClearOptionE0ELb0ELb0ELb0ENS4_9NoPermuteESQ_SQ_vE10SelectBaseISN_vEEEEvNT_6ParamsE",
    jax.ShapeDtypeStruct((M, N), jnp.float32),  # Output shape and dtype
    M, N, K, A, B,
    grid_dims=(1, 1, 1),                        # Grid dimensions
    block_dims=(256, 1, 1),                     # Thread block dimensions
    shared_mem_bytes=8448,                      # Shared memory size
)
# compare with jax gemm
jax_result = jax.numpy.matmul(A, B)
assert jnp.allclose(device_kernel_result, jax_result)

print(device_kernel_result)