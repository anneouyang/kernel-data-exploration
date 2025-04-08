import os
# Ensure Triton dumps all IR/assembly artifacts.
os.environ["TRITON_KERNEL_DUMP"] = "1"

import torch
import triton
import triton.language as tl
import pprint

import torch

def matmul_kernel(a, b, c):
    # Use torch's matmul operation
    c.copy_(torch.matmul(a, b))

def main():
    M, N, K = 256, 256, 256
    # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 16, 16, 16

    # Create example tensors on the GPU.
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    
    # Compute grid: one kernel instance handles a block of the output matrix.
    # grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    # Launch the kernel (triggers JIT compilation and caching).
    # matmul_kernel[grid](a, b, c, M, N, K, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K)
    matmul_kernel(a, b, c)
    print("Result of matmul: ", c)

    # Access and print the full kernel cache.
    cache = matmul_kernel.cache
    print("Full kernel cache:")
    pprint.pprint(cache)
    
    # The cache is a dict mapping outer keys to inner dicts.
    for outer_key, inner_dict in cache.items():
        print(f"Outer key: {outer_key}")
        for sig, compiled_kernel in inner_dict.items():
            print("Signature:", sig)
            print("Compiled kernel object type:", type(compiled_kernel))
            # List available attributes using dir()
            attrs = dir(compiled_kernel)
            print("Attributes:", attrs)
            # If the object has an 'asm' attribute, try to get the PTX code.
            if hasattr(compiled_kernel, "asm"):
                asm_dict = getattr(compiled_kernel, "asm")
                print("Assembly dictionary keys:", asm_dict.keys())
                ptx_code = asm_dict.get("ptx")
                if ptx_code:
                    print("PTX code:")
                    print(ptx_code)
                    with open("matmul_kernel.ptx", "w") as f:
                        f.write(ptx_code)
                    print("PTX code written to matmul_kernel.ptx")
                else:
                    print("No 'ptx' key found in asm dictionary.")
            else:
                print("Compiled kernel object does not have an 'asm' attribute.")
            print("-" * 40)

if __name__ == "__main__":
    main()
