import os
# Ensure Triton dumps all IR/assembly artifacts.
os.environ["TRITON_KERNEL_DUMP"] = "1"

import torch
import triton
import triton.language as tl
import pprint

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each kernel instance (program) processes BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y
    tl.store(out_ptr + offsets, result, mask=mask)

def main():
    n = 1024
    BLOCK_SIZE = 1024

    # Create example tensors on the GPU.
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    
    # Compute grid: one kernel instance handles BLOCK_SIZE elements.
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    # Launch the kernel (triggers JIT compilation and caching).
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Access and print the full kernel cache.
    cache = add_kernel.cache
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
                    with open("add_kernel.ptx", "w") as f:
                        f.write(ptx_code)
                    print("PTX code written to add_kernel.ptx")
                else:
                    print("No 'ptx' key found in asm dictionary.")
            else:
                print("Compiled kernel object does not have an 'asm' attribute.")
            print("-" * 40)

if __name__ == "__main__":
    main()

