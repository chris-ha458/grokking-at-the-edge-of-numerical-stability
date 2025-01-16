from utils import s, log_stablemax
import torch
import time
from typing import Tuple, List
import pandas as pd

COMPUTE_DEVICE = 'cpu'
if torch.cuda.is_available():
    COMPUTE_DEVICE = 'cuda'
elif torch.mps.is_available():
    COMPUTE_DEVICE = 'mps' 
print(f'Using {COMPUTE_DEVICE} for computation')

def softmax(x, dim=-1):
    return torch.nn.functional.softmax(x, dim=dim)

def stablemax(x, dim=-1):
    s_x = s(x)
    return s_x/torch.sum(s_x, dim=dim, keepdim=True)

def time_function(func, tensor: torch.Tensor, iterations: int = 100) -> float:
    start_time = time.perf_counter()
    for _ in range(iterations):
        _result = func(tensor) # result not used
        if COMPUTE_DEVICE == 'cuda':
            torch.cuda.synchronize()  # Ensure GPU operations complete
        elif COMPUTE_DEVICE == 'mps':
            torch.mps.synchronize()
    end_time = time.perf_counter()
    return (end_time - start_time) / iterations

def run_comparison():
    sizes = [
        (1, 2**10),    # 1K
        (1, 2**11),    # 2K
        (1, 2**12),    # 4K
        (1, 2**13),    # 8K
        (1, 2**14),   # 16K
        (1, 2**15),   # 32K
        (1, 2**16),   # 64K
        (1, 2**17),   # 128K
        (1, 2**18),   # 256K
        (1, 2**19),   # 512K
        (1, 2**20),   # 1M
    ]
    dtypes = [torch.float32, torch.bfloat16]
    dtype_names = ['fp32', 'bf16'] # fp16 is basically the same as bf16
    results = []

    for size in sizes:
        for dtype, dtype_name in zip(dtypes, dtype_names):
            tensor = torch.randn(size, dtype=dtype).to(COMPUTE_DEVICE)
            
            stablemax_time = time_function(lambda x: stablemax(x, dim=-1), tensor)
            softmax_time = time_function(lambda x: softmax(x, dim=-1), tensor)
            
            results.append({
                'Size': f'1x{size[1]}',
                'Precision': dtype_name,
                'StableMax (ms)': stablemax_time * 1000,
                'SoftMax (ms)': softmax_time * 1000,
                'Stable/Soft ratio': stablemax_time / softmax_time
            })

    df = pd.DataFrame(results)
    print("\nSpeed Test Results:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_comparison()