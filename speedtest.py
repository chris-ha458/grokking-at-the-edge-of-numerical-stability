from utils import s
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

def create_test_tensor(size: Tuple[int, ...], dtype=torch.float32) -> torch.Tensor:
    return torch.randn(size, dtype=dtype)

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
        (128, 128),    # Small
        (512, 512),    # Medium
        (2048, 2048),  # Large
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    dtype_names = ['fp32', 'fp16', 'bf16']
    results = []

    for size in sizes:
        for dtype, dtype_name in zip(dtypes, dtype_names):
            tensor = create_test_tensor(size, dtype).to(COMPUTE_DEVICE)
            
            s_time = time_function(s, tensor)
            exp_time = time_function(torch.exp, tensor)
            
            results.append({
                'Size': f'{size[0]}x{size[1]}',
                'Precision': dtype_name,
                'S_time (ms)': s_time * 1000,
                'Exp_time (ms)': exp_time * 1000,
                'S/Exp ratio': s_time / exp_time
            })

    df = pd.DataFrame(results)
    print("\nSpeed Test Results:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_comparison()