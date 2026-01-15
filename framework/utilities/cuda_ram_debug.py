"""
Utilities for debugging CUDA RAM usage in PyTorch.
"""

import gc, torch, sys, types, inspect

def log_cuda_memory(message=""):
    if not torch.cuda.is_available():
        return
    print(f"[CUDA MEMORY] {message}")
    print(f"  Peak Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"  Cached:    {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

def find_cuda_holders(limit=50):
    cuda_objs = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                cuda_objs.append(obj)
        except Exception:
            continue
    # Sort by size (numel)
    cuda_objs.sort(key=lambda t: t.numel() if hasattr(t, "numel") else 0, reverse=True)
    out = []
    for t in cuda_objs[:limit]:
        try:
            out.append({
                "type": type(t).__name__,
                "shape": tuple(t.shape),
                "dtype": str(t.dtype),
                "device": str(t.device),
                "numel": t.numel(),
                "bytes": t.element_size() * t.numel()
            })
        except Exception:
            pass
    return out, len(cuda_objs)

def debug_cuda_holders():
    summary, count = find_cuda_holders(100)
    torch.cuda.reset_peak_memory_stats()
    log_cuda_memory("After resetting peak memory stats")
    print(f"CUDA tensors found: {count}")
    for s in summary:
        print(s)

    # If you want to inspect referrers to the largest tensor:
    if count>0:
        import gc
        largest = None
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    if largest is None or obj.numel() > largest.numel():
                        largest = obj
            except Exception:
                continue
        if largest is not None:
            print("\nlargest tensor info:", largest.shape, largest.dtype, largest.device)
            refs = gc.get_referrers(largest)
            print("Number of referrers for largest tensor:", len(refs))
            for i, r in enumerate(refs[:20], 1):
                print(f"Referrer {i}: type={type(r)}")
                # optionally print a short repr (be careful)
                try:
                    print(" repr:", repr(r)[:200])
                except Exception:
                    pass

