import torch

M, N, K = 1, 2048, 2048

A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
B = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

with torch.autograd.profiler.emit_nvtx():
    out = torch.nn.functional.linear(A, B)  # slow on B200!
