import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import profile, ProfilerActivity

M, N, K = 1, 2048, 2048
A = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
B = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")

# Make the region long enough to measure
iters = 50000

# 1) Time (steady-state) with CUDA events
for _ in range(100):  # warmup
    torch.nn.functional.linear(A, B)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    out = torch.nn.functional.linear(A, B)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)
time_s = ms / 1e3

# 2) Collect HBM bytes via torch.profiler CUPTI counters
# (These metric names are the same ones used in Kineto/HTA examples.)
exp = _ExperimentalConfig(
    profiler_metrics=["dram__bytes_read.sum", "dram__bytes_write.sum"],
    profiler_measure_per_kernel=True,
)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    experimental_config=exp,
) as prof:
    for _ in range(iters):
        out = torch.nn.functional.linear(A, B)
        prof.step()

# This writes a Chrome trace containing counter events (cuda_profiler_range)
prof.export_chrome_trace("trace.json")

print(f"timed {iters} iters: {ms/iters:.6f} ms/iter")
print("wrote trace.json with dram__bytes_read.sum / dram__bytes_write.sum")
