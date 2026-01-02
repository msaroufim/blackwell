# blackwell

This repo is a scratchpad and collection of misc stuff, don't expect to learn anything skimming this

# profiling notes

https://github.com/chenyu-jiang/nsys2json
https://github.com/ezyang/nvprof2json
Trace event format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw



dram__read_throughput.avg.pct_of_peak_sustained_elapsed
dram__write_throughput.avg.pct_of_peak_sustained_elapsed



```
nsys profile -t cuda,nvtx,osrt \
  --gpu-metrics-devices=0 \
  --gpu-metrics-set=gb10x \
  --gpu-metrics-frequency=20000 \
  -o bw \
  python3 test.py
```

Generates a -rep file and can get bw numbers using `nsys stats bw.nsys-rep --report gpu_metric_sum`

You can collect experimental metrics in pytorch using- the full pattern is in `cup.py`

```
exp = _ExperimentalConfig(
    profiler_metrics=["dram__bytes_read.sum", "dram__bytes_write.sum"],
    profiler_measure_per_kernel=True,
)
```
