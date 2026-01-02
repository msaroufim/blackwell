# blackwell

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
