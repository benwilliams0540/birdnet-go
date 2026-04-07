# Uno Q GPU Runtime Spikes

## Scope

This document captures the April 7, 2026 exploratory work around GPU-oriented
BirdNET runtimes on the Arduino Uno Q. It is intentionally lightweight so the
repo keeps the findings without committing large generated model artifacts.

## NCNN custom frontend layer

We implemented an experimental `BirdNETFrontend` NCNN custom layer that accepts
raw audio and emits the validated split CNN tensor shape.

Measured on the Uno Q with the `benchmark` command:

- Split NCNN CPU: about `1012 ms`
- Split NCNN Vulkan: about `1331 ms`
- Custom-layer NCNN CPU: about `4188 ms`
- Custom-layer NCNN Vulkan: about `4583 ms`

Conclusion:

- The first custom-layer implementation was much slower than the existing split
  path.
- The bottleneck was the CPU-only frontend implementation itself, not NCNN's
  CNN tail.
- Moving the frontend into a naive custom layer does not make the workload GPU
  accelerated.

## MNN spike

We also evaluated MNN as an alternative runtime because the Uno Q exposes both
Vulkan and OpenCL.

What worked:

- The trusted `source_models/BirdNET_V2.4.onnx` model can be rewritten and
  split so that MNN conversion succeeds.
- The frontend rewrite matched ONNX Runtime closely through the mel/STFT stages.

What did not work:

- Full-graph import still stalled on BirdNET frontend operators such as `DFT`.
- The converted CNN tail produced incorrect class rankings even when fed the
  correct intermediate tensor.
- OpenCL fell back to CPU on the Uno Q, and Vulkan only engaged in mixed mode.

Observed Uno Q timings:

- Full-graph MNN CPU: about `364-520 ms`
- Full-graph MNN Vulkan mixed backend: about `535 ms`

Conclusion:

- MNN was not a deployable BirdNET GPU path for the Uno Q in this investigation.
- The practical value of the spike was in confirming that BirdNET frontend work
  dominates latency and that a split-tail GPU backend alone is unlikely to win.

## Takeaways

- TFLite remained the practical production baseline.
- The only plausible acceleration route left in this spike was a much more
  optimized NCNN frontend path, likely with aggressive CPU optimization first
  and a GPU implementation only if that proved worthwhile.
- Large generated artifacts from these experiments were intentionally left out
  of git and should be regenerated from source models if needed.
