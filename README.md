# Minimal Transformer LLM: From Scratch to Optimization

This project implements a minimal decoder-only transformer model from scratch in PyTorch and progressively applies performance optimizations across training and compilation stages, e.g. compilation, AMP...

The goal is to deeply understand how LLMs function internally and how to optimize them at different levels of the PyTorch stack.

---

## 📁 Project Structure

### Step 1: Baseline LLM (`LLM_1`)
- Implemented a transformer-based decoder-only model from first principles.
- **Custom components**:
  - Tokenization using Byte Pair Encoding (BPE)
  - Positional encoding
  - Causal multi-head self-attention
  - Decoder block with LayerNorm and feedforward layers
- **Model**: Small GPT-style decoder-only transformer
- **Dataset**: Tiny Shakespeare
- **Training**:
  - Manual training & validation loops
  - Cosine LR scheduler with warm-up
  - Loss: `CrossEntropyLoss`
  - Optimizer: `AdamW`
- **Results**:
  - Validation loss convergence to ~1.95 (character-level BPE tokens)

---

### Step 2: Torch Compile & Optimization (`LLM_opt_1`)
- Refactored code for clarity and modularity
- Integrated `torch.compile()` with multiple backends:
  - `default`, `reduce-overhead`, `max-autotune`
- Explored backend execution:
  - Interpreted autotuner output
  - Observed GEMM kernel strategies (e.g., `addmm`, `bias_addmm`)
- **Training improvements**:
  - Slight speedups (~5%)
  - Maintained validation loss (~1.85)
- **Tools**:
  - TensorBoard for logging
  - TorchDynamo, TorchInductor, NVFuser

---

### Step 3: Mixed Precision + Granular Compilation (`LLM_opt_2`)
- Compiled selected submodules (e.g., attention, decoder layers)
- Compared performance of:
  - Full model compilation
  - Layer-level compilation
- **Profiling**:
  - Used `torch.profiler` (inference)
  - Analyzed `aten::addmm`, `volta_sgemm_*` kernels
  - Understood `Self` vs `Total` time across CPU and CUDA
- **Manual Timing**:
  - Created `Timer` class with `cuda.synchronize()` for accurate CUDA timings
- **Mixed Precision (AMP)**:
  - Integrated `torch.cuda.amp.autocast` + `GradScaler`
  - Achieved **2–3× speedup per epoch**
  - Verified stability and lower memory usage
- **Data I/O Benchmarking**:
  - Measured train loader raw time (~7s out of ~160s per epoch)
  - I/O not a bottleneck (~4%)

---

## Technical Stack

- Python 3.10+
- PyTorch 2.1+
- CUDA-enabled GPU (tested with limited SMs)
- TorchInductor / TorchDynamo (`torch.compile`)
- TensorBoard for metric visualization

---

## Roadmap

This is part of a broader, structured exploration of LLM optimization:

| STEP | Focus Area                                     | Status  |
|-----|------------------------------------------------|---------|
| 1   | Core architecture & training                   |  Done |
| 2   | PyTorch-level optimization (AMP, checkpointing, profiling) | In Progress |
| 3   | Custom kernel development (Triton, NVFuser)    | Planned |
| 4   | Distributed training (DDP, model/pipeline parallelism) | Planned |
| 5   | Inference optimization (quantization, ONNX, TensorRT) | Planned |

---

##  Next Steps

- Implement gradient checkpointing for memory savings
- Profile memory usage and runtime across model variants
- Begin custom kernel prototyping with Triton
- Extend performance metrics (FLOPs, energy usage, SM utilization)

---

## License

This project is open-source and intended for educational and experimental use.
