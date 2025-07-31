# Minimal Transformer LLM: From Scratch to Optimization

This project implements a minimal decoder-only transformer model from scratch in PyTorch and progressively applies performance optimizations across training and compilation stages.

The goal is to deeply understand how LLMs work internally and how to optimize them at different levels of the PyTorch stack.

## Project Structure
Step 1: Baseline LLM (LLM_1)
Implemented a transformer-based decoder-only model from first principles.

Custom components:

Tokenization using Byte Pair Encoding (BPE)

Positional encoding

Causal multi-head self-attention

Decoder block with layer normalization and feedforward layers

Model: GPT-style architecture, small scale

Dataset: Tiny Shakespeare

Training:

Manual training and validation loops

Cosine LR scheduler with warm-up

Loss function: CrossEntropyLoss

Optimizer: AdamW

Result:

Validation loss convergence to ~1.95 on character-level BPE tokens

Step 2: Torch Compile & Optimization (LLM_opt_1)
Refactored code for modularity and clarity

Integrated PyTorch 2.0 torch.compile() to accelerate training

Tested multiple compilation modes:

default

reduce-overhead

max-autotune

Investigated kernel-level decisions using Inductor and NVFuser:

Interpreted autotuner logs

Observed addmm vs bias_addmm choices across shapes

Training improvements:

Slight reduction in runtime

Comparable validation loss to non-compiled version (~1.85)

Logged training metrics and hyperparameters using TensorBoard

Technical Stack
Python 3.10+

PyTorch 2.1+

CUDA-enabled GPU (tested with limited SMs)

TensorBoard for monitoring

TorchInductor / TorchDynamo backend via torch.compile

Roadmap
This is part of a broader multi-month exploration of LLM training optimization:

MES 1: Core architecture & training ✅

MES 2: PyTorch-level optimization (AMP, checkpointing, profiling) 🔜

MES 3: Custom kernel development with Triton / NVFuser

MES 4: Distributed training (DDP, model parallelism)

MES 5: Inference optimization (quantization, TensorRT, ONNX export)

Next Steps
Integrate Automatic Mixed Precision (torch.cuda.amp)

Apply gradient checkpointing for memory savings

Benchmark training time and memory across all modes

Begin custom kernel prototyping with Triton

License
This project is open-source and intended for educational and experimental use.