# FlashAttention (CUDA, From Scratch)

This repository contains a from-scratch CUDA C++ implementation of FlashAttention, built incrementally to understand both the algorithmic foundations and low-level GPU optimizations behind modern attention kernels.

The project follows a structured, kernel-by-kernel approach inspired by FlashAttention v2, where an initial implementation is progressively optimized across multiple iterations — focusing on memory movement, tensor core utilization, and kernel efficiency.

By the final stages, the goal is to approach (and analyze) near state-of-the-art performance on Ampere GPUs.

---

## What This Repo Covers

- Scaled dot-product attention with streaming, numerically stable softmax
- Tiled FlashAttention (no materialization of QKᵀ)
- Incremental kernel optimization pipeline:
  - Efficient GMEM → SMEM → RF data movement (`cp.async`, `ldmatrix`)
  - Tensor Core programming via `mma.sync`
  - Shared memory layout design and swizzling
  - Multi-stage pipelining and overlap of memory + compute
  - Vectorized memory operations
  - Warp-level coordination and synchronization

---

## Kernel Specification (Current Scope)

To keep the implementation focused and tractable:

- Forward pass only
- Non-causal attention
- Head dimension = 128
- No dropout or KV caching
- Equal sequence lengths for Q, K, V
- Sequence lengths divisible by tile sizes (typically 64–128)
- FP16 / BF16 inputs and outputs
- Softmax computed in FP32 (numerical stability)

---

## Project Structure

The codebase is modular and mirrors the structure of modern CUDA kernels:

```text
flash-attention/
├── include/
│   ├── flash_attention.cuh
│   ├── forward_kernel.cuh
│   ├── load_store.cuh
│   ├── gemm.cuh
│   ├── softmax.cuh
│   └── tensor.cuh
├── src/
│   └── main.cu
└── README.md
```



---

## Goals

- Build FlashAttention entirely from first principles (no external libraries)
- Develop a deep understanding of GPU memory hierarchy and data movement
- Explore how high-performance kernels approach tensor-core peak throughput
- Serve as a practical reference for CUDA kernel design and optimization

---

## Progression

The implementation evolves across multiple kernel iterations, each introducing new optimizations:

1. Baseline tiled attention kernel
2. Improved memory layouts
3. Bank conflict reduction and swizzling
4. Tensor core optimization (mma usage)
5. Pipeline parallelism (overlapping compute and memory)
6. Instruction-level and layout optimizations
7. Profiling and performance tuning

---

## Status

- Forward pass implementation in progress
- Focused on Ampere GPUs (e.g., RTX 3090, A100)
- Single-GPU kernels
- Performance optimization ongoing

---

## References

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention v2 (DAO-AILab)
- NVIDIA CUDA Programming Guide
- CUTLASS

---

## Future Work

Planned extensions:

- Backward pass (dQ, dK, dV)
- Causal masking and general attention masks
- KV caching (inference optimization)
- Hopper optimizations:
  - TMA (Tensor Memory Accelerator)
  - Async WGMMA instructions
- Blackwell architecture exploration (e.g., tcgen05)
- Auto-tuning and kernel configuration search

---

## Notes

This repository is intended for educational and experimental purposes, with a strong emphasis on understanding *why* each optimization works — not just implementing it.
