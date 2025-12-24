# FlashAttention (CUDA)

This repository contains from-scratch CUDA C++ implementations of FlashAttention, developed incrementally to understand both the algorithmic and systems-level optimizations behind modern attention kernels.

The code progresses from a basic FlashAttention v1 implementation (streaming softmax without materializing the attention matrix) to more optimized variants inspired by FlashAttention v2 and v3, incorporating advanced GPU programming techniques.

## What this repo covers

- Scaled dot-product attention with streaming, numerically stable softmax
- FlashAttention v1: tiled Q/K/V streaming without storing QKᵀ
- Progressive kernel optimizations:
  - Improved shared memory layouts
  - Shared memory swizzling
  - Multi-stage pipelining
  - Vectorized loads and ldmatrix usage
  - Better overlap of memory and compute
- Performance comparisons against PyTorch and flash-attn implementations

## Goals

- Build a deep understanding of FlashAttention from first principles
- Explore how modern CUDA kernels approach near–tensor-core peak performance
- Serve as a learning resource for CUDA, GPU memory hierarchies, and attention kernels

## Status

- Forward pass only
- Focused on BF16/FP16
- Single-GPU, single-node kernels

## References

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- DAO-AILab/flash-attention
- NVIDIA CUDA Programming Guide
- CUTLASS examples

This repository is intended for educational and experimental purposes.
