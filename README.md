# EEmicroGPT

> *"This file is the complete algorithm. Everything else is just efficiency."*
> — Andrej Karpathy, [microgpt.py](https://karpathy.github.io/2026/02/12/microgpt/)

**This file is the everything else.**

[EEmicroGPT](https://github.com/Entrpi/eemicrogpt/blob/master/eemicrogpt.c) is a single-file, dependency-free C implementation of GPT training — forward pass, backward pass, Adam optimizer, and autoregressive generation — optimized from the ground up for Apple Silicon. It trains a character-level name generator on the same architecture and dataset as Karpathy's microgpt.py, producing identical learning dynamics **up to 19,000x faster** per training sample.

The name stands for "Everything Else" (or "Extreme Efficiency") — the half of the equation that microgpt.py intentionally leaves on the table.

## Quick start

```bash
# Scalar Neon path (any Apple Silicon Mac)
clang -O3 -ffast-math -o eemicrogpt eemicrogpt.c -lm
./eemicrogpt

# SME2 path (M4/M5+, ~2x faster at d_model>=64)
clang -O3 -mcpu=native+sme2 -ffast-math -o eemicrogpt eemicrogpt.c -lm
./eemicrogpt
```

Requires `names.txt` in the working directory (the [Karpathy names dataset](https://raw.githubusercontent.com/karpathy/makemore/master/names.txt), ~32K names).

Configurable at compile time:
```bash
# Smaller/faster model
clang -O3 -ffast-math -DD_MODEL=16 -DN_STEPS=5000 -DLR_INIT=0.008 -o eemicrogpt eemicrogpt.c -lm

# Larger model with SME2
clang -O3 -mcpu=native+sme2 -ffast-math -DD_MODEL=128 -DN_HEADS=8 -DN_STEPS=10000 -DLR_INIT=0.003 -o eemicrogpt eemicrogpt.c -lm
```

## Architecture

A 1-layer GPT matching Karpathy's microgpt exactly:

| Component | Details |
|---|---|
| Layers | 1 transformer block |
| d_model | 64 (configurable: 16, 32, 64, 128) |
| Heads | 4 (configurable) |
| d_ff | 4 * d_model |
| Vocab | 27 (a-z + boundary token) |
| Max seq | 16 |
| Norm | RMS norm (pre-attention, pre-FFN) |
| Activation | ReLU |
| Optimizer | Adam (beta1=0.85, beta2=0.99) |
| LR schedule | Linear decay to zero |

The forward pass: `embed → rms_norm → rms_norm → QKV → causal attention → O proj + residual → rms_norm → FFN (expand → ReLU → contract) + residual → LM head → softmax → cross-entropy loss`

## Performance

All benchmarks on Apple M5, single P-core.

### bpc@1s: best quality within ~1 second of training (batch=16)

| d_model | backend | us/step | steps/1s | loss |
|---------|---------|---------|----------|------|
| 16 | scalar | 57.1 | 16,700 | 2.0869 |
| 32 | scalar | 181 | 5,150 | **2.0747** |
| 64 | scalar | 832 | 1,100 | 2.1384 |
| 64 | SME2 | 589 | 1,700 | 2.0974 |
| 128 | scalar | 4,779 | 220 | 2.2633 |
| 128 | SME2 | 1,904 | 545 | 2.1645 |

**Winner: d32 scalar at LR=0.007 → loss 2.0747**

d32 hits the sweet spot: enough capacity to learn well, small enough to run thousands of steps per second. At d16 the model is capacity-limited; at d64+ the per-step cost eats into the time budget. The fused streaming-mode backward (optimization #10) gives the d64 SME2 path 200 extra steps, closing the gap with d32.

### Convergence reference (long runs, tuned LR)

| d_model | steps | LR | loss |
|---------|-------|-----|------|
| 16 | 100k | 0.006 | ~2.06 (capacity floor) |
| 32 | 500k | 0.002 | 1.92 |
| 64 (SME2) | 1M | 0.0007 | 1.74 (~10 min wall time) |

### Work-equivalent comparison (per training sample, same architecture)

All implementations train the exact same model on the same data. CPython, PyPy, and [microgpt.cpp](https://github.com/Charbel199/microgpt.cpp) use the autograd `Value` class approach with batch=1 and f64. [rust-microgpt](https://github.com/mplekh/rust-microgpt) uses the same autograd approach with batch=1 but f32. EEmicroGPT uses explicit forward/backward, batch=16, f32.

**d16 (n_embd=16, block_size=16, 10K training samples):**

| Implementation | Wall time | us/sample | Speedup |
|---|---|---|---|
| CPython 3.14 | 490s | 49,000 | 1x |
| PyPy 7.3.17 | 176.4s | 17,640 | 2.8x |
| microgpt.cpp | 2.70s | 270 | 181x |
| rust-microgpt | 1.18s | 118 | 415x |
| **EEmicroGPT** | **0.030s** | **3.0** | **16,333x** |

**d64 (n_embd=64, block_size=16, 1K training samples):**

| Implementation | Wall time | us/sample | Speedup |
|---|---|---|---|
| CPython 3.14 | 713s | 713,200 | 1x |
| PyPy 7.3.17 | 301.4s | 301,400 | 2.4x |
| microgpt.cpp | 3.26s | 3,260 | 219x |
| rust-microgpt | 1.62s | 1,620 | 440x |
| EEmicroGPT scalar | 0.047s | 46.8 | 15,239x |
| **EEmicroGPT SME2** | **0.037s** | **36.8** | **19,380x** |

~44x faster than rust-microgpt (the fastest autograd implementation) at both sizes. The gap comes from batched GEMM (16 samples amortize weight loads), float vs double, explicit gradients (no autograd graph), and Neon/SME2 SIMD.

## Why this likely beats a $40K GPU

This isn't just faster than Python — a single M5 P-core likely beats an NVIDIA B200 Blackwell for this workload. Here's the math.

### The compute is trivial

Total FLOPs per training step (forward + backward + Adam), with batch=16 and ~104 active positions:

| d_model | FLOPs/step | params | weight memory |
|---------|-----------|--------|---------------|
| 16 | 2.3M | 4,192 | 16 KB |
| 32 | 8.5M | 14,528 | 57 KB |
| 64 | 32.5M | 53,632 | 210 KB |
| 128 | 127M | 205,568 | 803 KB |

The B200 delivers [90 TFLOPS FP32](https://www.rightnowai.co/guides/gpu-comparison/b200). At peak throughput, these steps would take:

| d_model | B200 @ 100% | B200 @ 5% | EEmicroGPT (M5) |
|---------|-------------|-----------|-----------------|
| 16 | 0.026 us | 0.5 us | **47 us** |
| 32 | 0.094 us | 1.9 us | **182 us** |
| 64 | 0.36 us | 7.2 us | **589 us** |
| 128 | 1.4 us | 28 us | **1,904 us** |

If the B200 could sustain even 5% utilization, it would crush us. It can't. The problem isn't compute — it's everything around the compute.

### The killer microsecond problem

A training step requires ~40 distinct GPU kernel launches: 7 forward GEMMs, 6 backward input-grad GEMMs, 7 weight-grad GEMMs, plus ~20 non-GEMM kernels (embeddings, 6 RMS norms, attention, ReLU, residuals, softmax, loss, Adam).

Each kernel launch has overhead. [Measured CUDA launch latency](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/) is ~10–20 us host-to-device; back-to-back kernel-to-kernel gaps are ~3–5 us even with CUDA graphs. This is what NVIDIA calls the ["killer microsecond"](https://github.com/zartbot/blog/issues/3) problem — Blackwell executes small kernels faster than the overhead of launching them.

| Scenario | Kernel overhead | Compute | Total | vs M5 d32 |
|----------|---------------|---------|-------|-----------|
| CUDA graphs (best case) | 40 × 3 us = 120 us | ~10 us | ~130 us | 0.7x |
| Async launch (typical) | 40 × 10 us = 400 us | ~10 us | ~410 us | 2.3x slower |
| PyTorch (framework tax) | 40 × 15 us = 600 us | ~10 us | ~610 us | 3.4x slower |

At d32, even the most optimistic GPU scenario (CUDA graphs with custom CUDA kernels) is roughly tied with a single M5 P-core. In practice, nobody writes custom CUDA training loops with CUDA graphs for a 57 KB model — they use PyTorch, which adds ~15 us per operation for Python dispatch, autograd graph construction, and memory management.

### GPU utilization is <5% for these matrices

Our largest GEMM is FFN expand at d128: `[512, 128] @ [128, 104]` — that's 53K output elements. cuBLAS tiles this into 128×128 thread blocks: `ceil(512/128) × ceil(104/128) = 4 × 1 = 4` thread blocks. The B200 has [148 SMs](https://arxiv.org/html/2512.02189v1). Four of them do useful work; 144 sit idle.

At d32, FFN expand is `[128, 32] @ [32, 104]` — one or two thread blocks on 148 SMs. Under 1.5% utilization. The GPU's 20,480 CUDA cores and 8 TB/s HBM3e bandwidth are designed for matrices with M, N, K > 1024. Our matrices are 100x too small.

### CUDA graphs don't fully help

CUDA graphs pre-record a kernel sequence to eliminate host-side launch overhead. But they require static shapes — every tensor dimension must be fixed at graph capture time. Our variable-length sequences (names average ~6.5 chars but range from 2 to 16) break this: we'd need to pad everything to MAX_SEQ=16, throwing away the 41% savings from padding skip, or capture multiple graphs for different lengths.

### The cache advantage

On the M5, our entire training state fits close to the compute:

| d_model | Weights | L1 hit? | Total state | L2 hit? |
|---------|---------|---------|-------------|---------|
| 16 | 16 KB | yes (128 KB L1) | ~180 KB | yes (32 MB L2) |
| 32 | 57 KB | yes | ~630 KB | yes |
| 64 | 210 KB | partial | ~2.3 MB | yes |
| 128 | 803 KB | no | ~8.8 MB | yes |

The M5 L1 serves data at 3-cycle latency (~1.5 ns). The B200's L1 is [39 cycles / ~20 ns](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut) per SM, and its [L2 is ~150 ns](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut). More importantly, GPU data in L2 doesn't persist across kernel boundaries — each of the 40 kernels reloads from HBM or L2 cache. CPU data stays in cache across the entire training step because there's no kernel boundary, no context switch, no scheduler — just one thread running straight-line code.

### Energy: 67x more efficient

The B200 draws [1000W TDP](https://www.rightnowai.co/guides/gpu-comparison/b200). The M5 MacBook runs this workload at ~15W.

| d_model | M5 time | M5 energy | B200 time (best) | B200 energy | M5 efficiency |
|---------|---------|-----------|-------------------|-------------|---------------|
| 32 | 182 us | 2.7 mJ | ~130 us | 130 mJ | **48x** |
| 64 | 589 us | 8.8 mJ | ~140 us | 140 mJ | **16x** |
| 128 | 1,904 us | 29 mJ | ~170 us | 170 mJ | **5.9x** |

Even at d128 where the B200 wins on wall time, the M5 uses 6x less energy per training step. The GPU pays 1000W whether its 148 SMs are busy or idle.

### Where GPU wins

The crossover is real — and lower than these theoretical estimates suggest. Our [measurements on the same M5](#measured-cpu-vs-gpu-on-m5) show the crossover at d256 with batch=16, and as low as d64 with batch=128. A B200 would crossover even earlier.

Still, for a 1-layer character model with d_model <= 128 and batch=16, a single CPU core with L1-resident weights and zero-overhead dispatch is hard to beat at any price — especially a core with native matrix operations.

## Measured: CPU vs GPU on M5

Theory is nice. Here are actual numbers: EEmicroGPT (1 CPU core, batch=16) vs [MLX](https://github.com/ml-explore/mlx) (Apple's ML framework on the M5's 10-core GPU), both on the same machine.

### Per-sample time at equal batch size (batch=16)

| d_model | EEmicroGPT | MLX GPU | Faster |
|---------|-----------|---------|--------|
| 16 | 3.0 us | 72 us | EE 24x |
| 32 | 11.4 us | 72 us | EE 6.3x |
| 64 | 36.8 us | 72 us | EE 2.0x |
| 128 | 119 us | 197 us | EE 1.7x |
| 256 | 443 us | 204 us | MLX 2.2x |
| 512 | 2,335 us | 423 us | MLX 5.5x |
| 1024 | 13,558 us | 1,482 us | MLX 9.2x |

The GPU has a ~1 ms overhead floor per step regardless of model size (Metal command buffer submission). At batch=16 this amortizes to ~72 us/sample — a floor that EE beats up to d128.

![CPU vs GPU crossover](docs/crossover.png)

### Larger batches favor GPU

This model trains well with large batches — loss improves monotonically up to batch=128 with no per-sample throughput penalty on GPU. Larger batches amortize GPU dispatch overhead, pushing the crossover down:

| d_model | EE (batch=16) | MLX (batch=128) | Faster |
|---------|--------------|-----------------|--------|
| 16 | 3.0 us | 21 us | EE 7x |
| 32 | 11.4 us | 23 us | EE 2x |
| 64 | 36.8 us | 36 us | ~tied |
| 128 | 119 us | 41 us | MLX 2.9x |

At batch=128 the crossover drops to d64. At batch=256, it drops below d32.

### Where EE matters: seconds-scale sweeps

The GPU catches up and eventually wins — especially at larger batches and model sizes. But EE's advantage is in exactly the regime that matters most for this class of model: getting the best result you can in a few seconds.

Sweeping across (d_model, batch_size, learning_rate) for fastest time to loss < 2.0:

| Engine | Best config | Time to loss < 2.0 |
|--------|------------|-------------------|
| **EEmicroGPT** | d32, batch=64, lr=0.007 | **5.1s** |
| MLX GPU | d64, batch=256, lr=0.007 | 7.2s |

![Pareto frontier: loss vs wall time](docs/pareto_clean.png)

EE dominates the 0–7 second regime. MLX pulls ahead after ~20 seconds, where larger models have time to converge.

At EE's speed, a 100-configuration hyperparameter sweep finishes in under 10 minutes. That's the difference between an interactive experience — tweak, train, evaluate, repeat — and an overnight batch job. For small models where the goal is rapid exploration, a single CPU core with L1-resident weights is the faster path to a good model.

## How it works: optimization deep-dive

The codebase went from a straightforward C port (~200 us/step at d16) to 46.8 us/step through 20 incremental optimizations across forward, backward, and Adam. Here are the most impactful ones.

### 1. Neon SIMD everywhere (200 → 165 us, 18%)

Every inner loop operates on `float32x4_t` — ARM Neon's 128-bit SIMD type processing 4 floats per instruction. The core linear projection:

```c
static inline void linear(float *y, const float *W, const float *x, int out_dim, int in_dim) {
    for (int j = 0; j < out_dim; j++) {
        float32x4_t acc = vdupq_n_f32(0);
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(row + i), vld1q_f32(x + i));
        y[j] = vaddvq_f32(acc);
    }
}
```

`vfmaq_f32` is a fused multiply-add: `acc += a * b` in one cycle for 4 floats. This alone handles the dot products in all linear projections, attention score computation, and gradient accumulation.

Force-inlining (`__attribute__((always_inline))`) on `rms_norm`, `softmax`, and `linear` eliminated ~2048 function calls per step — a 15% win at d16 where each call processes only 16–27 elements.

### 2. Fast vectorized exp (165 → 147 us, 11%)

Softmax calls `expf()` ~23,000 times per step (BATCH * MAX_SEQ * N_HEADS * MAX_SEQ for attention, plus BATCH * MAX_SEQ * VOCAB for output). The standard `expf` is accurate to ULP but slow. We replace it with a degree-3 minimax polynomial:

```c
static inline float32x4_t fast_expq_f32(float32x4_t x) {
    x = vmulq_f32(x, vdupq_n_f32(1.442695041f));     // x * log2(e)
    float32x4_t xf = vrndmq_f32(x);                   // floor(x)
    float32x4_t r = vsubq_f32(x, xf);                  // fractional part
    int32x4_t pow2i = vshlq_n_s32(                     // 2^floor via IEEE754 exponent
        vaddq_s32(vcvtq_s32_f32(xf), vdupq_n_s32(127)), 23);
    // 2^frac via polynomial: 1 + 0.6931472*r + 0.2402265*r^2 + 0.0558011*r^3
    float32x4_t p = vfmaq_f32(vdupq_n_f32(0.2402265f), vdupq_n_f32(0.0558011f), r);
    p = vfmaq_f32(vdupq_n_f32(0.6931472f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    return vmulq_f32(vreinterpretq_f32_s32(pow2i), p);
}
```

The trick: `exp(x) = 2^(x * log2(e))`. Split into integer part (bit-shift into IEEE754 exponent field) and fractional part (cheap polynomial). Relative error ~1e-4, more than sufficient for softmax where only ratios matter.

### 3. Skip padding positions (123 → 72 us, 41%)

Names average ~6.5 characters but sequences are padded to MAX_SEQ=16. About 56% of positions are padding that contributes nothing to the loss or gradients. We skip them entirely:

```c
int slen = c->seq_lens[b];  // actual sequence length (varies per name)
for (int t = 0; t < slen; t++) { ... }  // instead of MAX_SEQ
```

This applies to forward projections, attention, FFN, backward input grads, and weight grads. The only caveat: SME2 batched operations still operate on MAX_SEQ-wide matrices (FMOPA processes 16 columns at once), so padding skip is primarily a scalar-path optimization.

### 4. Register-accumulator weight gradients (93 → 72 us, 23%)

Weight gradients are outer-product sums across positions: `dW[i][j] += sum_t dOut[t][i] * Input[t][j]`. The naive approach loads and stores `dW` for every position. Instead, we keep the accumulator in Neon registers across the entire position loop:

```c
for (int d = 0; d < D_MODEL; d++) {
    float32x4_t aq[D_MODEL/4], ak[D_MODEL/4], av[D_MODEL/4];
    // Load current grad row into registers
    for (int k = 0; k < D_MODEL/4; k++) {
        aq[k] = vld1q_f32(&g->Wq[d][k*4]);
        ak[k] = vld1q_f32(&g->Wk[d][k*4]);
        av[k] = vld1q_f32(&g->Wv[d][k*4]);
    }
    // Accumulate across ALL positions in registers
    for (int t = 0; t < slen - 1; t++) {
        float32x4_t dq = vdupq_n_f32(dQ[t][d]);
        for (int k = 0; k < D_MODEL/4; k++) {
            float32x4_t n1 = vld1q_f32(&c->norm1[b][t][k*4]);
            aq[k] = vfmaq_f32(aq[k], dq, n1);
            ak[k] = vfmaq_f32(ak[k], vdupq_n_f32(dK[t][d]), n1);
            av[k] = vfmaq_f32(av[k], vdupq_n_f32(dV[t][d]), n1);
        }
    }
    // Store once
    for (int k = 0; k < D_MODEL/4; k++) {
        vst1q_f32(&g->Wq[d][k*4], aq[k]);
        ...
    }
}
```

At d16, `D_MODEL/4 = 4` registers per weight matrix. With QKV sharing the position loop, that's 12 Neon registers for accumulators — fits perfectly in ARM's 32-register file. The position loop body becomes pure FMA with no stores.

### 5. Transposed-weight backward matvecs (68.6 → 64.3 us, 6%)

Backward input gradients need `W^T @ dOut`. Computing this from row-major `W[out][in]` requires strided column access (cache-unfriendly). We maintain explicit transposed copies:

```c
static float Wq_T[D_MODEL][D_MODEL];  // Wq[i][j] transposed to Wq_T[j][i]
static float Wf1_T[D_MODEL][D_FF];    // etc.
```

These live outside the `Model` struct so Adam doesn't waste time on them. They're synced after each Adam update. The backward matvec then reads contiguous rows:

```c
// dx += W^T @ dlogits  becomes  dx += Wlm_T @ dlogits (contiguous row access)
linear(dnorm2, (float*)Wf1_T, dff_hidden, D_MODEL, D_FF);
```

### 6. 2-position batched linear (55.5 → 50.4 us, 9%)

Instead of processing one position at a time, we share weight loads across two positions:

```c
static inline void linear_2(float *y0, float *y1, const float *W,
                             const float *x0, const float *x1, int out_dim, int in_dim) {
    for (int j = 0; j < out_dim; j++) {
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i += 4) {
            float32x4_t w = vld1q_f32(row + i);  // one weight load, used twice
            a0 = vfmaq_f32(a0, w, vld1q_f32(x0 + i));
            a1 = vfmaq_f32(a1, w, vld1q_f32(x1 + i));
        }
        y0[j] = vaddvq_f32(a0);
        y1[j] = vaddvq_f32(a1);
    }
}
```

This halves weight memory bandwidth for QKV, O projection, and FFN. Fused variants (`linear_relu_2`, `linear_residual_2`) eliminate intermediate storage too.

### 7. Fused backward pass (three-pass structure)

The backward pass is organized into three passes to maximize register and cache utilization:

**Pass 1 — Input gradients + attention backward** (sequential per position, from last to first):
- LM head backward (sparse — only 1/27 logit derivatives are nonzero)
- FFN backward via transposed-weight matvecs (`Wf2_T`, `Wf1_T`)
- RMS norm backward
- O projection backward (`Wo_T`)
- Full attention backward (softmax Jacobian, QKV gradients)

**Pass 2 — Weight gradients** (register-accumulator outer products):
- QKV weight grads accumulated across all positions in registers
- Wo, Wf1, Wf2 weight grads with tiled register accumulators

**Pass 3 — QKV input gradients + embedding backward**:
- `dnorm1 = Wq^T @ dQ + Wk^T @ dK + Wv^T @ dV` (three matvecs)
- Two RMS norm backward passes (norm1, emb_norm)
- Token and position embedding gradient scatter

### 8. SME2 outer-product engine (d_model >= 64)

On M4/M5 Macs, Apple's Scalable Matrix Extension (SME2) provides a 16x16 outer-product accumulator in hardware. The `FMOPA` instruction computes `ZA += col_vec * row_vec` — a rank-1 update of a 16x16 tile in a single instruction.

This maps perfectly to two operations:

**Forward projections** (GEMM): `Y[M][16] = W[M][K] @ X[K][16]`
```c
// Process 16 output rows at a time via tiled FMOPA
for (int bi = 0; bi < M; bi += 16) {
    svzero_za();
    for (int k = 0; k < K; k++) {
        svfloat32_t wt = svld1_f32(pred, W_T + k * M + bi);  // 16 weights
        svfloat32_t x  = svld1_f32(pred, X + k * MAX_SEQ);   // 16 positions
        svmopa_za32_f32_m(0, pred, pred, wt, x);              // ZA += outer(w, x)
    }
    // Read out 16 rows of 16 results
}
```

**Weight gradients** (outer-product sum): `dW[M][N] = sum_t dOut[t][:M] * Input[t][:N]`
```c
for (int bi = 0; bi < M; bi += 16) {
    for (int bj = 0; bj < N; bj += 16) {
        svzero_za();
        for (int t = 0; t < MAX_SEQ; t++) {
            svfloat32_t l = svld1_f32(pred, left + t * M + bi);
            svfloat32_t r = svld1_f32(pred, right + t * N + bj);
            svmopa_za32_f32_m(0, pred, pred, l, r);  // ZA += outer(dOut, Input)
        }
        // Read out 16x16 weight gradient tile
    }
}
```

SME2 is auto-enabled for d_model >= 64 (where FMOPA throughput beats the overhead of streaming mode entry/exit). At d_model < 64, the scalar Neon path wins because the SMSTART/SMSTOP overhead (~10 cycles each) dominates the tiny matrix operations.

### 9. Operation-sequential forward for L1 cache reuse

At d_model=64, the combined weight matrices (~210 KB) exceed the M5's 128 KB L1 data cache. The default batch-sequential approach (all operations for batch item 0, then batch item 1, ...) thrashes L1 because each batch item touches all weights.

The SME2 forward path restructures this into operation-sequential loops:

```
Loop 1: Embeddings + norms for all 16 batch items
Loop 2: QKV projections for all 16 batch items  (Wq_T, Wk_T, Wv_T stay in L1)
Loop 3: Attention for all 16 batch items          (no weights needed)
Loop 4: O projection for all 16 batch items       (Wo_T stays in L1)
Loop 5: Pre-FFN norm for all 16 batch items
Loop 6: FFN for all 16 batch items                (Wf1_T, Wf2_T stay in L1)
Loop 7: LM head for all 16 batch items            (Wlm stays in L1)
```

Each operation's weights are loaded into L1 once and reused across all 16 batch items before moving to the next operation. This reduced d64 SME2 forward+backward from 670 to 603 us/step (10%).

### 10. Fused streaming-mode backward (d64: 717 → 582 us, 19%)

SME2 functions tagged `__arm_locally_streaming` force the CPU into streaming mode (SMSTART) on entry and back out (SMSTOP) on exit. The backward pass called `sme2_outer_sum` six times back-to-back for weight gradients (dWq, dWk, dWv, dWo, dWf2, dWf1), plus `sme2_qkv_input_grad` — that's 7 streaming mode transitions per batch item, 112 per step.

Each transition has direct costs (SMSTART/SMSTOP ~20-40 cycles each, ZA zeroing, register save/restore) but also indirect costs: the old code wrote each outer-product result to a temporary `dW_tmp` array, then ran a scalar loop to accumulate it into the gradient array. That's 6 extra load-accumulate-store passes over D_MODEL×D_MODEL floats.

The fix: one function that enters streaming mode once, computes all 6 outer-product sums with inline SVE accumulation directly into the gradient arrays, transposes dQ/dK/dV, and computes the QKV input gradients — all before exiting streaming mode:

```c
__arm_locally_streaming __arm_new("za")
static void sme2_backward_wgrads_and_input_grads(Grads *g, float *dnorm1_out, ...) {
    // For each weight matrix: FMOPA tiles → SVE-add directly into g->Wq etc.
    #define OUTER_SUM_ACCUM(dW, left, right, M, N) \
        for (int bi = 0; bi < (M); bi += 16) { \
            for (int bj = 0; bj < (N); bj += 16) { \
                svzero_za(); \
                for (int t = 0; t < MAX_SEQ; t++) { \
                    svmopa_za32_f32_m(0, pred, pred, \
                        svld1_f32(pred, (left) + t*(M) + bi), \
                        svld1_f32(pred, (right) + t*(N) + bj)); \
                } \
                for (int i = 0; i < 16; i++) { \
                    svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i); \
                    svfloat32_t cur = svld1_f32(pred, (dW) + (bi+i)*(N) + bj); \
                    svst1_f32(pred, (dW) + (bi+i)*(N) + bj, svadd_f32_x(pred, cur, row)); \
                } \
            } \
        }
    OUTER_SUM_ACCUM(g->Wq, dQ, norm1, D_MODEL, D_MODEL)  // ... ×6 weight matrices
    // Then: transpose dQ/dK/dV and compute QKV input grads via FMOPA
}
```

Results:

| Config | Before | After | Improvement |
|--------|--------|-------|-------------|
| d64 SME2 | 717 us | 582 us | **19%** |
| d128 SME2 | 2625 us | 1855 us | **29%** |

The d128 win is larger because it has more FMOPA tiles (8×8 vs 4×4 at d64) so the eliminated intermediate memory traffic is proportionally bigger.

## Optimization progression (d_model=16)

| Step | Change | us/step | Improvement |
|------|--------|---------|-------------|
| 0 | Baseline C port | 200 | — |
| 1 | Force-inline rms_norm/softmax | 170 | 15% |
| 2 | Neon HEAD_DIM=4 attention | 165 | 3% |
| 3 | Fast Neon exp approximation | 147 | 11% |
| 4 | Batched GEMM forward | 140 | 5% |
| 5 | Fused backward (Phase 1+2 merge) | 132 | 6% |
| 6 | Neon-vectorize all backward loops | 126 | 5% |
| 7 | Register-accumulator GEMM | 123 | 2% |
| 8 | Skip padding (fwd+bwd) | 81 | 34% |
| 9 | Variable-N GEMM | 72 | 11% |
| 10 | Neon linear(), eliminate transposes | 68.6 | 5% |
| 11 | Dot-product backward via transposed weights | 64.3 | 6% |
| 12 | Fused linear_relu, Neon Adam | 61 | 5% |
| 13 | `__restrict__` on hot functions | 55.5 | 9% |
| 14 | 2-pos batching, fused attention, reg-accum wgrads | 50.4 | 9% |
| 15 | Fused attention, final tuning | 46.8 | 7% |

**Total: 200 → 46.8 us/step (4.3x)**

## What didn't work

Not every idea was a win:

- **SME2 at d_model=16**: 10% regression. The SMSTART/SMSTOP streaming mode transitions (~10 cycles each) dominate when matrix tiles are only 16x16 (one tile per projection).
- **Removing zero-checks in LM head backward**: Regression. The dlogits vector is 96% zeros (26/27 entries), so the zero-skip branch saves ~96% of the inner loop work. Similar story for ReLU-masked FFN hidden gradients (~50% zero).
- **Splitting fused QKV weight+input grad loop**: Regression from cache pressure. The fused loop keeps dQ/dK/dV/norm1 hot in L1; splitting forces re-reads.
- **Reducing Cache struct size**: No effect. The per-batch working set (~29 KB) already fits L1 (128 KB).
- **Fusing FFN expand + ReLU + contract in streaming mode**: Regression. The ReLU + cache save between expand and contract involves a transposing scatter (`[D_FF][MAX_SEQ]` → `[MAX_SEQ][D_FF]`). Inside streaming mode, this runs as SSVE scalar code at ~3.6x the cost of Neon, outweighing the saved SMSTART/SMSTOP transition.

## File structure

It's [one file](https://github.com/Entrpi/eemicrogpt/blob/master/eemicrogpt.c) . 1791 lines. That's the point.

```
eemicrogpt.c    — everything
names.txt       — training data (32K names, from Karpathy's makemore)
```

## License

MIT
