// FemtoGPT: A tiny GPT trained from scratch in C with SME2/Neon on Apple M5
// Single-file implementation: 1-layer transformer, character-level name generation
// Architecture matches AttoGPT: https://gist.github.com/JGalego/26d617e5c939af0c32f3c16e4e392803
//
// Compile (without SME2): clang -O3 -ffast-math -o femtogpt femtogpt.c -lm
// Compile (with SME2):    clang -O3 -mcpu=native+sme2 -ffast-math -o femtogpt femtogpt.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mach/mach_time.h>
#include <arm_neon.h>

// ============================================================
// Architecture constants (matching AttoGPT)
// ============================================================
#ifndef D_MODEL
#define D_MODEL    64
#endif
#ifndef N_HEADS
#define N_HEADS    4
#endif
#define HEAD_DIM   (D_MODEL / N_HEADS)
#define D_FF       (4 * D_MODEL)
#define VOCAB      27  // 26 letters (0..25) + boundary token (26)
#define BOUNDARY   26  // boundary/null token index (B in AttoGPT)
#define MAX_SEQ    16
#ifndef BATCH
#define BATCH      16
#endif
#ifndef N_STEPS
#define N_STEPS    1000
#endif
#ifndef LR_INIT
#define LR_INIT    0.01f
#endif
#define BETA1      0.85f
#define BETA2      0.99f
#define EPS        1e-8f
#define INV_SQRT_HD (1.0f / sqrtf((float)HEAD_DIM))

#ifdef __ARM_FEATURE_SME2
#include <arm_sme.h>
#ifndef USE_SME2
#if D_MODEL >= 64
#define USE_SME2 1
#else
#define USE_SME2 0
#endif
#endif
#else
#define USE_SME2 0
#endif

// ============================================================
// RNG: xorshift64
// ============================================================
static uint64_t rng_state = 42;

static uint64_t xorshift64(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static float randf(void) {
    return (float)(xorshift64() & 0xFFFFFF) / (float)0xFFFFFF;
}

static float rand_normal(void) {
    float u1 = randf() + 1e-10f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// ============================================================
// Model parameters
// AttoGPT stores weights as W[out][in] (each row dotted with input).
// We store as W[out][in] to match, and our linear() handles this.
// ============================================================
typedef struct {
    float tok_emb[VOCAB][D_MODEL];       // token embeddings: te[V][E]
    float pos_emb[MAX_SEQ][D_MODEL];     // position embeddings: pe[T][E]

    // Attention weights
    float Wq[D_MODEL][D_MODEL];          // query projection: q[E][E]
    float Wk[D_MODEL][D_MODEL];          // key projection
    float Wv[D_MODEL][D_MODEL];          // value projection
    float Wo[D_MODEL][D_MODEL];          // output projection

    // FFN weights
    // AttoGPT: S["01"] = M(E*4, E) = [64][16], S["02"] = M(E, E*4) = [16][64]
    float Wf1[D_FF][D_MODEL];            // expand: [64][16], out=64, in=16
    float Wf2[D_MODEL][D_FF];            // contract: [16][64], out=16, in=64

    // LM head: AttoGPT S["lm"] = M(V, E) = [27][16]
    float Wlm[VOCAB][D_MODEL];           // lm head: [27][16]
} Model;

// ============================================================
// Activation cache for backprop
// ============================================================
typedef struct {
    int    tokens[BATCH][MAX_SEQ];
    int    seq_lens[BATCH];                         // number of tokens including boundary
    float  raw_emb[BATCH][MAX_SEQ][D_MODEL];        // tok_emb + pos_emb (before first norm)
    float  emb[BATCH][MAX_SEQ][D_MODEL];            // after first RMS norm (= residual stream input)
    float  emb_rms[BATCH][MAX_SEQ];                 // RMS values for first norm backward
    float  norm1[BATCH][MAX_SEQ][D_MODEL];          // after second RMS norm (pre-attention)
    float  norm1_rms[BATCH][MAX_SEQ];               // RMS values for norm1 backward
    float  Q[BATCH][MAX_SEQ][D_MODEL];
    float  K[BATCH][MAX_SEQ][D_MODEL];
    float  V[BATCH][MAX_SEQ][D_MODEL];
    float  attn_scores[BATCH][N_HEADS][MAX_SEQ][MAX_SEQ]; // post-softmax
    float  attn_out[BATCH][MAX_SEQ][D_MODEL];       // attention value aggregation output
    float  res1[BATCH][MAX_SEQ][D_MODEL];           // after first residual (attn + emb)
    float  norm2[BATCH][MAX_SEQ][D_MODEL];          // after RMS norm pre-FFN
    float  norm2_rms[BATCH][MAX_SEQ];
    float  ff_pre_relu[BATCH][MAX_SEQ][D_FF];       // FFN hidden before ReLU
    float  ff_hidden[BATCH][MAX_SEQ][D_FF];         // FFN hidden after ReLU
    float  res2[BATCH][MAX_SEQ][D_MODEL];           // after second residual (FFN + res1)
    float  logits[BATCH][MAX_SEQ][VOCAB];
    float  probs[BATCH][MAX_SEQ][VOCAB];
    float  loss;
} Cache;

// ============================================================
// Gradients (mirrors Model)
// ============================================================
typedef struct {
    float tok_emb[VOCAB][D_MODEL];
    float pos_emb[MAX_SEQ][D_MODEL];
    float Wq[D_MODEL][D_MODEL];
    float Wk[D_MODEL][D_MODEL];
    float Wv[D_MODEL][D_MODEL];
    float Wo[D_MODEL][D_MODEL];
    float Wf1[D_FF][D_MODEL];
    float Wf2[D_MODEL][D_FF];
    float Wlm[VOCAB][D_MODEL];
} Grads;

static int param_count(void) {
    return sizeof(Model) / sizeof(float);
}

// ============================================================
// Data loading
// ============================================================
#define MAX_NAMES 40000
static int    all_tokens[MAX_NAMES][MAX_SEQ];
static int    all_lengths[MAX_NAMES];
static int    num_names = 0;

static void load_names(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    char line[256];
    while (fgets(line, sizeof(line), f) && num_names < MAX_NAMES) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r')) len--;
        if (len == 0) continue;

        // AttoGPT encoding: t = [B] + [char indices] + [B]
        // Where B = 26 (boundary token), chars = 0..25 (a=0, b=1, ..., z=25)
        // Sequence: [BOUNDARY, char0, char1, ..., charN-1, BOUNDARY, pad...]
        int *tok = all_tokens[num_names];
        memset(tok, 0, MAX_SEQ * sizeof(int));

        int pos = 0;
        tok[pos++] = BOUNDARY; // start boundary

        for (int i = 0; i < len && pos < MAX_SEQ - 1; i++) {
            char c = line[i];
            if (c >= 'a' && c <= 'z') tok[pos++] = c - 'a';
            else if (c >= 'A' && c <= 'Z') tok[pos++] = c - 'A';
        }

        tok[pos++] = BOUNDARY; // end boundary
        all_lengths[num_names] = pos; // total length including both boundaries

        // Clamp
        if (all_lengths[num_names] > MAX_SEQ)
            all_lengths[num_names] = MAX_SEQ;

        num_names++;
    }
    fclose(f);
}

// ============================================================
// Initialization (AttoGPT uses gauss(0, 0.08))
// ============================================================
static void gauss_init(float *w, int count, float std) {
    for (int i = 0; i < count; i++)
        w[i] = rand_normal() * std;
}

static void init_model(Model *m) {
    memset(m, 0, sizeof(Model));
    float std = 0.08f;
    gauss_init((float*)m->tok_emb, VOCAB * D_MODEL, std);
    gauss_init((float*)m->pos_emb, MAX_SEQ * D_MODEL, std);
    gauss_init((float*)m->Wq, D_MODEL * D_MODEL, std);
    gauss_init((float*)m->Wk, D_MODEL * D_MODEL, std);
    gauss_init((float*)m->Wv, D_MODEL * D_MODEL, std);
    gauss_init((float*)m->Wo, D_MODEL * D_MODEL, std);
    gauss_init((float*)m->Wf1, D_FF * D_MODEL, std);
    gauss_init((float*)m->Wf2, D_MODEL * D_FF, std);
    gauss_init((float*)m->Wlm, VOCAB * D_MODEL, std);
}

// ============================================================
// Forward pass primitives
// ============================================================

static inline __attribute__((always_inline)) void rms_norm(float *out, const float *x, int n, float *rms_out) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / (float)n + 1e-5f);
    if (rms_out) *rms_out = rms;
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms;
}

// y[j] = sum_i W[j][i] * x[i] for j=0..out_dim-1
// W is stored as W[out_dim][in_dim] (row-major), matching AttoGPT's li(x,w)
static inline __attribute__((always_inline)) void linear(
    float * __restrict__ y, const float * __restrict__ W,
    const float * __restrict__ x, int out_dim, int in_dim
) {
    for (int j = 0; j < out_dim; j++) {
        float32x4_t acc = vdupq_n_f32(0);
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(row + i), vld1q_f32(x + i));
        y[j] = vaddvq_f32(acc);
    }
}

// y[j] = base[j] + sum_i W[j][i] * x[i] — fused matvec + residual
static inline __attribute__((always_inline)) void linear_residual(
    float * __restrict__ y, const float * __restrict__ base,
    const float * __restrict__ W, const float * __restrict__ x, int out_dim, int in_dim
) {
    for (int j = 0; j < out_dim; j++) {
        float32x4_t acc = vdupq_n_f32(0);
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(row + i), vld1q_f32(x + i));
        y[j] = base[j] + vaddvq_f32(acc);
    }
}

// Fused linear + ReLU: pre_relu[j] = W@x, hidden[j] = max(0, pre_relu[j])
static inline __attribute__((always_inline)) void linear_relu(
    float * __restrict__ pre_relu, float * __restrict__ hidden,
    const float * __restrict__ W, const float * __restrict__ x, int out_dim, int in_dim
) {
    for (int j = 0; j < out_dim; j++) {
        float32x4_t acc = vdupq_n_f32(0);
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i += 4)
            acc = vfmaq_f32(acc, vld1q_f32(row + i), vld1q_f32(x + i));
        float val = vaddvq_f32(acc);
        pre_relu[j] = val;
        hidden[j] = val > 0.0f ? val : 0.0f;
    }
}

// 2-position batched linear: shares weight loads across 2 input vectors
static inline __attribute__((always_inline)) void linear_2(
    float * __restrict__ y0, float * __restrict__ y1,
    const float * __restrict__ W,
    const float * __restrict__ x0, const float * __restrict__ x1,
    int out_dim, int in_dim
) {
    for (int j = 0; j < out_dim; j++) {
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i += 4) {
            float32x4_t w = vld1q_f32(row + i);
            a0 = vfmaq_f32(a0, w, vld1q_f32(x0 + i));
            a1 = vfmaq_f32(a1, w, vld1q_f32(x1 + i));
        }
        y0[j] = vaddvq_f32(a0);
        y1[j] = vaddvq_f32(a1);
    }
}

// 2-position batched linear + ReLU: shares weight loads across 2 input vectors
static inline __attribute__((always_inline)) void linear_relu_2(
    float * __restrict__ pr0, float * __restrict__ h0,
    float * __restrict__ pr1, float * __restrict__ h1,
    const float * __restrict__ W,
    const float * __restrict__ x0, const float * __restrict__ x1,
    int out_dim, int in_dim
) {
    for (int j = 0; j < out_dim; j++) {
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i += 4) {
            float32x4_t w = vld1q_f32(row + i);
            a0 = vfmaq_f32(a0, w, vld1q_f32(x0 + i));
            a1 = vfmaq_f32(a1, w, vld1q_f32(x1 + i));
        }
        float v0 = vaddvq_f32(a0), v1 = vaddvq_f32(a1);
        pr0[j] = v0; h0[j] = v0 > 0.0f ? v0 : 0.0f;
        pr1[j] = v1; h1[j] = v1 > 0.0f ? v1 : 0.0f;
    }
}

// 2-position batched linear + residual: shares weight loads across 2 input vectors
static inline __attribute__((always_inline)) void linear_residual_2(
    float * __restrict__ y0, const float * __restrict__ b0,
    float * __restrict__ y1, const float * __restrict__ b1,
    const float * __restrict__ W,
    const float * __restrict__ x0, const float * __restrict__ x1,
    int out_dim, int in_dim
) {
    for (int j = 0; j < out_dim; j++) {
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i += 4) {
            float32x4_t w = vld1q_f32(row + i);
            a0 = vfmaq_f32(a0, w, vld1q_f32(x0 + i));
            a1 = vfmaq_f32(a1, w, vld1q_f32(x1 + i));
        }
        y0[j] = b0[j] + vaddvq_f32(a0);
        y1[j] = b1[j] + vaddvq_f32(a1);
    }
}

// Fast scalar exp approximation (relative error ~1e-4, matching the vector version)
static inline float fast_expf_s(float x) {
    x *= 1.442695041f; // log2(e)
    if (x < -126.0f) x = -126.0f;
    if (x > 126.0f) x = 126.0f;
    float xf = __builtin_floorf(x);
    float r = x - xf;
    int xi = (int)xf;
    union { float f; int32_t i; } u;
    u.i = (xi + 127) << 23;
    float p = 0.0558011f;
    p = 0.2402265f + p * r;
    p = 0.6931472f + p * r;
    p = 1.0f + p * r;
    return u.f * p;
}

// Fast vectorized exp approximation (relative error ~1e-4, sufficient for softmax)
// Uses exp(x) = exp2(x * log2(e)) with polynomial refinement
static inline float32x4_t fast_expq_f32(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.442695041f);
    x = vmulq_f32(x, log2e); // x * log2(e) = log2(exp(x))
    // Clamp to avoid overflow/underflow
    x = vmaxq_f32(x, vdupq_n_f32(-126.0f));
    x = vminq_f32(x, vdupq_n_f32(126.0f));
    // Split into integer and fractional parts
    float32x4_t xf = vrndmq_f32(x); // floor
    float32x4_t r = vsubq_f32(x, xf); // fractional part [0, 1)
    int32x4_t xi = vcvtq_s32_f32(xf);
    // 2^integer via bit shift into float exponent
    int32x4_t pow2i = vshlq_n_s32(vaddq_s32(xi, vdupq_n_s32(127)), 23);
    float32x4_t pow2f = vreinterpretq_f32_s32(pow2i);
    // 2^fractional via degree-3 minimax polynomial on [0, 1)
    // p(r) ≈ 1 + 0.6931472*r + 0.2402265*r^2 + 0.0558011*r^3
    float32x4_t p = vdupq_n_f32(0.0558011f);
    p = vfmaq_f32(vdupq_n_f32(0.2402265f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.6931472f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    return vmulq_f32(pow2f, p);
}

static inline __attribute__((always_inline)) void softmax(float *out, const float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
#if MAX_SEQ == 16
    if (n == MAX_SEQ) {
        // Specialized Neon path for n=16 (attention softmax)
        float32x4_t vmax = vdupq_n_f32(max_val);
        float32x4_t v0 = fast_expq_f32(vsubq_f32(vld1q_f32(x), vmax));
        float32x4_t v1 = fast_expq_f32(vsubq_f32(vld1q_f32(x+4), vmax));
        float32x4_t v2 = fast_expq_f32(vsubq_f32(vld1q_f32(x+8), vmax));
        float32x4_t v3 = fast_expq_f32(vsubq_f32(vld1q_f32(x+12), vmax));
        sum = vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
        float32x4_t inv = vdupq_n_f32(1.0f / sum);
        vst1q_f32(out, vmulq_f32(v0, inv));
        vst1q_f32(out+4, vmulq_f32(v1, inv));
        vst1q_f32(out+8, vmulq_f32(v2, inv));
        vst1q_f32(out+12, vmulq_f32(v3, inv));
        return;
    }
#endif
    {
        float32x4_t vmax = vdupq_n_f32(max_val);
        float32x4_t vsum = vdupq_n_f32(0.0f);
        int i;
        for (i = 0; i + 3 < n; i += 4) {
            float32x4_t v = fast_expq_f32(vsubq_f32(vld1q_f32(x + i), vmax));
            vst1q_f32(out + i, v);
            vsum = vaddq_f32(vsum, v);
        }
        sum = vaddvq_f32(vsum);
        for (; i < n; i++) {
            out[i] = expf(x[i] - max_val);
            sum += out[i];
        }
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) out[i] *= inv_sum;
}

// Transposed weight copies for backward dot-product matvecs (and SME2 FMOPA).
// Maintained outside Model struct so Adam doesn't touch them.
static float Wq_T[D_MODEL][D_MODEL];
static float Wk_T[D_MODEL][D_MODEL];
static float Wv_T[D_MODEL][D_MODEL];
static float Wo_T[D_MODEL][D_MODEL];
static float Wf1_T[D_MODEL][D_FF];   // Wf1[D_FF][D_MODEL] transposed
static float Wf2_T[D_FF][D_MODEL];   // Wf2[D_MODEL][D_FF] transposed

static void sync_transposed_weights(Model *m) {
    for (int i = 0; i < D_MODEL; i++)
        for (int j = 0; j < D_MODEL; j++) {
            Wq_T[j][i] = m->Wq[i][j];
            Wk_T[j][i] = m->Wk[i][j];
            Wv_T[j][i] = m->Wv[i][j];
            Wo_T[j][i] = m->Wo[i][j];
        }
    for (int i = 0; i < D_FF; i++)
        for (int j = 0; j < D_MODEL; j++)
            Wf1_T[j][i] = m->Wf1[i][j];
    for (int i = 0; i < D_MODEL; i++)
        for (int j = 0; j < D_FF; j++)
            Wf2_T[j][i] = m->Wf2[i][j];
}

// ============================================================
// SME2 FMOPA matmul functions (generalized for any D_MODEL multiple of 16)
// ============================================================
#if USE_SME2

// C[M][16] = W_T[K][M] @ X[K][16] using tiled FMOPA
// FMOPA: ZA[i][j] += left[i] * right[j]
// For tile bi: ZA[i_local][j] += W_T[k][bi+i_local] * X[k][j]
// = W[bi+i_local][k] * X[k][j]  ✓

// QKV projections in one streaming session
__arm_locally_streaming __arm_new("za")
static void sme2_qkv_project(
    float *Q_out, float *K_out, float *V_out, const float *X
) {
    svbool_t pred = svptrue_b32();
    // Q = Wq @ X:  Q[D_MODEL][MAX_SEQ]
    for (int bi = 0; bi < D_MODEL; bi += 16) {
        svzero_za();
        for (int k = 0; k < D_MODEL; k++) {
            svfloat32_t wt = svld1_f32(pred, (float*)Wq_T + k * D_MODEL + bi);
            svfloat32_t x  = svld1_f32(pred, X + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wt, x);
        }
        for (int i = 0; i < 16; i++) {
            svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
            svst1_f32(pred, Q_out + (bi + i) * MAX_SEQ, row);
        }
    }
    // K = Wk @ X
    for (int bi = 0; bi < D_MODEL; bi += 16) {
        svzero_za();
        for (int k = 0; k < D_MODEL; k++) {
            svfloat32_t wt = svld1_f32(pred, (float*)Wk_T + k * D_MODEL + bi);
            svfloat32_t x  = svld1_f32(pred, X + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wt, x);
        }
        for (int i = 0; i < 16; i++) {
            svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
            svst1_f32(pred, K_out + (bi + i) * MAX_SEQ, row);
        }
    }
    // V = Wv @ X
    for (int bi = 0; bi < D_MODEL; bi += 16) {
        svzero_za();
        for (int k = 0; k < D_MODEL; k++) {
            svfloat32_t wt = svld1_f32(pred, (float*)Wv_T + k * D_MODEL + bi);
            svfloat32_t x  = svld1_f32(pred, X + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wt, x);
        }
        for (int i = 0; i < 16; i++) {
            svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
            svst1_f32(pred, V_out + (bi + i) * MAX_SEQ, row);
        }
    }
}

// O projection: O[D_MODEL][MAX_SEQ] = Wo @ X
__arm_locally_streaming __arm_new("za")
static void sme2_o_project(float *O_out, const float *X) {
    svbool_t pred = svptrue_b32();
    for (int bi = 0; bi < D_MODEL; bi += 16) {
        svzero_za();
        for (int k = 0; k < D_MODEL; k++) {
            svfloat32_t wt = svld1_f32(pred, (float*)Wo_T + k * D_MODEL + bi);
            svfloat32_t x  = svld1_f32(pred, X + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wt, x);
        }
        for (int i = 0; i < 16; i++) {
            svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
            svst1_f32(pred, O_out + (bi + i) * MAX_SEQ, row);
        }
    }
}

// FFN expand: H[D_FF][MAX_SEQ] = Wf1[D_FF][D_MODEL] @ X[D_MODEL][MAX_SEQ]
// Using Wf1_T[D_MODEL][D_FF]. Tiles over D_FF output rows.
__arm_locally_streaming __arm_new("za")
static void sme2_ffn_expand(float *H, const float *X) {
    svbool_t pred = svptrue_b32();
    for (int bi = 0; bi < D_FF; bi += 16) {
        svzero_za();
        for (int k = 0; k < D_MODEL; k++) {
            svfloat32_t wt = svld1_f32(pred, (float*)Wf1_T + k * D_FF + bi);
            svfloat32_t x  = svld1_f32(pred, X + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wt, x);
        }
        for (int i = 0; i < 16; i++) {
            svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
            svst1_f32(pred, H + (bi + i) * MAX_SEQ, row);
        }
    }
}

// FFN contract: Y[D_MODEL][MAX_SEQ] = Wf2[D_MODEL][D_FF] @ H[D_FF][MAX_SEQ]
// Using Wf2_T[D_FF][D_MODEL]. Tiles over D_MODEL output rows.
__arm_locally_streaming __arm_new("za")
static void sme2_ffn_contract(float *Y, const float *H) {
    svbool_t pred = svptrue_b32();
    for (int bi = 0; bi < D_MODEL; bi += 16) {
        svzero_za();
        for (int k = 0; k < D_FF; k++) {
            svfloat32_t wt = svld1_f32(pred, (float*)Wf2_T + k * D_MODEL + bi);
            svfloat32_t h  = svld1_f32(pred, H + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wt, h);
        }
        for (int i = 0; i < 16; i++) {
            svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
            svst1_f32(pred, Y + (bi + i) * MAX_SEQ, row);
        }
    }
}

// ---- Backward helpers ----

// General outer product sum: dW[M][N] = sum_{t=0}^{MAX_SEQ-1} left[t][:M] outer right[t][:N]
// left is [MAX_SEQ][M], right is [MAX_SEQ][N]
__arm_locally_streaming __arm_new("za")
static void sme2_outer_sum(float *dW, const float *left, const float *right, int M, int N) {
    svbool_t pred = svptrue_b32();
    for (int bi = 0; bi < M; bi += 16) {
        for (int bj = 0; bj < N; bj += 16) {
            svzero_za();
            for (int t = 0; t < MAX_SEQ; t++) {
                svfloat32_t l = svld1_f32(pred, left + t * M + bi);
                svfloat32_t r = svld1_f32(pred, right + t * N + bj);
                svmopa_za32_f32_m(0, pred, pred, l, r);
            }
            for (int i = 0; i < 16; i++) {
                svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
                svst1_f32(pred, dW + (bi + i) * N + bj, row);
            }
        }
    }
}

// General batched matmul: C[M][MAX_SEQ] = W^T @ X[K][MAX_SEQ]
// W_T_flat points to the "transposed weight" stored as [K][M] (row-major).
// For forward: pass explicit transposed copy (Wq_T, etc.)
// For backward input grads: pass ORIGINAL weight (Wf2, Wf1, Wo) directly.
__arm_locally_streaming __arm_new("za")
static void sme2_matmul(float *C, const float *W_T_flat, const float *X, int M, int K) {
    svbool_t pred = svptrue_b32();
    for (int bi = 0; bi < M; bi += 16) {
        svzero_za();
        for (int k = 0; k < K; k++) {
            svfloat32_t w = svld1_f32(pred, W_T_flat + k * M + bi);
            svfloat32_t x = svld1_f32(pred, X + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, w, x);
        }
        for (int i = 0; i < 16; i++) {
            svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
            svst1_f32(pred, C + (bi + i) * MAX_SEQ, row);
        }
    }
}

// Batched QKV input gradient:
// dnorm1[D_MODEL][MAX_SEQ] = Wq_T @ dQ_T + Wk_T @ dK_T + Wv_T @ dV_T
// All inputs/outputs in [D_MODEL][MAX_SEQ] layout.
__arm_locally_streaming __arm_new("za")
static void sme2_qkv_input_grad(
    float *dnorm1, const float *dQ_T, const float *dK_T, const float *dV_T
) {
    svbool_t pred = svptrue_b32();
    for (int bi = 0; bi < D_MODEL; bi += 16) {
        svzero_za();
        // Accumulate Wq_T @ dQ_T + Wk_T @ dK_T + Wv_T @ dV_T
        for (int k = 0; k < D_MODEL; k++) {
            svfloat32_t wq = svld1_f32(pred, (float*)Wq_T + k * D_MODEL + bi);
            svfloat32_t dq = svld1_f32(pred, dQ_T + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wq, dq);
        }
        for (int k = 0; k < D_MODEL; k++) {
            svfloat32_t wk = svld1_f32(pred, (float*)Wk_T + k * D_MODEL + bi);
            svfloat32_t dk = svld1_f32(pred, dK_T + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wk, dk);
        }
        for (int k = 0; k < D_MODEL; k++) {
            svfloat32_t wv = svld1_f32(pred, (float*)Wv_T + k * D_MODEL + bi);
            svfloat32_t dv = svld1_f32(pred, dV_T + k * MAX_SEQ);
            svmopa_za32_f32_m(0, pred, pred, wv, dv);
        }
        for (int i = 0; i < 16; i++) {
            svfloat32_t row = svread_hor_za32_f32_m(svdup_f32(0), pred, 0, i);
            svst1_f32(pred, dnorm1 + (bi + i) * MAX_SEQ, row);
        }
    }
}

#endif // USE_SME2

#if !USE_SME2
#endif // !USE_SME2

// ============================================================
// Forward pass
// Matches AttoGPT flow:
//   x = rms(tok_emb[t] + pos_emb[p])
//   r = x; y = rms(x)
//   QKV on y, attention, O proj
//   x = o_proj + r
//   r = x; FFN(rms(x)) + r
//   logits = lm(x)
// ============================================================
static float forward(Model *m, Cache *c) {
#if USE_SME2
    // Operation-sequential: process all batch items per operation for L1 cache reuse
    // Weights (~210 KB at d=64) exceed L1 (128 KB), so keeping one weight set hot helps

    // 1-3. Embeddings + RMS norms for all batch items
    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];
        for (int t = 0; t < slen; t++) {
            int tok = c->tokens[b][t];
            for (int d = 0; d < D_MODEL; d++)
                c->raw_emb[b][t][d] = m->tok_emb[tok][d] + m->pos_emb[t][d];
            rms_norm(c->emb[b][t], c->raw_emb[b][t], D_MODEL, &c->emb_rms[b][t]);
            rms_norm(c->norm1[b][t], c->emb[b][t], D_MODEL, &c->norm1_rms[b][t]);
        }
    }

    // 4. QKV projections for all batch items (Wq_T, Wk_T, Wv_T stay hot in L1)
    for (int b = 0; b < BATCH; b++) {
        float X_T[D_MODEL][MAX_SEQ], Q_T[D_MODEL][MAX_SEQ], K_T[D_MODEL][MAX_SEQ], V_T[D_MODEL][MAX_SEQ];
        for (int t = 0; t < MAX_SEQ; t++)
            for (int d = 0; d < D_MODEL; d++)
                X_T[d][t] = c->norm1[b][t][d];
        sme2_qkv_project((float*)Q_T, (float*)K_T, (float*)V_T, (float*)X_T);
        for (int t = 0; t < MAX_SEQ; t++)
            for (int d = 0; d < D_MODEL; d++) {
                c->Q[b][t][d] = Q_T[d][t];
                c->K[b][t][d] = K_T[d][t];
                c->V[b][t][d] = V_T[d][t];
            }
    }

    // 5. Attention for all batch items (no weight matrices, just activations)
    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];
        for (int h = 0; h < N_HEADS; h++) {
            int hoff = h * HEAD_DIM;
            for (int t = 0; t < slen; t++) {
                float max_score = -1e30f;
#if HEAD_DIM == 4
                float32x4_t qt = vld1q_f32(&c->Q[b][t][hoff]);
                for (int s = 0; s <= t; s++) {
                    float32x4_t ks = vld1q_f32(&c->K[b][s][hoff]);
                    float score = vaddvq_f32(vmulq_f32(qt, ks)) * INV_SQRT_HD;
                    c->attn_scores[b][h][t][s] = score;
                    if (score > max_score) max_score = score;
                }
                float sum = 0.0f;
                float32x4_t vacc = vdupq_n_f32(0.0f);
                for (int s = 0; s <= t; s++) {
                    float e = fast_expf_s(c->attn_scores[b][h][t][s] - max_score);
                    c->attn_scores[b][h][t][s] = e;
                    sum += e;
                    vacc = vfmaq_n_f32(vacc, vld1q_f32(&c->V[b][s][hoff]), e);
                }
                float inv_sum = 1.0f / sum;
                for (int s = 0; s <= t; s++)
                    c->attn_scores[b][h][t][s] *= inv_sum;
                vst1q_f32(&c->attn_out[b][t][hoff], vmulq_n_f32(vacc, inv_sum));
#else
                for (int s = 0; s <= t; s++) {
                    float score = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++)
                        score += c->Q[b][t][hoff + d] * c->K[b][s][hoff + d];
                    score *= INV_SQRT_HD;
                    c->attn_scores[b][h][t][s] = score;
                    if (score > max_score) max_score = score;
                }
                float sum = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) c->attn_out[b][t][hoff + d] = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float e = fast_expf_s(c->attn_scores[b][h][t][s] - max_score);
                    c->attn_scores[b][h][t][s] = e;
                    sum += e;
                    for (int d = 0; d < HEAD_DIM; d++)
                        c->attn_out[b][t][hoff + d] += e * c->V[b][s][hoff + d];
                }
                float inv_sum = 1.0f / sum;
                for (int s = 0; s <= t; s++)
                    c->attn_scores[b][h][t][s] *= inv_sum;
                for (int d = 0; d < HEAD_DIM; d++)
                    c->attn_out[b][t][hoff + d] *= inv_sum;
#endif
            }
        }
    }

    // 6. O projection + residual for all batch items (Wo_T stays hot)
    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];
        float A_T[D_MODEL][MAX_SEQ], O_T[D_MODEL][MAX_SEQ];
        for (int t = 0; t < MAX_SEQ; t++)
            for (int d = 0; d < D_MODEL; d++)
                A_T[d][t] = c->attn_out[b][t][d];
        sme2_o_project((float*)O_T, (float*)A_T);
        for (int t = 0; t < slen; t++)
            for (int d = 0; d < D_MODEL; d++)
                c->res1[b][t][d] = c->emb[b][t][d] + O_T[d][t];
    }

    // 7. RMS norm (pre-FFN) for all batch items
    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];
        for (int t = 0; t < slen; t++)
            rms_norm(c->norm2[b][t], c->res1[b][t], D_MODEL, &c->norm2_rms[b][t]);
    }

    // 8. FFN for all batch items (Wf1_T, Wf2_T stay hot)
    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];
        float N2_T[D_MODEL][MAX_SEQ];
        for (int t = 0; t < MAX_SEQ; t++)
            for (int d = 0; d < D_MODEL; d++)
                N2_T[d][t] = c->norm2[b][t][d];

        float H[D_FF][MAX_SEQ];
        sme2_ffn_expand((float*)H, (float*)N2_T);

        for (int t = 0; t < slen; t++) {
            for (int j = 0; j < D_FF; j++) {
                c->ff_pre_relu[b][t][j] = H[j][t];
                float v = H[j][t] > 0.0f ? H[j][t] : 0.0f;
                c->ff_hidden[b][t][j] = v;
                H[j][t] = v;
            }
        }

        float Y[D_MODEL][MAX_SEQ];
        sme2_ffn_contract((float*)Y, (float*)H);

        for (int t = 0; t < slen; t++)
            for (int d = 0; d < D_MODEL; d++)
                c->res2[b][t][d] = c->res1[b][t][d] + Y[d][t];
    }

    // 9. LM head + softmax for all batch items (Wlm stays hot)
    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];
        int t = 0;
        for (; t + 1 < slen; t += 2) {
            linear_2(c->logits[b][t], c->logits[b][t+1], (float*)m->Wlm, c->res2[b][t], c->res2[b][t+1], VOCAB, D_MODEL);
            softmax(c->probs[b][t], c->logits[b][t], VOCAB);
            softmax(c->probs[b][t+1], c->logits[b][t+1], VOCAB);
        }
        if (t < slen) {
            linear(c->logits[b][t], (float*)m->Wlm, c->res2[b][t], VOCAB, D_MODEL);
            softmax(c->probs[b][t], c->logits[b][t], VOCAB);
        }
    }

#else
    // Batch-sequential for non-SME2 (data fits L1 at small D_MODEL)
    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];

        // 1-3. Embeddings + RMS norms (element-wise, not matmuls)
        for (int t = 0; t < slen; t++) {
            int tok = c->tokens[b][t];
            for (int d = 0; d < D_MODEL; d++)
                c->raw_emb[b][t][d] = m->tok_emb[tok][d] + m->pos_emb[t][d];
            rms_norm(c->emb[b][t], c->raw_emb[b][t], D_MODEL, &c->emb_rms[b][t]);
            rms_norm(c->norm1[b][t], c->emb[b][t], D_MODEL, &c->norm1_rms[b][t]);
        }

        // 4. QKV projections
        {
            int t = 0;
            for (; t + 1 < slen; t += 2) {
                linear_2(c->Q[b][t], c->Q[b][t+1], (float*)m->Wq, c->norm1[b][t], c->norm1[b][t+1], D_MODEL, D_MODEL);
                linear_2(c->K[b][t], c->K[b][t+1], (float*)m->Wk, c->norm1[b][t], c->norm1[b][t+1], D_MODEL, D_MODEL);
                linear_2(c->V[b][t], c->V[b][t+1], (float*)m->Wv, c->norm1[b][t], c->norm1[b][t+1], D_MODEL, D_MODEL);
            }
            if (t < slen) {
                linear(c->Q[b][t], (float*)m->Wq, c->norm1[b][t], D_MODEL, D_MODEL);
                linear(c->K[b][t], (float*)m->Wk, c->norm1[b][t], D_MODEL, D_MODEL);
                linear(c->V[b][t], (float*)m->Wv, c->norm1[b][t], D_MODEL, D_MODEL);
            }
        }

        // 5. Multi-head causal attention (fused: QK dots + softmax + V weighted sum)
        for (int h = 0; h < N_HEADS; h++) {
            int hoff = h * HEAD_DIM;
            for (int t = 0; t < slen; t++) {
                float max_score = -1e30f;
#if HEAD_DIM == 4
                float32x4_t qt = vld1q_f32(&c->Q[b][t][hoff]);
                for (int s = 0; s <= t; s++) {
                    float32x4_t ks = vld1q_f32(&c->K[b][s][hoff]);
                    float score = vaddvq_f32(vmulq_f32(qt, ks)) * INV_SQRT_HD;
                    c->attn_scores[b][h][t][s] = score;
                    if (score > max_score) max_score = score;
                }
                float sum = 0.0f;
                float32x4_t vacc = vdupq_n_f32(0.0f);
                for (int s = 0; s <= t; s++) {
                    float e = fast_expf_s(c->attn_scores[b][h][t][s] - max_score);
                    c->attn_scores[b][h][t][s] = e;
                    sum += e;
                    vacc = vfmaq_n_f32(vacc, vld1q_f32(&c->V[b][s][hoff]), e);
                }
                float inv_sum = 1.0f / sum;
                for (int s = 0; s <= t; s++)
                    c->attn_scores[b][h][t][s] *= inv_sum;
                vst1q_f32(&c->attn_out[b][t][hoff], vmulq_n_f32(vacc, inv_sum));
#else
                for (int s = 0; s <= t; s++) {
                    float score = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++)
                        score += c->Q[b][t][hoff + d] * c->K[b][s][hoff + d];
                    score *= INV_SQRT_HD;
                    c->attn_scores[b][h][t][s] = score;
                    if (score > max_score) max_score = score;
                }
                float sum = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) c->attn_out[b][t][hoff + d] = 0.0f;
                for (int s = 0; s <= t; s++) {
                    float e = fast_expf_s(c->attn_scores[b][h][t][s] - max_score);
                    c->attn_scores[b][h][t][s] = e;
                    sum += e;
                    for (int d = 0; d < HEAD_DIM; d++)
                        c->attn_out[b][t][hoff + d] += e * c->V[b][s][hoff + d];
                }
                float inv_sum = 1.0f / sum;
                for (int s = 0; s <= t; s++)
                    c->attn_scores[b][h][t][s] *= inv_sum;
                for (int d = 0; d < HEAD_DIM; d++)
                    c->attn_out[b][t][hoff + d] *= inv_sum;
#endif
            }
        }

        // 6. O projection + residual
        {
            int t = 0;
            for (; t + 1 < slen; t += 2)
                linear_residual_2(c->res1[b][t], c->emb[b][t],
                                  c->res1[b][t+1], c->emb[b][t+1],
                                  (float*)m->Wo, c->attn_out[b][t], c->attn_out[b][t+1], D_MODEL, D_MODEL);
            if (t < slen)
                linear_residual(c->res1[b][t], c->emb[b][t], (float*)m->Wo, c->attn_out[b][t], D_MODEL, D_MODEL);
        }

        // 7. RMS norm (pre-FFN)
        for (int t = 0; t < slen; t++)
            rms_norm(c->norm2[b][t], c->res1[b][t], D_MODEL, &c->norm2_rms[b][t]);

        // 8. FFN: expand + ReLU + contract + residual
        {
            int t = 0;
            for (; t + 1 < slen; t += 2) {
                linear_relu_2(c->ff_pre_relu[b][t], c->ff_hidden[b][t],
                              c->ff_pre_relu[b][t+1], c->ff_hidden[b][t+1],
                              (float*)m->Wf1, c->norm2[b][t], c->norm2[b][t+1], D_FF, D_MODEL);
                linear_residual_2(c->res2[b][t], c->res1[b][t],
                                  c->res2[b][t+1], c->res1[b][t+1],
                                  (float*)m->Wf2, c->ff_hidden[b][t], c->ff_hidden[b][t+1], D_MODEL, D_FF);
            }
            if (t < slen) {
                linear_relu(c->ff_pre_relu[b][t], c->ff_hidden[b][t], (float*)m->Wf1, c->norm2[b][t], D_FF, D_MODEL);
                linear_residual(c->res2[b][t], c->res1[b][t], (float*)m->Wf2, c->ff_hidden[b][t], D_MODEL, D_FF);
            }
        }

        // 9. LM head + softmax
        {
            int t = 0;
            for (; t + 1 < slen; t += 2) {
                linear_2(c->logits[b][t], c->logits[b][t+1], (float*)m->Wlm, c->res2[b][t], c->res2[b][t+1], VOCAB, D_MODEL);
                softmax(c->probs[b][t], c->logits[b][t], VOCAB);
                softmax(c->probs[b][t+1], c->logits[b][t+1], VOCAB);
            }
            if (t < slen) {
                linear(c->logits[b][t], (float*)m->Wlm, c->res2[b][t], VOCAB, D_MODEL);
                softmax(c->probs[b][t], c->logits[b][t], VOCAB);
            }
        }
    }
#endif

    // Cross-entropy loss: predict next token for positions 0..slen-2
    // AttoGPT: loss = sum(-log(softmax(logits)[target])) / n
    float total_loss = 0.0f;
    int total_count = 0;
    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];
        for (int t = 0; t < slen - 1; t++) {
            int target = c->tokens[b][t + 1];
            float p = c->probs[b][t][target];
            if (p < 1e-10f) p = 1e-10f;
            total_loss -= logf(p);
            total_count++;
        }
    }
    c->loss = total_loss / (float)total_count;
    return c->loss;
}

// ============================================================
// Backward pass
// ============================================================
static void backward(Model *m, Cache *c, Grads *g) {
    memset(g, 0, sizeof(Grads));
#if !USE_SME2
    sync_transposed_weights(m);
#endif

    int total_count = 0;
    for (int b = 0; b < BATCH; b++)
        total_count += c->seq_lens[b] - 1;
    float inv_count = 1.0f / (float)total_count;

    for (int b = 0; b < BATCH; b++) {
        int slen = c->seq_lens[b];

        float dQ[MAX_SEQ][D_MODEL];
        float dK[MAX_SEQ][D_MODEL];
        float dV[MAX_SEQ][D_MODEL];
        memset(dQ, 0, sizeof(dQ));
        memset(dK, 0, sizeof(dK));
        memset(dV, 0, sizeof(dV));

        // Per-position intermediates for batched weight grads (SME2 only)
#if USE_SME2
        float do_proj_all[MAX_SEQ][D_MODEL];
        float dff_out_all[MAX_SEQ][D_MODEL];
        float dff_hidden_all[MAX_SEQ][D_FF];
        float dx_all[MAX_SEQ][D_MODEL];
#endif

#if USE_SME2
        // ---- SME2 backward: batched FMOPA for input grads ----
        {
            // Stage A: LM head backward for all positions
            float dx_T[D_MODEL][MAX_SEQ];
            memset(dx_T, 0, sizeof(dx_T));
            for (int t = 0; t < MAX_SEQ; t++) {
                float dlogits[VOCAB];
                memset(dlogits, 0, sizeof(dlogits));
                if (t < slen - 1) {
                    int target = c->tokens[b][t + 1];
                    for (int v = 0; v < VOCAB; v++)
                        dlogits[v] = c->probs[b][t][v] * inv_count;
                    dlogits[target] -= inv_count;
                }
                for (int v = 0; v < VOCAB; v++) {
                    if (dlogits[v] == 0.0f) continue;
                    for (int d = 0; d < D_MODEL; d++) {
                        g->Wlm[v][d] += dlogits[v] * c->res2[b][t][d];
                        dx_T[d][t] += m->Wlm[v][d] * dlogits[v];
                    }
                }
            }

            // dff_out = dx before norm2 update
            for (int t = 0; t < MAX_SEQ; t++)
                for (int d = 0; d < D_MODEL; d++)
                    dff_out_all[t][d] = dx_T[d][t];

            // Stage B: dff_hidden = Wf2^T @ dx (batched FMOPA)
            float dff_hidden_T[D_FF][MAX_SEQ];
            sme2_matmul((float*)dff_hidden_T, (float*)m->Wf2, (float*)dx_T, D_FF, D_MODEL);

            // Stage C: ReLU backward + store dff_hidden_all
            for (int t = 0; t < MAX_SEQ; t++)
                for (int j = 0; j < D_FF; j++) {
                    if (c->ff_pre_relu[b][t][j] <= 0.0f)
                        dff_hidden_T[j][t] = 0.0f;
                    dff_hidden_all[t][j] = dff_hidden_T[j][t];
                }

            // Stage D: dnorm2 = Wf1^T @ dff_hidden (batched FMOPA)
            float dnorm2_T[D_MODEL][MAX_SEQ];
            sme2_matmul((float*)dnorm2_T, (float*)m->Wf1, (float*)dff_hidden_T, D_MODEL, D_FF);

            // Stage E: RMS norm2 backward, update dx
            for (int t = 0; t < slen; t++) {
                float rms = c->norm2_rms[b][t];
                float inv_rms = 1.0f / rms;
                float dot = 0.0f;
                for (int d = 0; d < D_MODEL; d++)
                    dot += dnorm2_T[d][t] * c->res1[b][t][d];
                dot /= (rms * rms * D_MODEL);
                for (int d = 0; d < D_MODEL; d++)
                    dx_T[d][t] += dnorm2_T[d][t] * inv_rms - c->res1[b][t][d] * dot;
            }

            // dx_all and do_proj_all from updated dx
            for (int t = 0; t < MAX_SEQ; t++)
                for (int d = 0; d < D_MODEL; d++) {
                    dx_all[t][d] = dx_T[d][t];
                    do_proj_all[t][d] = dx_T[d][t];
                }

            // Stage F: dattn_out = Wo^T @ dx (batched FMOPA, using updated dx)
            float dattn_out_T[D_MODEL][MAX_SEQ];
            sme2_matmul((float*)dattn_out_T, (float*)m->Wo, (float*)dx_T, D_MODEL, D_MODEL);

            // Stage G: Attention backward
            for (int t = slen - 1; t >= 0; t--) {
                for (int h = 0; h < N_HEADS; h++) {
                    int hoff = h * HEAD_DIM;
                    float dattn_scores_h[MAX_SEQ];
                    memset(dattn_scores_h, 0, sizeof(dattn_scores_h));
#if HEAD_DIM == 4
                    // Gather dattn_out from transposed layout
                    float dao_tmp[4] = {dattn_out_T[hoff][t], dattn_out_T[hoff+1][t],
                                        dattn_out_T[hoff+2][t], dattn_out_T[hoff+3][t]};
                    float32x4_t dao = vld1q_f32(dao_tmp);
                    for (int s = 0; s <= t; s++) {
                        float32x4_t vs = vld1q_f32(&c->V[b][s][hoff]);
                        dattn_scores_h[s] = vaddvq_f32(vmulq_f32(dao, vs));
                        float32x4_t dv_old = vld1q_f32(&dV[s][hoff]);
                        vst1q_f32(&dV[s][hoff], vfmaq_n_f32(dv_old, dao, c->attn_scores[b][h][t][s]));
                    }
#else
                    for (int s = 0; s <= t; s++) {
                        float ds = 0.0f;
                        for (int d = 0; d < HEAD_DIM; d++) {
                            ds += dattn_out_T[hoff + d][t] * c->V[b][s][hoff + d];
                            dV[s][hoff + d] += c->attn_scores[b][h][t][s] * dattn_out_T[hoff + d][t];
                        }
                        dattn_scores_h[s] = ds;
                    }
#endif
                    float wsum = 0.0f;
                    for (int s = 0; s <= t; s++)
                        wsum += c->attn_scores[b][h][t][s] * dattn_scores_h[s];
                    float dscore_pre[MAX_SEQ];
                    for (int s = 0; s <= t; s++)
                        dscore_pre[s] = c->attn_scores[b][h][t][s] * (dattn_scores_h[s] - wsum);
#if HEAD_DIM == 4
                    float32x4_t qt_scaled = vmulq_n_f32(vld1q_f32(&c->Q[b][t][hoff]), INV_SQRT_HD);
                    for (int s = 0; s <= t; s++) {
                        float32x4_t ks = vld1q_f32(&c->K[b][s][hoff]);
                        float32x4_t dq_old = vld1q_f32(&dQ[t][hoff]);
                        vst1q_f32(&dQ[t][hoff], vfmaq_n_f32(dq_old, ks, dscore_pre[s] * INV_SQRT_HD));
                        float32x4_t dk_old = vld1q_f32(&dK[s][hoff]);
                        vst1q_f32(&dK[s][hoff], vfmaq_n_f32(dk_old, qt_scaled, dscore_pre[s]));
                    }
#else
                    float inv_sqrt_hd = INV_SQRT_HD;
                    for (int s = 0; s <= t; s++) {
                        for (int d = 0; d < HEAD_DIM; d++) {
                            dQ[t][hoff + d] += dscore_pre[s] * c->K[b][s][hoff + d] * inv_sqrt_hd;
                            dK[s][hoff + d] += dscore_pre[s] * c->Q[b][t][hoff + d] * inv_sqrt_hd;
                        }
                    }
#endif
                }
            }

            // ---- Batched weight gradients (FMOPA outer products) ----
            float dW_tmp[D_MODEL][D_MODEL];
            sme2_outer_sum((float*)dW_tmp, (float*)dQ, (float*)(c->norm1[b]), D_MODEL, D_MODEL);
            for (int d = 0; d < D_MODEL; d++)
                for (int i = 0; i < D_MODEL; i++)
                    g->Wq[d][i] += dW_tmp[d][i];

            sme2_outer_sum((float*)dW_tmp, (float*)dK, (float*)(c->norm1[b]), D_MODEL, D_MODEL);
            for (int d = 0; d < D_MODEL; d++)
                for (int i = 0; i < D_MODEL; i++)
                    g->Wk[d][i] += dW_tmp[d][i];

            sme2_outer_sum((float*)dW_tmp, (float*)dV, (float*)(c->norm1[b]), D_MODEL, D_MODEL);
            for (int d = 0; d < D_MODEL; d++)
                for (int i = 0; i < D_MODEL; i++)
                    g->Wv[d][i] += dW_tmp[d][i];

            sme2_outer_sum((float*)dW_tmp, (float*)do_proj_all, (float*)(c->attn_out[b]), D_MODEL, D_MODEL);
            for (int d = 0; d < D_MODEL; d++)
                for (int i = 0; i < D_MODEL; i++)
                    g->Wo[d][i] += dW_tmp[d][i];

            float dWf2_tmp[D_MODEL][D_FF];
            sme2_outer_sum((float*)dWf2_tmp, (float*)dff_out_all, (float*)(c->ff_hidden[b]), D_MODEL, D_FF);
            for (int d = 0; d < D_MODEL; d++)
                for (int j = 0; j < D_FF; j++)
                    g->Wf2[d][j] += dWf2_tmp[d][j];

            float dWf1_tmp[D_FF][D_MODEL];
            sme2_outer_sum((float*)dWf1_tmp, (float*)dff_hidden_all, (float*)(c->norm2[b]), D_FF, D_MODEL);
            for (int j = 0; j < D_FF; j++)
                for (int d = 0; d < D_MODEL; d++)
                    g->Wf1[j][d] += dWf1_tmp[j][d];

            // QKV input grads (batched matmul)
            float dQ_T[D_MODEL][MAX_SEQ], dK_T[D_MODEL][MAX_SEQ], dV_T[D_MODEL][MAX_SEQ];
            for (int t = 0; t < MAX_SEQ; t++)
                for (int d = 0; d < D_MODEL; d++) {
                    dQ_T[d][t] = dQ[t][d];
                    dK_T[d][t] = dK[t][d];
                    dV_T[d][t] = dV[t][d];
                }

            float dnorm1_T[D_MODEL][MAX_SEQ];
            sme2_qkv_input_grad((float*)dnorm1_T, (float*)dQ_T, (float*)dK_T, (float*)dV_T);

            // RMS norm backward + embedding backward
            for (int t = 0; t < slen; t++) {
                float dnorm1[D_MODEL];
                for (int d = 0; d < D_MODEL; d++)
                    dnorm1[d] = dnorm1_T[d][t];

                float demb[D_MODEL];
                {
                    float rms = c->norm1_rms[b][t];
                    float inv_rms = 1.0f / rms;
                    float dot = 0.0f;
                    for (int d = 0; d < D_MODEL; d++)
                        dot += dnorm1[d] * c->emb[b][t][d];
                    dot /= (rms * rms * D_MODEL);
                    for (int d = 0; d < D_MODEL; d++)
                        demb[d] = dnorm1[d] * inv_rms - c->emb[b][t][d] * dot;
                }

                for (int d = 0; d < D_MODEL; d++)
                    demb[d] += dx_all[t][d];

                float dtmp[D_MODEL];
                {
                    float rms = c->emb_rms[b][t];
                    float inv_rms = 1.0f / rms;
                    float dot = 0.0f;
                    for (int d = 0; d < D_MODEL; d++)
                        dot += demb[d] * c->raw_emb[b][t][d];
                    dot /= (rms * rms * D_MODEL);
                    for (int d = 0; d < D_MODEL; d++)
                        dtmp[d] = demb[d] * inv_rms - c->raw_emb[b][t][d] * dot;
                }

                int tok = c->tokens[b][t];
                for (int d = 0; d < D_MODEL; d++) {
                    g->tok_emb[tok][d] += dtmp[d];
                    g->pos_emb[t][d] += dtmp[d];
                }
            }
        }
#else
        // ---- Scalar backward: two-pass with register-accumulator weight grads ----
        // Pass 1: LM head + FFN backward + attention backward (sequential)
        // Pass 2: Weight grads with register accumulators (eliminates per-position load/stores)
        // Pass 3: QKV input grads + norm backward + emb backward
        {
            float dx_all[MAX_SEQ][D_MODEL];
            float dff_out_all[MAX_SEQ][D_MODEL];
            float dff_hidden_all[MAX_SEQ][D_FF];

            // --- Pass 1: input grad projections + attention backward ---
            for (int t = slen - 2; t >= 0; t--) {
                // LM head backward (including Wlm weight grads)
                float dlogits[VOCAB];
                int target = c->tokens[b][t + 1];
                for (int v = 0; v < VOCAB; v++)
                    dlogits[v] = c->probs[b][t][v] * inv_count;
                dlogits[target] -= inv_count;

                float dx[D_MODEL];
                memset(dx, 0, sizeof(dx));
                for (int v = 0; v < VOCAB; v++) {
                    float32x4_t dl_v = vdupq_n_f32(dlogits[v]);
                    for (int d = 0; d < D_MODEL; d += 4) {
                        float32x4_t r2 = vld1q_f32(&c->res2[b][t][d]);
                        vst1q_f32(&g->Wlm[v][d], vfmaq_f32(vld1q_f32(&g->Wlm[v][d]), dl_v, r2));
                        vst1q_f32(&dx[d], vfmaq_f32(vld1q_f32(&dx[d]), vld1q_f32(&m->Wlm[v][d]), dl_v));
                    }
                }

                // Save dx for Wf2 weight grads
                memcpy(dff_out_all[t], dx, sizeof(dx));

                // FFN backward: Wf2^T + ReLU mask
                float dff_hidden[D_FF];
                linear(dff_hidden, (float*)Wf2_T, dx, D_FF, D_MODEL);
                for (int j = 0; j < D_FF; j += 4) {
                    float32x4_t pre = vld1q_f32(&c->ff_pre_relu[b][t][j]);
                    uint32x4_t mask = vcgtq_f32(pre, vdupq_n_f32(0.0f));
                    float32x4_t dh = vld1q_f32(&dff_hidden[j]);
                    vst1q_f32(&dff_hidden[j], vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(dh), mask)));
                }
                memcpy(dff_hidden_all[t], dff_hidden, sizeof(dff_hidden));

                // FFN backward: Wf1^T
                float dnorm2[D_MODEL];
                linear(dnorm2, (float*)Wf1_T, dff_hidden, D_MODEL, D_FF);

                // Norm2 backward → update dx
                {
                    float rms = c->norm2_rms[b][t];
                    float inv_rms = 1.0f / rms;
                    float32x4_t dot_v = vdupq_n_f32(0.0f);
                    for (int d = 0; d < D_MODEL; d += 4)
                        dot_v = vfmaq_f32(dot_v, vld1q_f32(&dnorm2[d]), vld1q_f32(&c->res1[b][t][d]));
                    float dot = vaddvq_f32(dot_v) / (rms * rms * D_MODEL);
                    float32x4_t inv_rms_v = vdupq_n_f32(inv_rms);
                    float32x4_t dot_s = vdupq_n_f32(dot);
                    for (int d = 0; d < D_MODEL; d += 4) {
                        float32x4_t dn = vld1q_f32(&dnorm2[d]);
                        float32x4_t r1 = vld1q_f32(&c->res1[b][t][d]);
                        float32x4_t dxv = vld1q_f32(&dx[d]);
                        vst1q_f32(&dx[d], vfmsq_f32(vfmaq_f32(dxv, dn, inv_rms_v), r1, dot_s));
                    }
                }
                memcpy(dx_all[t], dx, sizeof(dx));

                // O projection backward: Wo^T
                float dattn_out[D_MODEL];
                linear(dattn_out, (float*)Wo_T, dx, D_MODEL, D_MODEL);

                // Attention backward
                for (int h = 0; h < N_HEADS; h++) {
                    int hoff = h * HEAD_DIM;
                    float dattn_scores_h[MAX_SEQ];
                    memset(dattn_scores_h, 0, sizeof(dattn_scores_h));
#if HEAD_DIM == 4
                    float32x4_t dao = vld1q_f32(&dattn_out[hoff]);
                    for (int s = 0; s <= t; s++) {
                        float32x4_t vs = vld1q_f32(&c->V[b][s][hoff]);
                        dattn_scores_h[s] = vaddvq_f32(vmulq_f32(dao, vs));
                        float32x4_t dv_old = vld1q_f32(&dV[s][hoff]);
                        vst1q_f32(&dV[s][hoff], vfmaq_n_f32(dv_old, dao, c->attn_scores[b][h][t][s]));
                    }
#else
                    for (int s = 0; s <= t; s++) {
                        float ds = 0.0f;
                        for (int d = 0; d < HEAD_DIM; d++) {
                            ds += dattn_out[hoff + d] * c->V[b][s][hoff + d];
                            dV[s][hoff + d] += c->attn_scores[b][h][t][s] * dattn_out[hoff + d];
                        }
                        dattn_scores_h[s] = ds;
                    }
#endif
                    float wsum = 0.0f;
                    for (int s = 0; s <= t; s++)
                        wsum += c->attn_scores[b][h][t][s] * dattn_scores_h[s];
                    float dscore_pre[MAX_SEQ];
                    for (int s = 0; s <= t; s++)
                        dscore_pre[s] = c->attn_scores[b][h][t][s] * (dattn_scores_h[s] - wsum);
#if HEAD_DIM == 4
                    float32x4_t qt_scaled = vmulq_n_f32(vld1q_f32(&c->Q[b][t][hoff]), INV_SQRT_HD);
                    for (int s = 0; s <= t; s++) {
                        float32x4_t ks = vld1q_f32(&c->K[b][s][hoff]);
                        float32x4_t dq_old = vld1q_f32(&dQ[t][hoff]);
                        vst1q_f32(&dQ[t][hoff], vfmaq_n_f32(dq_old, ks, dscore_pre[s] * INV_SQRT_HD));
                        float32x4_t dk_old = vld1q_f32(&dK[s][hoff]);
                        vst1q_f32(&dK[s][hoff], vfmaq_n_f32(dk_old, qt_scaled, dscore_pre[s]));
                    }
#else
                    float inv_sqrt_hd = INV_SQRT_HD;
                    for (int s = 0; s <= t; s++) {
                        for (int d = 0; d < HEAD_DIM; d++) {
                            dQ[t][hoff + d] += dscore_pre[s] * c->K[b][s][hoff + d] * inv_sqrt_hd;
                            dK[s][hoff + d] += dscore_pre[s] * c->Q[b][t][hoff + d] * inv_sqrt_hd;
                        }
                    }
#endif
                }
            }

            // --- Pass 2: register-accumulator weight grads ---
            // QKV weight grads: accumulate across all positions in registers
            for (int d = 0; d < D_MODEL; d++) {
                float32x4_t aq[D_MODEL/4], ak[D_MODEL/4], av[D_MODEL/4];
                for (int k = 0; k < D_MODEL/4; k++) {
                    aq[k] = vld1q_f32(&g->Wq[d][k*4]);
                    ak[k] = vld1q_f32(&g->Wk[d][k*4]);
                    av[k] = vld1q_f32(&g->Wv[d][k*4]);
                }
                for (int t = 0; t < slen - 1; t++) {
                    float32x4_t dq = vdupq_n_f32(dQ[t][d]);
                    float32x4_t dk = vdupq_n_f32(dK[t][d]);
                    float32x4_t dv = vdupq_n_f32(dV[t][d]);
                    for (int k = 0; k < D_MODEL/4; k++) {
                        float32x4_t n1 = vld1q_f32(&c->norm1[b][t][k*4]);
                        aq[k] = vfmaq_f32(aq[k], dq, n1);
                        ak[k] = vfmaq_f32(ak[k], dk, n1);
                        av[k] = vfmaq_f32(av[k], dv, n1);
                    }
                }
                for (int k = 0; k < D_MODEL/4; k++) {
                    vst1q_f32(&g->Wq[d][k*4], aq[k]);
                    vst1q_f32(&g->Wk[d][k*4], ak[k]);
                    vst1q_f32(&g->Wv[d][k*4], av[k]);
                }
            }

            // Wo weight grads
            for (int d = 0; d < D_MODEL; d++) {
                float32x4_t aw[D_MODEL/4];
                for (int k = 0; k < D_MODEL/4; k++)
                    aw[k] = vld1q_f32(&g->Wo[d][k*4]);
                for (int t = 0; t < slen - 1; t++) {
                    float32x4_t dx_d = vdupq_n_f32(dx_all[t][d]);
                    for (int k = 0; k < D_MODEL/4; k++)
                        aw[k] = vfmaq_f32(aw[k], dx_d, vld1q_f32(&c->attn_out[b][t][k*4]));
                }
                for (int k = 0; k < D_MODEL/4; k++)
                    vst1q_f32(&g->Wo[d][k*4], aw[k]);
            }

            // Wf2 weight grads: tile D_FF in groups of 16
            for (int d = 0; d < D_MODEL; d++) {
                for (int j0 = 0; j0 < D_FF; j0 += 16) {
                    float32x4_t a0 = vld1q_f32(&g->Wf2[d][j0]);
                    float32x4_t a1 = vld1q_f32(&g->Wf2[d][j0+4]);
                    float32x4_t a2 = vld1q_f32(&g->Wf2[d][j0+8]);
                    float32x4_t a3 = vld1q_f32(&g->Wf2[d][j0+12]);
                    for (int t = 0; t < slen - 1; t++) {
                        float32x4_t df = vdupq_n_f32(dff_out_all[t][d]);
                        a0 = vfmaq_f32(a0, df, vld1q_f32(&c->ff_hidden[b][t][j0]));
                        a1 = vfmaq_f32(a1, df, vld1q_f32(&c->ff_hidden[b][t][j0+4]));
                        a2 = vfmaq_f32(a2, df, vld1q_f32(&c->ff_hidden[b][t][j0+8]));
                        a3 = vfmaq_f32(a3, df, vld1q_f32(&c->ff_hidden[b][t][j0+12]));
                    }
                    vst1q_f32(&g->Wf2[d][j0], a0);
                    vst1q_f32(&g->Wf2[d][j0+4], a1);
                    vst1q_f32(&g->Wf2[d][j0+8], a2);
                    vst1q_f32(&g->Wf2[d][j0+12], a3);
                }
            }

            // Wf1 weight grads
            for (int j = 0; j < D_FF; j++) {
                float32x4_t aw[D_MODEL/4];
                for (int k = 0; k < D_MODEL/4; k++)
                    aw[k] = vld1q_f32(&g->Wf1[j][k*4]);
                for (int t = 0; t < slen - 1; t++) {
                    float32x4_t dh = vdupq_n_f32(dff_hidden_all[t][j]);
                    for (int k = 0; k < D_MODEL/4; k++)
                        aw[k] = vfmaq_f32(aw[k], dh, vld1q_f32(&c->norm2[b][t][k*4]));
                }
                for (int k = 0; k < D_MODEL/4; k++)
                    vst1q_f32(&g->Wf1[j][k*4], aw[k]);
            }

            // --- Pass 3: QKV input grads + norm backward + emb backward ---
            for (int t = slen - 2; t >= 0; t--) {
                float dnorm1[D_MODEL];
                memset(dnorm1, 0, sizeof(dnorm1));
#if HEAD_DIM == 4
                for (int d = 0; d < D_MODEL; d++) {
                    float32x4_t dq_d = vdupq_n_f32(dQ[t][d]);
                    float32x4_t dk_d = vdupq_n_f32(dK[t][d]);
                    float32x4_t dv_d = vdupq_n_f32(dV[t][d]);
                    for (int i = 0; i < D_MODEL; i += 4) {
                        float32x4_t dn = vld1q_f32(&dnorm1[i]);
                        dn = vfmaq_f32(dn, vld1q_f32(&m->Wq[d][i]), dq_d);
                        dn = vfmaq_f32(dn, vld1q_f32(&m->Wk[d][i]), dk_d);
                        dn = vfmaq_f32(dn, vld1q_f32(&m->Wv[d][i]), dv_d);
                        vst1q_f32(&dnorm1[i], dn);
                    }
                }
#else
                for (int d = 0; d < D_MODEL; d++) {
                    for (int i = 0; i < D_MODEL; i++) {
                        dnorm1[i] += m->Wq[d][i] * dQ[t][d]
                                  +  m->Wk[d][i] * dK[t][d]
                                  +  m->Wv[d][i] * dV[t][d];
                    }
                }
#endif

                // Norm1 backward
                float demb[D_MODEL];
                {
                    float rms = c->norm1_rms[b][t];
                    float inv_rms = 1.0f / rms;
                    float32x4_t dot_v = vdupq_n_f32(0.0f);
                    for (int d = 0; d < D_MODEL; d += 4)
                        dot_v = vfmaq_f32(dot_v, vld1q_f32(&dnorm1[d]), vld1q_f32(&c->emb[b][t][d]));
                    float dot = vaddvq_f32(dot_v) / (rms * rms * D_MODEL);
                    float32x4_t inv_rms_v = vdupq_n_f32(inv_rms);
                    float32x4_t dot_s = vdupq_n_f32(dot);
                    for (int d = 0; d < D_MODEL; d += 4) {
                        float32x4_t dn = vld1q_f32(&dnorm1[d]);
                        float32x4_t e = vld1q_f32(&c->emb[b][t][d]);
                        vst1q_f32(&demb[d], vfmsq_f32(vmulq_f32(dn, inv_rms_v), e, dot_s));
                    }
                }

                // Residual connection
                for (int d = 0; d < D_MODEL; d += 4)
                    vst1q_f32(&demb[d], vaddq_f32(vld1q_f32(&demb[d]), vld1q_f32(&dx_all[t][d])));

                // Emb RMS norm backward
                float dtmp[D_MODEL];
                {
                    float rms = c->emb_rms[b][t];
                    float inv_rms = 1.0f / rms;
                    float32x4_t dot_v = vdupq_n_f32(0.0f);
                    for (int d = 0; d < D_MODEL; d += 4)
                        dot_v = vfmaq_f32(dot_v, vld1q_f32(&demb[d]), vld1q_f32(&c->raw_emb[b][t][d]));
                    float dot = vaddvq_f32(dot_v) / (rms * rms * D_MODEL);
                    float32x4_t inv_rms_v = vdupq_n_f32(inv_rms);
                    float32x4_t dot_s = vdupq_n_f32(dot);
                    for (int d = 0; d < D_MODEL; d += 4) {
                        float32x4_t de = vld1q_f32(&demb[d]);
                        float32x4_t re = vld1q_f32(&c->raw_emb[b][t][d]);
                        vst1q_f32(&dtmp[d], vfmsq_f32(vmulq_f32(de, inv_rms_v), re, dot_s));
                    }
                }

                int tok = c->tokens[b][t];
                for (int d = 0; d < D_MODEL; d += 4) {
                    float32x4_t dt = vld1q_f32(&dtmp[d]);
                    vst1q_f32(&g->tok_emb[tok][d], vaddq_f32(vld1q_f32(&g->tok_emb[tok][d]), dt));
                    vst1q_f32(&g->pos_emb[t][d], vaddq_f32(vld1q_f32(&g->pos_emb[t][d]), dt));
                }
            }
        }
#endif
    }
}

// ============================================================
// Adam optimizer
// ============================================================
static void adam_update(float *param, float *grad, float *m1, float *m2,
                        int n, int step, float lr) {
    float beta1_corr = 1.0f - powf(BETA1, (float)step);
    float beta2_corr = 1.0f - powf(BETA2, (float)step);

    float32x4_t vb1 = vdupq_n_f32(BETA1);
    float32x4_t vb1c = vdupq_n_f32(1.0f - BETA1);
    float32x4_t vb2 = vdupq_n_f32(BETA2);
    float32x4_t vb2c = vdupq_n_f32(1.0f - BETA2);
    float32x4_t vlr_b1 = vdupq_n_f32(lr / beta1_corr);
    float32x4_t vinv_b2 = vdupq_n_f32(1.0f / beta2_corr);
    float32x4_t veps = vdupq_n_f32(EPS);

    int i;
    for (i = 0; i + 3 < n; i += 4) {
        float32x4_t g = vld1q_f32(grad + i);
        float32x4_t vm1 = vfmaq_f32(vmulq_f32(vb1, vld1q_f32(m1 + i)), vb1c, g);
        float32x4_t vm2 = vfmaq_f32(vmulq_f32(vb2, vld1q_f32(m2 + i)), vb2c, vmulq_f32(g, g));
        vst1q_f32(m1 + i, vm1);
        vst1q_f32(m2 + i, vm2);
        // param -= lr/b1_corr * m1 / (sqrt(m2/b2_corr) + eps)
        float32x4_t m2h = vmulq_f32(vm2, vinv_b2);
        float32x4_t denom = vaddq_f32(vsqrtq_f32(m2h), veps);
        float32x4_t p = vld1q_f32(param + i);
        vst1q_f32(param + i, vfmsq_f32(p, vlr_b1, vdivq_f32(vm1, denom)));
    }
    for (; i < n; i++) {
        m1[i] = BETA1 * m1[i] + (1.0f - BETA1) * grad[i];
        m2[i] = BETA2 * m2[i] + (1.0f - BETA2) * grad[i] * grad[i];
        float m1_hat = m1[i] / beta1_corr;
        float m2_hat = m2[i] / beta2_corr;
        param[i] -= lr * m1_hat / (sqrtf(m2_hat) + EPS);
    }
}

// ============================================================
// Name generation
// ============================================================
static void generate(Model *m, int n_names) {
    float emb_buf[MAX_SEQ][D_MODEL];
    float norm_buf[D_MODEL];
    float Q_t[D_MODEL], K_all[MAX_SEQ][D_MODEL], V_all[MAX_SEQ][D_MODEL];
    float attn_out[D_MODEL];
    float o_proj_t[D_MODEL];
    float res1[D_MODEL];
    float norm2_t[D_MODEL];
    float ff_h[D_FF], ff_out_t[D_MODEL];
    float res2[D_MODEL];
    float logits_t[VOCAB], probs_t[VOCAB];

    for (int n = 0; n < n_names; n++) {
        int tokens[MAX_SEQ];
        memset(tokens, 0, sizeof(tokens));
        tokens[0] = BOUNDARY; // start with boundary token

        printf("  ");

        for (int t = 1; t < MAX_SEQ; t++) {
            // Build embeddings for all positions 0..t-1
            for (int p = 0; p < t; p++) {
                float raw[D_MODEL];
                for (int d = 0; d < D_MODEL; d++)
                    raw[d] = m->tok_emb[tokens[p]][d] + m->pos_emb[p][d];
                rms_norm(emb_buf[p], raw, D_MODEL, NULL);
            }

            // Pre-attn norm + QKV for all positions
            for (int p = 0; p < t; p++) {
                float norm_p[D_MODEL];
                rms_norm(norm_p, emb_buf[p], D_MODEL, NULL);
                linear(K_all[p], (float*)m->Wk, norm_p, D_MODEL, D_MODEL);
                linear(V_all[p], (float*)m->Wv, norm_p, D_MODEL, D_MODEL);
                if (p == t - 1) {
                    linear(Q_t, (float*)m->Wq, norm_p, D_MODEL, D_MODEL);
                }
            }

            // Multi-head attention at position t-1
            memset(attn_out, 0, sizeof(attn_out));
            for (int h = 0; h < N_HEADS; h++) {
                int hoff = h * HEAD_DIM;
                float scores[MAX_SEQ];
#if HEAD_DIM == 4
                float32x4_t qt = vld1q_f32(&Q_t[hoff]);
                for (int s = 0; s < t; s++) {
                    float32x4_t ks = vld1q_f32(&K_all[s][hoff]);
                    scores[s] = vaddvq_f32(vmulq_f32(qt, ks)) * INV_SQRT_HD;
                }
#else
                for (int s = 0; s < t; s++) {
                    float sc = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++)
                        sc += Q_t[hoff + d] * K_all[s][hoff + d];
                    scores[s] = sc * INV_SQRT_HD;
                }
#endif
                float max_s = scores[0];
                for (int s = 1; s < t; s++) if (scores[s] > max_s) max_s = scores[s];
                float sum_s = 0.0f;
                for (int s = 0; s < t; s++) { scores[s] = expf(scores[s] - max_s); sum_s += scores[s]; }
                for (int s = 0; s < t; s++) scores[s] /= sum_s;
#if HEAD_DIM == 4
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int s = 0; s < t; s++)
                    acc = vfmaq_n_f32(acc, vld1q_f32(&V_all[s][hoff]), scores[s]);
                vst1q_f32(&attn_out[hoff], acc);
#else
                for (int d = 0; d < HEAD_DIM; d++) {
                    float v = 0.0f;
                    for (int s = 0; s < t; s++) v += scores[s] * V_all[s][hoff + d];
                    attn_out[hoff + d] = v;
                }
#endif
            }

            // O projection + residual
            linear(o_proj_t, (float*)m->Wo, attn_out, D_MODEL, D_MODEL);
            for (int d = 0; d < D_MODEL; d++)
                res1[d] = emb_buf[t-1][d] + o_proj_t[d];

            // FFN
            rms_norm(norm2_t, res1, D_MODEL, NULL);
            linear(ff_h, (float*)m->Wf1, norm2_t, D_FF, D_MODEL);
            for (int j = 0; j < D_FF; j++) ff_h[j] = ff_h[j] > 0 ? ff_h[j] : 0;
            linear(ff_out_t, (float*)m->Wf2, ff_h, D_MODEL, D_FF);
            for (int d = 0; d < D_MODEL; d++) res2[d] = res1[d] + ff_out_t[d];

            // LM head
            linear(logits_t, (float*)m->Wlm, res2, VOCAB, D_MODEL);

            // Temperature-scaled softmax (AttoGPT uses temp=0.5)
            float temp = 0.5f;
            for (int v = 0; v < VOCAB; v++) logits_t[v] /= temp;
            softmax(probs_t, logits_t, VOCAB);

            // Sample
            float r = randf();
            float cumsum = 0.0f;
            int sampled = BOUNDARY;
            for (int v = 0; v < VOCAB; v++) {
                cumsum += probs_t[v];
                if (r <= cumsum) { sampled = v; break; }
            }

            tokens[t] = sampled;
            if (sampled == BOUNDARY) break; // end of name
            printf("%c", 'a' + sampled);
        }
        printf("\n");
    }
}

// ============================================================
// Main: training loop
// ============================================================
int main(int argc, char **argv) {
    (void)argc; (void)argv;

    load_names("names.txt");
    int n_params = param_count();
    printf("FemtoGPT: Loaded %d names, %d parameters (~%.1f KB)\n",
           num_names, n_params, (float)n_params * 4.0f / 1024.0f);
    printf("Architecture: d_model=%d, n_heads=%d, d_ff=%d, vocab=%d, max_seq=%d\n",
           D_MODEL, N_HEADS, D_FF, VOCAB, MAX_SEQ);
    printf("Training: %d steps, batch=%d, lr=%.4f\n", N_STEPS, BATCH, LR_INIT);
#if USE_SME2
    printf("Backend: SME2 + auto-vectorized C\n");
#else
    printf("Backend: auto-vectorized C\n");
#endif

    Model *model = (Model*)calloc(1, sizeof(Model));
    Cache *cache = (Cache*)calloc(1, sizeof(Cache));
    Grads *grads = (Grads*)calloc(1, sizeof(Grads));
    float *adam_m1 = (float*)calloc((size_t)n_params, sizeof(float));
    float *adam_m2 = (float*)calloc((size_t)n_params, sizeof(float));

    if (!model || !cache || !grads || !adam_m1 || !adam_m2) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    init_model(model);

    sync_transposed_weights(model);

    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    // Component timing accumulators
    uint64_t t_fwd_total = 0, t_bwd_total = 0, t_adam_total = 0;

    uint64_t t_start = mach_absolute_time();

    for (int step = 1; step <= N_STEPS; step++) {
        // Sample batch
        for (int b = 0; b < BATCH; b++) {
            int idx = (int)(xorshift64() % (uint64_t)num_names);
            memcpy(cache->tokens[b], all_tokens[idx], MAX_SEQ * sizeof(int));
            cache->seq_lens[b] = all_lengths[idx];
        }

        // Linear LR decay: lr = lr_init * (1 - step/N)
        float lr = LR_INIT * (1.0f - (float)(step - 1) / (float)N_STEPS);

        uint64_t t0 = mach_absolute_time();
        float loss = forward(model, cache);
        uint64_t t1 = mach_absolute_time();
        backward(model, cache, grads);
        uint64_t t2 = mach_absolute_time();

        float *param_ptr = (float*)model;
        float *grad_ptr = (float*)grads;
        adam_update(param_ptr, grad_ptr, adam_m1, adam_m2, n_params, step, lr);
        uint64_t t3 = mach_absolute_time();

#if USE_SME2
        sync_transposed_weights(model);
#endif

        t_fwd_total += (t1 - t0);
        t_bwd_total += (t2 - t1);
        t_adam_total += (t3 - t2);

        if (step % 10 == 0 || step == 1) {
            printf("Step %3d/%d  loss=%.4f  lr=%.6f\n", step, N_STEPS, loss, lr);
        }
    }

    uint64_t t_end = mach_absolute_time();
    uint64_t elapsed_ns = (t_end - t_start) * tb.numer / tb.denom;
    double elapsed_us = (double)elapsed_ns / 1000.0;
    double elapsed_ms = elapsed_us / 1000.0;

    printf("\nTraining complete in %.2f ms (%.1f us)\n", elapsed_ms, elapsed_us);
    printf("Average step: %.2f us\n", elapsed_us / N_STEPS);

    // Component timing breakdown
    double fwd_us = (double)t_fwd_total * (double)tb.numer / (double)tb.denom / 1000.0;
    double bwd_us = (double)t_bwd_total * (double)tb.numer / (double)tb.denom / 1000.0;
    double adam_us = (double)t_adam_total * (double)tb.numer / (double)tb.denom / 1000.0;
    printf("\nTiming breakdown (total across %d steps):\n", N_STEPS);
    printf("  Forward:  %8.1f us (%5.1f us/step, %.0f%%)\n",
           fwd_us, fwd_us / N_STEPS, 100.0 * fwd_us / elapsed_us);
    printf("  Backward: %8.1f us (%5.1f us/step, %.0f%%)\n",
           bwd_us, bwd_us / N_STEPS, 100.0 * bwd_us / elapsed_us);
    printf("  Adam:     %8.1f us (%5.1f us/step, %.0f%%)\n",
           adam_us, adam_us / N_STEPS, 100.0 * adam_us / elapsed_us);

    printf("\nGenerated names:\n");
    generate(model, 10);

    free(model);
    free(cache);
    free(grads);
    free(adam_m1);
    free(adam_m2);

    return 0;
}
