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
#if D_MODEL >= 32
#define USE_SME2 1
#else
#define USE_SME2 0
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
    float  o_proj[BATCH][MAX_SEQ][D_MODEL];         // after O projection
    float  res1[BATCH][MAX_SEQ][D_MODEL];           // after first residual (attn + emb)
    float  norm2[BATCH][MAX_SEQ][D_MODEL];          // after RMS norm pre-FFN
    float  norm2_rms[BATCH][MAX_SEQ];
    float  ff_pre_relu[BATCH][MAX_SEQ][D_FF];       // FFN hidden before ReLU
    float  ff_hidden[BATCH][MAX_SEQ][D_FF];         // FFN hidden after ReLU
    float  ff_out[BATCH][MAX_SEQ][D_MODEL];         // FFN output
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
static inline __attribute__((always_inline)) void linear(float *y, const float *W, const float *x, int out_dim, int in_dim) {
    for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        const float *row = W + j * in_dim;
        for (int i = 0; i < in_dim; i++) {
            sum += row[i] * x[i];
        }
        y[j] = sum;
    }
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

// ============================================================
// SME2 FMOPA matmul functions (generalized for any D_MODEL multiple of 16)
// ============================================================
#if USE_SME2

// Transposed weight copies for FMOPA (contiguous column access).
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
    for (int b = 0; b < BATCH; b++) {
        // 1-3. Embeddings + RMS norms (element-wise, not matmuls)
        for (int t = 0; t < MAX_SEQ; t++) {
            int tok = c->tokens[b][t];
            for (int d = 0; d < D_MODEL; d++)
                c->raw_emb[b][t][d] = m->tok_emb[tok][d] + m->pos_emb[t][d];
            rms_norm(c->emb[b][t], c->raw_emb[b][t], D_MODEL, &c->emb_rms[b][t]);
            rms_norm(c->norm1[b][t], c->emb[b][t], D_MODEL, &c->norm1_rms[b][t]);
        }

        // 4. QKV projections
#if USE_SME2
        {
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
#else
        for (int t = 0; t < MAX_SEQ; t++) {
            linear(c->Q[b][t], (float*)m->Wq, c->norm1[b][t], D_MODEL, D_MODEL);
            linear(c->K[b][t], (float*)m->Wk, c->norm1[b][t], D_MODEL, D_MODEL);
            linear(c->V[b][t], (float*)m->Wv, c->norm1[b][t], D_MODEL, D_MODEL);
        }
#endif

        // 5. Multi-head causal attention
        for (int h = 0; h < N_HEADS; h++) {
            int hoff = h * HEAD_DIM;
            for (int t = 0; t < MAX_SEQ; t++) {
#if HEAD_DIM == 4
                float32x4_t qt = vld1q_f32(&c->Q[b][t][hoff]);
                for (int s = 0; s <= t; s++) {
                    float32x4_t ks = vld1q_f32(&c->K[b][s][hoff]);
                    c->attn_scores[b][h][t][s] = vaddvq_f32(vmulq_f32(qt, ks)) * INV_SQRT_HD;
                }
#else
                for (int s = 0; s <= t; s++) {
                    float score = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++)
                        score += c->Q[b][t][hoff + d] * c->K[b][s][hoff + d];
                    c->attn_scores[b][h][t][s] = score * INV_SQRT_HD;
                }
#endif
                for (int s = t + 1; s < MAX_SEQ; s++)
                    c->attn_scores[b][h][t][s] = -1e9f;
                softmax(c->attn_scores[b][h][t], c->attn_scores[b][h][t], MAX_SEQ);
#if HEAD_DIM == 4
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int s = 0; s <= t; s++)
                    acc = vfmaq_n_f32(acc, vld1q_f32(&c->V[b][s][hoff]), c->attn_scores[b][h][t][s]);
                vst1q_f32(&c->attn_out[b][t][hoff], acc);
#else
                for (int d = 0; d < HEAD_DIM; d++) {
                    float sum = 0.0f;
                    for (int s = 0; s <= t; s++)
                        sum += c->attn_scores[b][h][t][s] * c->V[b][s][hoff + d];
                    c->attn_out[b][t][hoff + d] = sum;
                }
#endif
            }
        }

        // 6. O projection + residual
#if USE_SME2
        {
            float A_T[D_MODEL][MAX_SEQ], O_T[D_MODEL][MAX_SEQ];
            for (int t = 0; t < MAX_SEQ; t++)
                for (int d = 0; d < D_MODEL; d++)
                    A_T[d][t] = c->attn_out[b][t][d];
            sme2_o_project((float*)O_T, (float*)A_T);
            for (int t = 0; t < MAX_SEQ; t++)
                for (int d = 0; d < D_MODEL; d++) {
                    c->o_proj[b][t][d] = O_T[d][t];
                    c->res1[b][t][d] = c->emb[b][t][d] + O_T[d][t];
                }
        }
#else
        for (int t = 0; t < MAX_SEQ; t++) {
            linear(c->o_proj[b][t], (float*)m->Wo, c->attn_out[b][t], D_MODEL, D_MODEL);
            for (int d = 0; d < D_MODEL; d++)
                c->res1[b][t][d] = c->emb[b][t][d] + c->o_proj[b][t][d];
        }
#endif

        // 7. RMS norm (pre-FFN)
        for (int t = 0; t < MAX_SEQ; t++)
            rms_norm(c->norm2[b][t], c->res1[b][t], D_MODEL, &c->norm2_rms[b][t]);

        // 8. FFN: expand + ReLU + contract + residual
#if USE_SME2
        {
            float N2_T[D_MODEL][MAX_SEQ];
            for (int t = 0; t < MAX_SEQ; t++)
                for (int d = 0; d < D_MODEL; d++)
                    N2_T[d][t] = c->norm2[b][t][d];

            float H[D_FF][MAX_SEQ];
            sme2_ffn_expand((float*)H, (float*)N2_T);

            for (int t = 0; t < MAX_SEQ; t++) {
                for (int j = 0; j < D_FF; j++) {
                    c->ff_pre_relu[b][t][j] = H[j][t];
                    float v = H[j][t] > 0.0f ? H[j][t] : 0.0f;
                    c->ff_hidden[b][t][j] = v;
                    H[j][t] = v;
                }
            }

            float Y[D_MODEL][MAX_SEQ];
            sme2_ffn_contract((float*)Y, (float*)H);

            for (int t = 0; t < MAX_SEQ; t++)
                for (int d = 0; d < D_MODEL; d++) {
                    c->ff_out[b][t][d] = Y[d][t];
                    c->res2[b][t][d] = c->res1[b][t][d] + Y[d][t];
                }
        }
#else
        for (int t = 0; t < MAX_SEQ; t++) {
            linear(c->ff_pre_relu[b][t], (float*)m->Wf1, c->norm2[b][t], D_FF, D_MODEL);
            for (int d = 0; d < D_FF; d++)
                c->ff_hidden[b][t][d] = c->ff_pre_relu[b][t][d] > 0.0f ? c->ff_pre_relu[b][t][d] : 0.0f;
            linear(c->ff_out[b][t], (float*)m->Wf2, c->ff_hidden[b][t], D_MODEL, D_FF);
            for (int d = 0; d < D_MODEL; d++)
                c->res2[b][t][d] = c->res1[b][t][d] + c->ff_out[b][t][d];
        }
#endif

        // 9. LM head + softmax (27x16 — too small for FMOPA)
        for (int t = 0; t < MAX_SEQ; t++) {
            linear(c->logits[b][t], (float*)m->Wlm, c->res2[b][t], VOCAB, D_MODEL);
            softmax(c->probs[b][t], c->logits[b][t], VOCAB);
        }
    }

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

        // Per-position intermediates for batched weight grads
        float do_proj_all[MAX_SEQ][D_MODEL];
        float dff_out_all[MAX_SEQ][D_MODEL];
        float dff_hidden_all[MAX_SEQ][D_FF];
        float dx_all[MAX_SEQ][D_MODEL]; // dx after FFN bwd (for residual in phase 3)

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
            for (int t = 0; t < MAX_SEQ; t++) {
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
            for (int t = MAX_SEQ - 1; t >= 0; t--) {
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
                    for (int s = 0; s < MAX_SEQ; s++)
                        wsum += c->attn_scores[b][h][t][s] * dattn_scores_h[s];
                    float dscore_pre[MAX_SEQ];
                    for (int s = 0; s < MAX_SEQ; s++)
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
            for (int t = 0; t < MAX_SEQ; t++) {
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
        // ---- Scalar backward ----
        {
            float dlogits[VOCAB];
            float dx[D_MODEL];
            float dattn_out[D_MODEL];

            // Phase 1: Per-position backward
            for (int t = MAX_SEQ - 1; t >= 0; t--) {
                memset(dlogits, 0, sizeof(dlogits));
                if (t < slen - 1) {
                    int target = c->tokens[b][t + 1];
                    for (int v = 0; v < VOCAB; v++)
                        dlogits[v] = c->probs[b][t][v] * inv_count;
                    dlogits[target] -= inv_count;
                }

                memset(dx, 0, sizeof(dx));
                for (int v = 0; v < VOCAB; v++) {
                    if (dlogits[v] == 0.0f) continue;
                    for (int d = 0; d < D_MODEL; d++) {
                        g->Wlm[v][d] += dlogits[v] * c->res2[b][t][d];
                        dx[d] += m->Wlm[v][d] * dlogits[v];
                    }
                }

                memcpy(dff_out_all[t], dx, D_MODEL * sizeof(float));

                float dff_hidden[D_FF];
                memset(dff_hidden, 0, sizeof(dff_hidden));
                for (int d = 0; d < D_MODEL; d++) {
                    if (dx[d] == 0.0f) continue;
                    for (int j = 0; j < D_FF; j++)
                        dff_hidden[j] += m->Wf2[d][j] * dx[d];
                }

                for (int j = 0; j < D_FF; j++) {
                    if (c->ff_pre_relu[b][t][j] <= 0.0f) dff_hidden[j] = 0.0f;
                }
                memcpy(dff_hidden_all[t], dff_hidden, D_FF * sizeof(float));

                float dnorm2[D_MODEL];
                memset(dnorm2, 0, sizeof(dnorm2));
                for (int j = 0; j < D_FF; j++) {
                    if (dff_hidden[j] == 0.0f) continue;
                    for (int d = 0; d < D_MODEL; d++)
                        dnorm2[d] += m->Wf1[j][d] * dff_hidden[j];
                }

                {
                    float rms = c->norm2_rms[b][t];
                    float inv_rms = 1.0f / rms;
                    float dot = 0.0f;
                    for (int d = 0; d < D_MODEL; d++)
                        dot += dnorm2[d] * c->res1[b][t][d];
                    dot /= (rms * rms * D_MODEL);
                    for (int d = 0; d < D_MODEL; d++)
                        dx[d] += dnorm2[d] * inv_rms - c->res1[b][t][d] * dot;
                }

                memcpy(dx_all[t], dx, D_MODEL * sizeof(float));

                memcpy(do_proj_all[t], dx, D_MODEL * sizeof(float));
                memset(dattn_out, 0, sizeof(dattn_out));
                for (int d = 0; d < D_MODEL; d++) {
                    if (dx[d] == 0.0f) continue;
                    for (int i = 0; i < D_MODEL; i++)
                        dattn_out[i] += m->Wo[d][i] * dx[d];
                }

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
                    for (int s = 0; s < MAX_SEQ; s++)
                        wsum += c->attn_scores[b][h][t][s] * dattn_scores_h[s];
                    float dscore_pre[MAX_SEQ];
                    for (int s = 0; s < MAX_SEQ; s++)
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

            // Phase 2: Weight grads + QKV backward + norm/emb backward
            for (int t = MAX_SEQ - 1; t >= 0; t--) {
                float dnorm1[D_MODEL];
                memset(dnorm1, 0, sizeof(dnorm1));
                for (int d = 0; d < D_MODEL; d++) {
                    for (int i = 0; i < D_MODEL; i++) {
                        g->Wq[d][i] += dQ[t][d] * c->norm1[b][t][i];
                        g->Wk[d][i] += dK[t][d] * c->norm1[b][t][i];
                        g->Wv[d][i] += dV[t][d] * c->norm1[b][t][i];
                        dnorm1[i] += m->Wq[d][i] * dQ[t][d]
                                  +  m->Wk[d][i] * dK[t][d]
                                  +  m->Wv[d][i] * dV[t][d];
                    }
                }

                for (int d = 0; d < D_MODEL; d++) {
                    for (int i = 0; i < D_MODEL; i++)
                        g->Wo[d][i] += do_proj_all[t][d] * c->attn_out[b][t][i];
                }

                for (int d = 0; d < D_MODEL; d++) {
                    if (dff_out_all[t][d] == 0.0f) continue;
                    for (int j = 0; j < D_FF; j++)
                        g->Wf2[d][j] += dff_out_all[t][d] * c->ff_hidden[b][t][j];
                }
                for (int j = 0; j < D_FF; j++) {
                    if (dff_hidden_all[t][j] == 0.0f) continue;
                    for (int d = 0; d < D_MODEL; d++)
                        g->Wf1[j][d] += dff_hidden_all[t][j] * c->norm2[b][t][d];
                }

                memset(dQ[t], 0, D_MODEL * sizeof(float));
                memset(dK[t], 0, D_MODEL * sizeof(float));
                memset(dV[t], 0, D_MODEL * sizeof(float));

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

    for (int i = 0; i < n; i++) {
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

#if USE_SME2
    sync_transposed_weights(model);
#endif

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
