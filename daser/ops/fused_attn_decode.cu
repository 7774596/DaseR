// SPDX-License-Identifier: Apache-2.0
// Split-KV decode attention over AF1 records (see fused_attn_decode.py).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "af1_common.cuh"

namespace {

using namespace af1;

constexpr int kPs = 8 * 256;  // 8 heads x 128 tokens bf16
// Record metadata is staged per chunk of pages plus one lookahead page (the
// next K record is issued one page early).
constexpr int kMetaChunk = 128;
constexpr int kMetaPages = kMetaChunk + 1;
constexpr int kRedRel = kPs;                               // float[64] max
constexpr int kRed2Rel = kRedRel + 256;                    // float[64] sum
constexpr int kMshRel = kRed2Rel + 256;                    // float[8]
constexpr int kMetaRel = kMshRel + 64;                     // int64[kMetaPages*2]
constexpr int kLenRel = kMetaRel + kMetaPages * 2 * 8;     // int32[kMetaPages*2]
constexpr int kEscRel = (kLenRel + kMetaPages * 2 * 4 + 15) & ~15;
// K and V records each stream through one whole-record slot (TMA), sized for
// a raw record.
constexpr int kSlotRel = kEscRel + kEscCap;                // K slot, then V slot
constexpr int kBarRel = kSlotRel + 2 * kRawRecord;         // uint64 mbarrier[2]
constexpr int kSmem = kBarRel + 16;

struct Params {
  const uint32_t* q;
  const int64_t* rec_addr;
  const int32_t* rec_len;
  const uint32_t* plut;
  const uint8_t* slut;
  const int32_t* block_table;
  const int32_t* seq_lens;
  float* opart;
  float* ml;
  int64_t q_stride, bt_stride;  // q in uint32 words, block table in entries
  int kvh, splits;
  float scale_log2;
};

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

// c += A B for one m16n8k16 tile; a0..a3 / b0, b1 in the PTX fragment order.
__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                    uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}

__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

// Thread-issued bulk copy of ``bytes`` (multiple of 16) completing on ``bar``.
__device__ __forceinline__ void tma_load(void* dst, const void* src, uint32_t bytes,
                                         uint64_t* bar) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_u32(bar)), "r"(parity) : "memory");
}

__global__ void __launch_bounds__(kThreads, 2) decode_attn(Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ __align__(16) uint8_t sl[2][16];
  uint8_t* Ps = smem;
  float* red = reinterpret_cast<float*>(smem + kRedRel);
  float* red2 = reinterpret_cast<float*>(smem + kRed2Rel);
  float* msh = reinterpret_cast<float*>(smem + kMshRel);
  uint8_t* slots = smem + kSlotRel;
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + kBarRel);
  uint8_t* escbuf = smem + kEscRel;
  int64_t* maddr = reinterpret_cast<int64_t*>(smem + kMetaRel);
  int32_t* mlen = reinterpret_cast<int32_t*>(smem + kLenRel);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, c = lane & 3;
  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int seq = p.seq_lens[b];
  const int pages = (seq + 127) >> 7;
  const int pps = (pages + p.splits - 1) / p.splits;
  const int p0 = split * pps;
  const int n = max(0, min(pages, p0 + pps) - p0);
  const int32_t* bt = p.block_table + (int64_t)b * p.bt_stride + p0;
  // Local pages [base, base + kMetaPages) of this split into maddr / mlen.
  auto load_meta = [&](int base) {
    for (int i = tid; i < kMetaPages && base + i < n; i += kThreads) {
      const int64_t kidx = ((int64_t)bt[base + i] * 2) * p.kvh + h;
      maddr[2 * i] = p.rec_addr[kidx];
      maddr[2 * i + 1] = p.rec_addr[kidx + p.kvh];
      mlen[2 * i] = p.rec_len[kidx];
      mlen[2 * i + 1] = p.rec_len[kidx + p.kvh];
    }
  };
  load_meta(0);
  const uint32_t kl0 = p.plut[h * 2], kl1 = p.plut[h * 2 + 1];
  const uint32_t vl0 = p.plut[(p.kvh + h) * 2], vl1 = p.plut[(p.kvh + h) * 2 + 1];
  if (tid < 32) sl[tid >> 4][tid & 15] = p.slut[((tid >> 4) * p.kvh + h) * 16 + (tid & 15)];
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < 2; ++s) mbar_init(&mbar[s], 1);
    fence_mbar_init();
  }
  uint32_t qa[8][2];
  {
    const uint32_t* q = p.q + (int64_t)b * p.q_stride + (h * 8 + g) * 64;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      qa[s][0] = q[8 * s + c];
      qa[s][1] = q[8 * s + 4 + c];
    }
  }
  __syncthreads();

  // Thread 0 bulk-copies page i's K (which = 0) or V record into slot
  // ``which``, the slot's i-th use, so it completes mbarrier phase parity
  // i & 1. Each slot is reissued right after the barrier that retires its
  // decode. ``cb`` is the first local page held in maddr / mlen.
  int cb = 0;
  auto issue = [&](int i, int which) {
    if (i < n) {
      const int m = 2 * (i - cb) + which;
      tma_load(slots + which * kRawRecord, reinterpret_cast<const void*>(maddr[m]),
               kDenseBytes + mlen[m], &mbar[which]);
    }
  };
  auto slot_rec = [&](int i, int which) {
    mbar_wait(&mbar[which], i & 1);
    return slots + which * kRawRecord;
  };

  // Transposed tiles fill the 8 GQA heads into mma's n8 side: S^T = K Q^T and
  // O^T += V^T P^T, with K / V^T as the A operand. Thread (g, c) of warp w
  // holds s[2 jj + e] = S[head 2 c + e][token 16 w + 8 jj + g] and
  // o[2 jj + e] = O[head 2 c + e][dim 16 w + 8 jj + g].
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  float o[4] = {};
  if (tid == 0) {
    issue(0, 0);
    issue(0, 1);
  }
  for (int i = 0; i < n; ++i) {
    // Previous PV finished with Ps, esc and the V slot.
    __syncthreads();
    if (i - cb == kMetaChunk) {
      // Page i's K record was issued from the lookahead entry of the old chunk.
      cb = i;
      load_meta(cb);
      __syncthreads();
    }

    float s[4] = {};
    if (tid == 0 && i > 0) issue(i, 1);
    decode_rec(slot_rec(i, 0), mlen[2 * (i - cb)], kl0, kl1, sl[0], escbuf, tid,
               [&](int st, const uint32_t (&ka)[4]) {
                 mma(s, ka[0], ka[2], ka[1], ka[3], qa[st][0], qa[st][1]);
               });
    const int tok0 = (p0 + i) * 128 + 16 * warp + g;
    float mx[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
#pragma unroll
      for (int jj = 0; jj < 2; ++jj)
        s[2 * jj + e] = tok0 + 8 * jj < seq ? s[2 * jj + e] * p.scale_log2 : -INFINITY;
      mx[e] = fmaxf(s[e], s[2 + e]);
#pragma unroll
      for (int x = 4; x < 32; x <<= 1) mx[e] = fmaxf(mx[e], __shfl_xor_sync(0xffffffffu, mx[e], x));
    }
    if (g == 0) *reinterpret_cast<float2*>(red + warp * 8 + 2 * c) = make_float2(mx[0], mx[1]);
    // Every warp is done with the K slot.
    __syncthreads();
    if (tid == 0) issue(i + 1, 0);
    float2 pm = *reinterpret_cast<const float2*>(red + 2 * c);
#pragma unroll
    for (int w = 1; w < 8; ++w) {
      const float2 t = *reinterpret_cast<const float2*>(red + w * 8 + 2 * c);
      pm = make_float2(fmaxf(pm.x, t.x), fmaxf(pm.y, t.y));
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const float mnew = fmaxf(m_run[e], e ? pm.y : pm.x);
      const float alpha = exp2f(m_run[e] - mnew);
      m_run[e] = mnew;
      float psum = 0.f;
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const float pv = exp2f(s[2 * jj + e] - mnew);
        psum += pv;
        const int hq = 2 * c + e;
        *reinterpret_cast<__nv_bfloat16*>(Ps + hq * 256 + ((2 * warp + jj) ^ hq) * 16 + 2 * g) =
            __float2bfloat16_rn(pv);
        o[2 * jj + e] *= alpha;
      }
      l_run[e] = l_run[e] * alpha + psum;
    }
    __syncthreads();

    // O^T += V^T P^T for d in 16 warp .. 16 warp + 15.
    const int mi = lane >> 3;
    const int prow = lane & 7;
    uint32_t pb[4];
    decode_rec(slot_rec(i, 1), mlen[2 * (i - cb) + 1], vl0, vl1, sl[1], escbuf, tid,
               [&](int st, const uint32_t (&va)[4]) {
                 if ((st & 1) == 0) ldsm4(pb, Ps + prow * 256 + ((2 * st + mi) ^ prow) * 16);
                 const int h2 = st & 1;
                 mma(o, va[0], va[2], va[1], va[3], pb[2 * h2], pb[2 * h2 + 1]);
               });
  }

  // Partial outputs: rows are query heads, columns d.
  const int64_t row0 = ((int64_t)(b * p.kvh + h) * p.splits + split) * 8;
#pragma unroll
  for (int jj = 0; jj < 2; ++jj)
#pragma unroll
    for (int e = 0; e < 2; ++e)
      p.opart[(row0 + 2 * c + e) * 128 + 16 * warp + 8 * jj + g] = o[2 * jj + e];
  float l[2] = {l_run[0], l_run[1]};
#pragma unroll
  for (int e = 0; e < 2; ++e)
#pragma unroll
    for (int x = 4; x < 32; x <<= 1) l[e] += __shfl_xor_sync(0xffffffffu, l[e], x);
  __syncthreads();
  if (g == 0) *reinterpret_cast<float2*>(red2 + warp * 8 + 2 * c) = make_float2(l[0], l[1]);
  if (warp == 0 && g == 0) *reinterpret_cast<float2*>(msh + 2 * c) = make_float2(m_run[0], m_run[1]);
  __syncthreads();
  if (tid < 8) {
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) sum += red2[w * 8 + tid];
    p.ml[(row0 + tid) * 2] = msh[tid];
    p.ml[(row0 + tid) * 2 + 1] = sum;
  }
}

// Rows with no context (cudagraph padding) produce zeros.
__global__ void combine(const float* opart, const float* ml, __nv_bfloat16* out,
                        int64_t out_stride, int kvh, int splits) {
  const int b = blockIdx.x, qh = blockIdx.y, d = threadIdx.x;
  const int h = qh >> 3, g = qh & 7;
  const int64_t base = (int64_t)(b * kvh + h) * splits;
  float gm = -INFINITY;
  for (int s = 0; s < splits; ++s) gm = fmaxf(gm, ml[((base + s) * 8 + g) * 2]);
  float den = 0.f, acc = 0.f;
  if (gm != -INFINITY) {
    for (int s = 0; s < splits; ++s) {
      const int64_t r = (base + s) * 8 + g;
      const float w = exp2f(ml[r * 2] - gm);
      den += w * ml[r * 2 + 1];
      acc += w * opart[r * 128 + d];
    }
  }
  out[(int64_t)b * out_stride + qh * 128 + d] = __float2bfloat16(den > 0.f ? acc / den : 0.f);
}

}  // namespace

void fused_decode(torch::Tensor q, torch::Tensor rec_addr, torch::Tensor rec_len,
                  torch::Tensor plut, torch::Tensor slut, torch::Tensor block_table,
                  torch::Tensor seq_lens, torch::Tensor opart, torch::Tensor ml,
                  torch::Tensor out, int64_t splits, double scale) {
  TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && q.stride(2) == 1 && q.stride(1) == 128 &&
                  q.stride(0) % 2 == 0,
              "q must be [B, H, 128] with contiguous heads");
  TORCH_CHECK(out.dim() == 3 && out.stride(2) == 1 && out.stride(1) == 128,
              "out must be [B, H, 128] with contiguous heads");
  TORCH_CHECK(block_table.stride(1) == 1, "block_table rows must be contiguous");
  const int B = q.size(0), kvh = q.size(1) / 8;
  if (B == 0) return;
  TORCH_CHECK(opart.numel() >= (int64_t)B * kvh * splits * 8 * 128 &&
                  ml.numel() >= (int64_t)B * kvh * splits * 16,
              "decode workspace too small");
  Params p;
  p.q = reinterpret_cast<const uint32_t*>(q.data_ptr());
  p.rec_addr = rec_addr.data_ptr<int64_t>();
  p.rec_len = rec_len.data_ptr<int32_t>();
  p.plut = reinterpret_cast<const uint32_t*>(plut.data_ptr());
  p.slut = slut.data_ptr<uint8_t>();
  p.block_table = block_table.data_ptr<int32_t>();
  p.seq_lens = seq_lens.data_ptr<int32_t>();
  p.opart = opart.data_ptr<float>();
  p.ml = ml.data_ptr<float>();
  p.q_stride = q.stride(0) / 2;
  p.bt_stride = block_table.stride(0);
  p.kvh = kvh;
  p.splits = splits;
  p.scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaFuncSetAttribute(decode_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
  decode_attn<<<dim3(splits, kvh, B), kThreads, kSmem, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  combine<<<dim3(B, kvh * 8), 128, 0, stream>>>(
      p.opart, p.ml, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), out.stride(0), kvh,
      splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
