// SPDX-License-Identifier: Apache-2.0
// HBM-resident FragPack KV store kernels (see hbm_kv.py for the state machine).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "fragpack_common.cuh"

namespace {

using namespace fragpack;

// Counter slots (int32[kCounters]).
enum : int {
  kFreeTop = 0,      // staging free-stack depth
  kTrans = 1,        // transient slots handed out this step
  kClaims = 2,       // blocks claimed this step
  kCloses = 3,       // blocks closed this step
  kErrStaging = 4,   // dropped claims: staging exhausted
  kErrOverflow = 5,  // dropped encodes: overflow pool exhausted
  kFallbacks = 6,    // records stored raw in the overflow pool (cumulative)
  kEncoded = 7,      // records encoded (cumulative)
  kSpills = 8,       // coded records spilled raw: block-layer page full (cumulative)
};
// Per-block step stamps (int32[kTags, NB]).
enum : int { kTagLive = 0, kTagClose = 1, kTagClaim = 2, kTagCloseDedupe = 3, kTags = 4 };

constexpr int kPrepThreads = 1024;

struct Store {
  int32_t* counters;
  int32_t* wslot;       // [NB] staging slot of an open block, -1 if none
  int32_t* tags;        // [kTags, NB]
  int32_t* free_stack;  // [S_p]
  uint32_t* ovf_bits;   // [ceil(O / 32)] overflow slots in use (tail bits preset)
  int32_t* claim_list;  // [cap]
  int32_t* close_list;  // [cap]
  int64_t* rec_addr;    // [L, NB, 2, H]
  int32_t* rec_len;     // [L, NB, 2, H]
  int64_t staging;      // [S_p, L, 2, H, kRawRecord] persistent staging
  int64_t scratch;      // [S_t, 2, H, kRawRecord] one layer's transient staging
  int64_t ovf_base;     // [O, kRawRecord] raw records that do not code
  int64_t ovf_end;
  int32_t* fill;        // [L, NB] arena bytes used in each block-layer page
  int L, H, NB, S_p, S_t, ovf_words;
};

Store make_store(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                 torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                 torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                 torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                 torch::Tensor fill) {
  Store s;
  s.counters = counters.data_ptr<int32_t>();
  s.wslot = wslot.data_ptr<int32_t>();
  s.tags = tags.data_ptr<int32_t>();
  s.free_stack = free_stack.data_ptr<int32_t>();
  s.ovf_bits = reinterpret_cast<uint32_t*>(ovf_bits.data_ptr<int32_t>());
  s.ovf_words = ovf_bits.numel();
  s.claim_list = claim_list.data_ptr<int32_t>();
  s.close_list = close_list.data_ptr<int32_t>();
  s.rec_addr = rec_addr.data_ptr<int64_t>();
  s.rec_len = rec_len.data_ptr<int32_t>();
  s.staging = reinterpret_cast<int64_t>(staging.data_ptr());
  s.scratch = reinterpret_cast<int64_t>(scratch.data_ptr());
  s.ovf_base = reinterpret_cast<int64_t>(overflow.data_ptr());
  s.ovf_end = s.ovf_base + overflow.numel();
  s.fill = fill.data_ptr<int32_t>();
  s.L = rec_addr.size(0);
  s.NB = rec_addr.size(1);
  s.H = rec_addr.size(3);
  s.S_p = free_stack.numel();
  s.S_t = scratch.numel() / (2LL * s.H * kRawRecord);
  return s;
}

// Staging record (kv * H + h) of the block in staging slot ``w`` at ``layer``.
__device__ __forceinline__ int64_t staging_addr(const Store& s, int w, int layer, int kvh) {
  return w < s.S_p ? s.staging + (((int64_t)w * s.L + layer) * 2 * s.H + kvh) * kRawRecord
                   : s.scratch + ((int64_t)(w - s.S_p) * 2 * s.H + kvh) * kRawRecord;
}

// Overflow slots are a bitmap so that the model stream (encode, claims) and a
// load stream (imports) may allocate and free concurrently. Returns the slot,
// or -1 when the pool is full. ``hint`` spreads concurrent callers.
__device__ int ovf_alloc(const Store& s, uint32_t hint) {
  for (int i = 0; i < s.ovf_words; ++i) {
    const int w = (int)((hint + i) % (uint32_t)s.ovf_words);
    uint32_t cur = *reinterpret_cast<volatile uint32_t*>(&s.ovf_bits[w]);
    while (cur != 0xffffffffu) {
      const int bit = __ffs(~cur) - 1;
      const uint32_t prev = atomicOr(&s.ovf_bits[w], 1u << bit);
      if (!(prev & (1u << bit))) return 32 * w + bit;
      cur = prev | (1u << bit);
    }
  }
  return -1;
}

// Frees the overflow slot holding ``addr``; other addresses are ignored.
__device__ __forceinline__ void ovf_free(const Store& s, int64_t addr) {
  if (addr < s.ovf_base || addr >= s.ovf_end) return;
  const int slot = (int)((addr - s.ovf_base) / kRawRecord);
  atomicAnd(&s.ovf_bits[slot >> 5], ~(1u << (slot & 31)));
}

// One CTA. Phases (each ends at a barrier):
// 1. GC: blocks not in ``live`` drop their persistent staging slot.
// 2. Mark blocks receiving their last token (offset 127) this step.
// 3. Claim: a block receiving offset 0 gets a staging slot (kept if it
//    still owns one; transient if it also closes this step; else popped).
// 4. Close list: blocks receiving offset 127 that own a slot.
// 5. Point every claimed block's records (all layers) at its staging, release
//    overflow records they held, and empty its arena pages for the encodes.
__global__ void __launch_bounds__(kPrepThreads) prepare_kernel(Store s, const int64_t* slots,
                                                               int n_tok, const int32_t* live,
                                                               int n_live, int32_t step) {
  const int tid = threadIdx.x;
  int32_t* live_tag = s.tags + kTagLive * s.NB;
  int32_t* close_tag = s.tags + kTagClose * s.NB;
  int32_t* claim_tag = s.tags + kTagClaim * s.NB;
  int32_t* dedupe_tag = s.tags + kTagCloseDedupe * s.NB;
  if (tid == 0) {
    s.counters[kTrans] = 0;
    s.counters[kClaims] = 0;
    s.counters[kCloses] = 0;
  }
  for (int i = tid; i < n_live; i += kPrepThreads) live_tag[live[i]] = step;
  __syncthreads();
  for (int b = tid; b < s.NB; b += kPrepThreads) {
    const int w = s.wslot[b];
    if (w >= s.S_p) {
      s.wslot[b] = -1;
    } else if (w >= 0 && live_tag[b] != step) {
      s.wslot[b] = -1;
      s.free_stack[atomicAdd(&s.counters[kFreeTop], 1)] = w;
    }
  }
  for (int i = tid; i < n_tok; i += kPrepThreads) {
    const int64_t sl = slots[i];
    if (sl >= 0 && (sl & 127) == 127) close_tag[sl >> 7] = step;
  }
  __syncthreads();
  for (int i = tid; i < n_tok; i += kPrepThreads) {
    const int64_t sl = slots[i];
    if (sl < 0 || (sl & 127) != 0) continue;
    const int b = sl >> 7;
    if (atomicExch(&claim_tag[b], step) == step) continue;
    int w = s.wslot[b];
    if (w < 0) {
      if (close_tag[b] == step) {
        const int t = atomicAdd(&s.counters[kTrans], 1);
        if (t < s.S_t) w = s.S_p + t;
      }
      if (w < 0) {
        const int top = atomicSub(&s.counters[kFreeTop], 1) - 1;
        if (top >= 0) {
          w = s.free_stack[top];
        } else {
          atomicAdd(&s.counters[kFreeTop], 1);
          atomicAdd(&s.counters[kErrStaging], 1);
        }
      }
    }
    s.wslot[b] = w;
    if (w >= 0) s.claim_list[atomicAdd(&s.counters[kClaims], 1)] = b;
  }
  __syncthreads();
  for (int i = tid; i < n_tok; i += kPrepThreads) {
    const int64_t sl = slots[i];
    if (sl < 0 || (sl & 127) != 127) continue;
    const int b = sl >> 7;
    if (atomicExch(&dedupe_tag[b], step) == step) continue;
    if (s.wslot[b] >= 0) s.close_list[atomicAdd(&s.counters[kCloses], 1)] = b;
  }
  const int n_claim = s.counters[kClaims];
  const int per = s.L * 2 * s.H;
  for (int x = tid; x < n_claim * per; x += kPrepThreads) {
    const int b = s.claim_list[x / per];
    const int r = x % per, layer = r / (2 * s.H), kvh = r % (2 * s.H);
    const int64_t idx = ((int64_t)layer * s.NB + b) * 2 * s.H + kvh;
    ovf_free(s, s.rec_addr[idx]);
    s.rec_addr[idx] = staging_addr(s, s.wslot[b], layer, kvh);
    s.rec_len[idx] = kRawVarLen;
    if (kvh == 0) s.fill[(int64_t)layer * s.NB + b] = 0;
  }
}

// One CTA per token; stores K / V in kernel order into the block's staging.
// K pair (dims 2 p, 2 p + 1) is one u32; a V pair is two tokens apart.
__global__ void write_kernel(Store s, const uint16_t* k, const uint16_t* v, int64_t ks0,
                             int64_t ks1, int64_t vs0, int64_t vs1, const int64_t* slots,
                             int layer) {
  const int64_t sl = slots[blockIdx.x];
  if (sl < 0) return;
  const int b = sl >> 7, off = sl & 127;
  const int w = s.wslot[b];
  if (w < 0) return;
  for (int x = threadIdx.x; x < s.H * 64; x += blockDim.x) {
    const int h = x >> 6, p = x & 63;
    {
      const uint16_t* src = k + blockIdx.x * ks0 + h * ks1 + 2 * p;
      const uint32_t val = (uint32_t)src[0] | ((uint32_t)src[1] << 16);
      const int st = p >> 3, f = (p >> 2) & 1, c = p & 3;
      const int tw = off >> 4, j = (off >> 3) & 1, g = off & 7;
      const int t = 32 * tw + 4 * g + c;
      uint8_t* rec = reinterpret_cast<uint8_t*>(staging_addr(s, w, layer, h));
      *reinterpret_cast<uint32_t*>(rec + 16 * (256 * st + t) + 2 * (4 * j + 2 * f)) = val;
    }
    {
      const uint16_t* src = v + blockIdx.x * vs0 + h * vs1 + 2 * p;
      const int st = off >> 4, f = (off >> 3) & 1, c = (off >> 1) & 3, e = off & 1;
      uint8_t* rec = reinterpret_cast<uint8_t*>(staging_addr(s, w, layer, s.H + h));
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int d = 2 * p + q;
        const int tw = d >> 4, j = (d >> 3) & 1, g = d & 7;
        const int t = 32 * tw + 4 * g + c;
        *reinterpret_cast<uint16_t*>(rec + 16 * (256 * st + t) + 2 * (4 * j + 2 * f + e)) = src[q];
      }
    }
  }
}

// grid (cap, 2 H): CTA (i, kvh) encodes record kvh of close_list[i] at
// ``layer`` from staging into the block's arena page, which the block's 2 H
// records share: each takes the next 16-byte aligned ``kDenseBytes + var_len``
// bytes. A record that does not code, or no longer fits the page, is stored
// raw in the overflow pool. Matches attn_page.encode_record bit for bit.
__global__ void __launch_bounds__(kThreads) encode_kernel(Store s, int layer, int64_t arena,
                                                          int64_t page_bytes,
                                                          const uint8_t* rank_of) {
  __shared__ uint8_t rank_s[256];
  __shared__ __align__(16) uint8_t var_s[kVarCap];
  __shared__ int wsum[2][8];
  __shared__ int s_off, s_ovf;
  if ((int)blockIdx.x >= s.counters[kCloses]) return;
  const int kvh = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = s.close_list[blockIdx.x];
  const uint8_t* src = reinterpret_cast<const uint8_t*>(staging_addr(s, s.wslot[b], layer, kvh));
  const int64_t idx = ((int64_t)layer * s.NB + b) * 2 * s.H + kvh;
  rank_s[tid] = rank_of[((int64_t)layer * 2 * s.H + kvh) * 256 + tid];
  uint4 wd[8];
#pragma unroll
  for (int st = 0; st < 8; ++st) wd[st] = reinterpret_cast<const uint4*>(src)[st * kThreads + tid];
  for (int i = tid; i < kVarCap / 16; i += kThreads)
    reinterpret_cast<uint4*>(var_s)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();

  uint32_t sym[6] = {0, 0, 0, 0, 0, 0};
  uint64_t emask = 0, rmask = 0;
#pragma unroll
  for (int st = 0; st < 8; ++st) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int sc = 8 * st + e, gi = sc >> 5, sg = sc & 31;
      const uint32_t val = (u4get(wd[st], e >> 1) >> (16 * (e & 1))) & 0xffffu;
      const int r = rank_s[val >> 8];
      const uint32_t code = r < kPrimaryDirect ? r : kPrimaryDirect;
      if (sg < 24) {
        sym[3 * gi + (sg >> 3)] |= code << (4 * (sg & 7));
      } else {
#pragma unroll
        for (int kk = 0; kk < 3; ++kk) sym[3 * gi + kk] |= ((code >> kk) & 1u) << (4 * (sg - 24) + 3);
      }
      if (r >= kPrimaryDirect) emask |= 1ull << sc;
      if (r >= kTableEntries) rmask |= 1ull << sc;
    }
  }
  const int ne = __popcll(emask), nr = __popcll(rmask);
  const int eex = warp_excl_scan(ne, lane), rex = warp_excl_scan(nr, lane);
  if (lane == 31) {
    wsum[0][warp] = eex + ne;
    wsum[1][warp] = rex + nr;
  }
  __syncthreads();
  int eo = eex, ro = rex, etot = 0, rtot = 0, rmax = 0;
#pragma unroll
  for (int w = 0; w < 8; ++w) {
    if (w < warp) {
      eo += wsum[0][w];
      ro += wsum[1][w];
    }
    etot += wsum[0][w];
    rtot += wsum[1][w];
    rmax = max(rmax, wsum[1][w]);
  }
  const int raw_offset = (kHeaderBytes + ((etot + 1) >> 1) + 3) & ~3;
  const int var_len = (raw_offset + 4 * rtot + 15) & ~15;

  const bool codes = var_len <= kVarCap && rmax <= kMaxWarpRaw;
  if (tid == 0) {
    // Claim page bytes with a CAS so a record that does not fit leaves the
    // fill unchanged for the block's smaller records.
    int off = -1;
    if (codes) {
      int32_t* fill = s.fill + (int64_t)layer * s.NB + b;
      const int need = kDenseBytes + var_len;
      int cur = *reinterpret_cast<volatile int32_t*>(fill);
      while (cur + need <= page_bytes) {
        const int prev = atomicCAS(fill, cur, cur + need);
        if (prev == cur) {
          off = cur;
          break;
        }
        cur = prev;
      }
      if (off < 0) atomicAdd(&s.counters[kSpills], 1);
    }
    s_off = off;
  }
  __syncthreads();
  if (s_off < 0) {
    if (tid == 0) {
      const int slot = ovf_alloc(s, (uint32_t)(b * 2 * s.H + kvh));
      if (slot < 0) atomicAdd(&s.counters[kErrOverflow], 1);
      atomicAdd(&s.counters[kFallbacks], 1);
      atomicAdd(&s.counters[kEncoded], 1);
      s_ovf = slot;
    }
    __syncthreads();
    if (s_ovf < 0) return;
    uint8_t* dst = reinterpret_cast<uint8_t*>(s.ovf_base) + (int64_t)s_ovf * kRawRecord;
#pragma unroll
    for (int st = 0; st < 8; ++st) reinterpret_cast<uint4*>(dst)[st * kThreads + tid] = wd[st];
    if (tid == 0) {
      s.rec_addr[idx] = reinterpret_cast<int64_t>(dst);
      s.rec_len[idx] = kRawVarLen;
    }
    return;
  }

  // Escape nibbles (from byte 48, two per byte, low nibble first) and raw
  // entries ``ordinal | high << 16`` in scalar order.
  uint32_t* var32 = reinterpret_cast<uint32_t*>(var_s);
#pragma unroll
  for (int st = 0; st < 8; ++st) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int sc = 8 * st + e;
      if ((emask >> sc) & 1ull) {
        const uint32_t high = ((u4get(wd[st], e >> 1) >> (16 * (e & 1))) >> 8) & 0xffu;
        const int r = rank_s[high];
        const uint32_t nib = r < kTableEntries ? r - kPrimaryDirect : 15u;
        const int bit = 8 * kHeaderBytes + 4 * eo;
        atomicOr(&var32[bit >> 5], nib << (bit & 31));
        if (r >= kTableEntries) var32[(raw_offset >> 2) + ro++] = (uint32_t)eo | (high << 16);
        ++eo;
      }
    }
  }
  if (tid == 0) {
    var32[0] = etot;
    var32[1] = rtot;
    var32[2] = raw_offset;
    var32[3] = 0;
    uint16_t* h16 = reinterpret_cast<uint16_t*>(var_s);
    int ep = 0, rp = 0;
    for (int w = 0; w < 8; ++w) {
      h16[8 + w] = ep;
      h16[16 + w] = rp;
      ep += wsum[0][w];
      rp += wsum[1][w];
    }
  }
  uint8_t* dst = reinterpret_cast<uint8_t*>(arena) + (int64_t)b * page_bytes + s_off;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 a = wd[2 * j], c = wd[2 * j + 1];
    reinterpret_cast<uint4*>(dst)[j * kThreads + tid] =
        make_uint4(__byte_perm(a.x, a.y, 0x6420), __byte_perm(a.z, a.w, 0x6420),
                   __byte_perm(c.x, c.y, 0x6420), __byte_perm(c.z, c.w, 0x6420));
  }
#pragma unroll
  for (int m = 0; m < 3; ++m)
    reinterpret_cast<uint2*>(dst + kLowBytes)[m * kThreads + tid] = make_uint2(sym[2 * m], sym[2 * m + 1]);
  __syncthreads();
  for (int i = tid; i < var_len / 16; i += kThreads)
    reinterpret_cast<uint4*>(dst + kDenseBytes)[i] = reinterpret_cast<const uint4*>(var_s)[i];
  if (tid == 0) {
    s.rec_addr[idx] = reinterpret_cast<int64_t>(dst);
    s.rec_len[idx] = var_len;
    atomicAdd(&s.counters[kEncoded], 1);
  }
}

// Closed blocks release their staging slot; the close list is then emptied so
// a forward without a fresh prepare (CUDA-graph capture) encodes nothing.
__global__ void commit_kernel(Store s) {
  const int n = s.counters[kCloses];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int b = s.close_list[i];
    const int w = s.wslot[b];
    if (w >= 0 && w < s.S_p) s.free_stack[atomicAdd(&s.counters[kFreeTop], 1)] = w;
    s.wslot[b] = -1;
  }
  __syncthreads();
  if (threadIdx.x == 0) s.counters[kCloses] = 0;
}

constexpr int kGatherSmem = kTile + kEscCap;

// grid (pages, 2 H): decodes record kvh of block src[i] at one layer into
// workspace page i, token-major [pages, 128, H, 128] BF16 (K or V).
__global__ void __launch_bounds__(kThreads) gather_kernel(const int32_t* src, const int64_t* rec_addr,
                                                          const int32_t* rec_len,
                                                          const uint32_t* plut, const uint8_t* slut,
                                                          uint8_t* wk, uint8_t* wv, int H) {
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* tile = smem;
  uint8_t* esc = smem + kTile;
  const int kvh = blockIdx.y, kv = kvh / H, h = kvh % H;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, c = lane & 3;
  const int64_t idx = (int64_t)src[blockIdx.x] * 2 * H + kvh;
  const uint8_t* rec = reinterpret_cast<const uint8_t*>(rec_addr[idx]);
  auto put = [&](int tok, int dim, uint32_t v, int bytes) {
    uint8_t* p = tile + tok * 256 + (((dim >> 3) ^ (tok & 7)) << 4) + 2 * (dim & 7);
    if (bytes == 4)
      *reinterpret_cast<uint32_t*>(p) = v;
    else
      *reinterpret_cast<uint16_t*>(p) = (uint16_t)v;
  };
  decode_rec(rec, rec_len[idx], plut[2 * kvh], plut[2 * kvh + 1], slut + 16 * kvh, esc, tid,
             [&](int st, const uint32_t (&bf)[4]) {
#pragma unroll
               for (int i = 0; i < 4; ++i) {
                 const int j = i >> 1, f = i & 1;
                 if (kv == 0) {
                   put(16 * warp + 8 * j + g, 16 * st + 8 * f + 2 * c, bf[i], 4);
                 } else {
                   const int tok = 16 * st + 8 * f + 2 * c, dim = 16 * warp + 8 * j + g;
                   put(tok, dim, bf[i] & 0xffffu, 2);
                   put(tok + 1, dim, bf[i] >> 16, 2);
                 }
               }
             });
  __syncthreads();
  uint8_t* out = kv == 0 ? wk : wv;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const int x = q * kThreads + tid, r = x >> 4, jj = x & 15;
    const uint4 val = *reinterpret_cast<const uint4*>(tile + r * 256 + ((jj ^ (r & 7)) << 4));
    *reinterpret_cast<uint4*>(out + (((int64_t)blockIdx.x * 128 + r) * H + h) * 256 + jj * 16) = val;
  }
}

// Block blob: one closed block's records in an external tier. A fixed header
// (u32 magic, version, L, H; i64 bytes, digest) and the i32 rec_len table
// [L, 2, H] fill a kBlobAlign-padded head; the records follow in (layer, kv,
// head) order, each kDenseBytes + rec_len bytes (raw records 32768), so every
// record stays 16-byte aligned. ``bytes`` excludes the tail padding.
constexpr uint32_t kBlobMagic = 0x424B5046u;  // "FPKB"
constexpr uint32_t kBlobVersion = 1;
constexpr int kBlobFixed = 64;
constexpr int64_t kBlobAlign = 4096;
constexpr int kMaxHeads = 16;
// Import status bits, per destination block.
enum : int { kBlobBadHeader = 1, kBlobBadLengths = 2, kBlobNoOverflow = 4 };

__host__ __device__ __forceinline__ int64_t blob_head_bytes(int L, int H) {
  return (kBlobFixed + 8LL * L * H + kBlobAlign - 1) / kBlobAlign * kBlobAlign;
}

__device__ long long block_sum(long long v, long long* red) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  long long t = 0;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) t += red[w];
  return t;
}

// grid (n, L): CTA (i, l) copies layer l's records of blocks[i] into the blob
// at out + i * stride; layer 0 also writes the head and lengths[i] (the blob
// rounded up to kBlobAlign). Every CTA sums the block's lengths itself.
__global__ void __launch_bounds__(kThreads) blob_export_kernel(const int32_t* blocks,
                                                               const int64_t* rec_addr,
                                                               const int32_t* rec_len, uint8_t* out,
                                                               int64_t stride, int64_t* lengths,
                                                               int L, int H, int NB, int64_t digest) {
  __shared__ long long red[kThreads / 32];
  const int i = blockIdx.x, l = blockIdx.y, tid = threadIdx.x, per = 2 * H;
  const int b = blocks[i];
  long long before = 0, total = 0;
  for (int g = tid; g < L * per; g += kThreads) {
    const long long n = kDenseBytes + rec_len[((int64_t)(g / per) * NB + b) * per + g % per];
    total += n;
    if (g / per < l) before += n;
  }
  before = block_sum(before, red);
  total = block_sum(total, red);
  const int64_t head = blob_head_bytes(L, H);
  uint8_t* dst = out + (int64_t)i * stride;
  if (l == 0) {
    if (tid == 0) {
      uint32_t* h32 = reinterpret_cast<uint32_t*>(dst);
      h32[0] = kBlobMagic;
      h32[1] = kBlobVersion;
      h32[2] = L;
      h32[3] = H;
      int64_t* h64 = reinterpret_cast<int64_t*>(dst + 16);
      h64[0] = head + total;
      h64[1] = digest;
      lengths[i] = (head + total + kBlobAlign - 1) / kBlobAlign * kBlobAlign;
    }
    int32_t* table = reinterpret_cast<int32_t*>(dst + kBlobFixed);
    for (int g = tid; g < L * per; g += kThreads)
      table[g] = rec_len[((int64_t)(g / per) * NB + b) * per + g % per];
  }
  int64_t off = head + before;
  for (int kvh = 0; kvh < per; ++kvh) {
    const int64_t idx = ((int64_t)l * NB + b) * per + kvh;
    const int n = kDenseBytes + rec_len[idx];
    const uint4* src = reinterpret_cast<const uint4*>(rec_addr[idx]);
    uint4* d = reinterpret_cast<uint4*>(dst + off);
    for (int q = tid; q < n / 16; q += kThreads) d[q] = src[q];
    off += n;
  }
}

// grid (n, L): CTA (i, l) validates the blob at src + offsets[i] and installs
// its layer-l records for blocks[i]: coded records packed into the block's
// arena page, raw ones (and coded ones that no longer fit) in overflow slots.
// The block's old overflow records are freed. Every CTA validates the whole
// head, so all layers agree; a failure sets status[i] and leaves the layer's
// tables untouched (the caller must recompute the block).
__global__ void __launch_bounds__(kThreads) blob_import_kernel(Store s, const uint8_t* src,
                                                               int64_t src_bytes,
                                                               const int64_t* offsets,
                                                               const int32_t* blocks, int32_t* status,
                                                               const int64_t* arena_base,
                                                               const int64_t* arena_stride,
                                                               int64_t digest) {
  __shared__ long long red[kThreads / 32];
  __shared__ int64_t s_src[2 * kMaxHeads], s_dst[2 * kMaxHeads];
  __shared__ int s_fail;
  const int i = blockIdx.x, l = blockIdx.y, tid = threadIdx.x, L = s.L, per = 2 * s.H;
  const int b = blocks[i];
  const int64_t off = offsets[i], head = blob_head_bytes(L, s.H);
  if (off < 0 || off % 16 || off + head > src_bytes) {
    if (tid == 0) atomicOr(&status[i], kBlobBadHeader);
    return;
  }
  const uint8_t* blob = src + off;
  const uint32_t* h32 = reinterpret_cast<const uint32_t*>(blob);
  const int64_t* h64 = reinterpret_cast<const int64_t*>(blob + 16);
  const int64_t declared = h64[0];
  if (h32[0] != kBlobMagic || h32[1] != kBlobVersion || (int)h32[2] != L || (int)h32[3] != s.H ||
      h64[1] != digest || declared < head || off + declared > src_bytes) {
    if (tid == 0) atomicOr(&status[i], kBlobBadHeader);
    return;
  }
  const int32_t* table = reinterpret_cast<const int32_t*>(blob + kBlobFixed);
  long long before = 0, total = 0;
  int bad = 0;
  for (int g = tid; g < L * per; g += kThreads) {
    const int v = table[g];
    const bool ok = v == kRawVarLen || (v >= kHeaderBytes && v <= kVarCap && v % 16 == 0);
    bad |= !ok;
    const long long n = kDenseBytes + (ok ? v : 0);
    total += n;
    if (g / per < l) before += n;
  }
  bad = __syncthreads_or(bad);
  before = block_sum(before, red);
  total = block_sum(total, red);
  if (bad || head + total != declared) {
    if (tid == 0) atomicOr(&status[i], kBlobBadLengths);
    return;
  }
  const int* lens = table + l * per;
  if (tid == 0) {
    const int64_t page = arena_stride[l];
    const int64_t page_ptr = arena_base[l] + (int64_t)b * page;
    int64_t from = head + before;
    int fill = 0, placed = 0;
    for (; placed < per; ++placed) {
      const int n = kDenseBytes + lens[placed];
      int64_t d;
      if (lens[placed] != kRawVarLen && fill + n <= page) {
        d = page_ptr + fill;
        fill += n;
      } else {
        const int slot = ovf_alloc(s, (uint32_t)(b * per + placed));
        if (slot < 0) break;
        d = s.ovf_base + (int64_t)slot * kRawRecord;
      }
      s_src[placed] = from;
      s_dst[placed] = d;
      from += n;
    }
    s_fail = placed < per;
    if (s_fail) {
      for (int k = 0; k < placed; ++k) ovf_free(s, s_dst[k]);
      atomicOr(&status[i], kBlobNoOverflow);
    } else {
      s.fill[(int64_t)l * s.NB + b] = fill;
    }
  }
  __syncthreads();
  if (s_fail) return;
  for (int kvh = 0; kvh < per; ++kvh) {
    const uint4* from = reinterpret_cast<const uint4*>(blob + s_src[kvh]);
    uint4* to = reinterpret_cast<uint4*>(s_dst[kvh]);
    for (int q = tid; q < (kDenseBytes + lens[kvh]) / 16; q += kThreads) to[q] = from[q];
  }
  __syncthreads();
  if (tid < per) {
    const int64_t idx = ((int64_t)l * s.NB + b) * per + tid;
    ovf_free(s, s.rec_addr[idx]);
    s.rec_addr[idx] = s_dst[tid];
    s.rec_len[idx] = lens[tid];
  }
}

}  // namespace

void hbm_prepare(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                 torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                 torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                 torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                 torch::Tensor fill,
                 torch::Tensor slot_mapping, int64_t n_tok, torch::Tensor live, int64_t n_live,
                 int64_t step) {
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64, "slot_mapping must be int64");
  // Claims and closes are both bounded by the token count.
  TORCH_CHECK(n_tok <= claim_list.numel() && n_tok <= close_list.numel(), "block lists too small");
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  prepare_kernel<<<1, kPrepThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      s, slot_mapping.data_ptr<int64_t>(), n_tok, live.data_ptr<int32_t>(), n_live,
      static_cast<int32_t>(step));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_write(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
               torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
               torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
               torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
               torch::Tensor fill,
               torch::Tensor key, torch::Tensor value, torch::Tensor slot_mapping,
               int64_t layer) {
  const int64_t n = key.size(0);
  if (n == 0) return;
  TORCH_CHECK(key.stride(2) == 1 && value.stride(2) == 1, "head dim must be contiguous");
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  write_kernel<<<n, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      s, reinterpret_cast<const uint16_t*>(key.data_ptr()),
      reinterpret_cast<const uint16_t*>(value.data_ptr()), key.stride(0), key.stride(1),
      value.stride(0), value.stride(1), slot_mapping.data_ptr<int64_t>(), layer);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_encode(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill,
                torch::Tensor rank_of, int64_t layer, int64_t arena, int64_t page_bytes,
                int64_t max_close) {
  if (max_close == 0) return;
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  TORCH_CHECK(max_close <= close_list.numel(), "close list too small");
  TORCH_CHECK(rank_of.numel() == (int64_t)s.L * 2 * s.H * 256, "rank_of must hold every layer");
  TORCH_CHECK(page_bytes % 16 == 0 && page_bytes < (1LL << 31), "bad arena page size");
  encode_kernel<<<dim3(max_close, 2 * s.H), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      s, layer, arena, page_bytes, rank_of.data_ptr<uint8_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_commit(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill) {
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  commit_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(s);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_gather(torch::Tensor src_blocks, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor plut, torch::Tensor slut, torch::Tensor wk, torch::Tensor wv,
                int64_t kv_heads) {
  const int64_t n = src_blocks.numel();
  if (n == 0) return;
  TORCH_CHECK(wk.numel() * wk.element_size() >= n * 128 * kv_heads * 256, "workspace too small");
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(gather_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kGatherSmem);
    attr = true;
  }
  gather_kernel<<<dim3(n, 2 * kv_heads), kThreads, kGatherSmem, at::cuda::getCurrentCUDAStream()>>>(
      src_blocks.data_ptr<int32_t>(), rec_addr.data_ptr<int64_t>(), rec_len.data_ptr<int32_t>(),
      reinterpret_cast<const uint32_t*>(plut.data_ptr()), slut.data_ptr<uint8_t>(),
      reinterpret_cast<uint8_t*>(wk.data_ptr()), reinterpret_cast<uint8_t*>(wv.data_ptr()),
      kv_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_export(torch::Tensor blocks, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor out, int64_t stride, torch::Tensor lengths, int64_t digest) {
  const int64_t n = blocks.numel();
  if (n == 0) return;
  const int L = rec_addr.size(0), NB = rec_addr.size(1), H = rec_addr.size(3);
  TORCH_CHECK(blocks.scalar_type() == torch::kInt32 && lengths.scalar_type() == torch::kInt64 &&
                  lengths.numel() >= n,
              "export blocks must be int32 and lengths int64 [>= n]");
  TORCH_CHECK(out.scalar_type() == torch::kUInt8 && out.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "export buffer must be contiguous 16-byte aligned uint8");
  TORCH_CHECK(stride % 16 == 0 &&
                  stride >= blob_head_bytes(L, H) + 2LL * L * H * kRawRecord &&
                  out.numel() >= n * stride,
              "export stride or buffer too small");
  blob_export_kernel<<<dim3(n, L), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      blocks.data_ptr<int32_t>(), rec_addr.data_ptr<int64_t>(), rec_len.data_ptr<int32_t>(),
      out.data_ptr<uint8_t>(), stride, lengths.data_ptr<int64_t>(), L, H, NB, digest);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void hbm_import(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill,
                torch::Tensor src, torch::Tensor offsets, torch::Tensor blocks,
                torch::Tensor status, torch::Tensor arena_base, torch::Tensor arena_stride,
                int64_t digest) {
  const int64_t n = blocks.numel();
  if (n == 0) return;
  Store s = make_store(counters, wslot, tags, free_stack, ovf_bits, claim_list, close_list,
                       rec_addr, rec_len, staging, scratch, overflow, fill);
  TORCH_CHECK(s.H <= kMaxHeads, "too many KV heads");
  TORCH_CHECK(src.scalar_type() == torch::kUInt8 && src.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0,
              "import buffer must be contiguous 16-byte aligned uint8");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64 && offsets.numel() == n &&
                  blocks.scalar_type() == torch::kInt32 && status.scalar_type() == torch::kInt32 &&
                  status.numel() >= n,
              "import metadata must be int64 offsets, int32 blocks / status");
  TORCH_CHECK(arena_base.numel() == s.L && arena_stride.numel() == s.L, "arena tables must hold every layer");
  blob_import_kernel<<<dim3(n, s.L), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      s, src.data_ptr<uint8_t>(), src.numel(), offsets.data_ptr<int64_t>(),
      blocks.data_ptr<int32_t>(), status.data_ptr<int32_t>(), arena_base.data_ptr<int64_t>(),
      arena_stride.data_ptr<int64_t>(), digest);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
