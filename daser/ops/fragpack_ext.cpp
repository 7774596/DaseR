// SPDX-License-Identifier: Apache-2.0
// Python bindings for the FragPack decode-attention and HBM KV store kernels.
#include <torch/extension.h>

void fused_decode(torch::Tensor q, torch::Tensor rec_addr, torch::Tensor rec_len,
                  torch::Tensor plut, torch::Tensor slut, torch::Tensor block_table,
                  torch::Tensor seq_lens, torch::Tensor opart, torch::Tensor ml,
                  torch::Tensor plan, torch::Tensor out, int64_t units, bool replan,
                  double scale);

void hbm_prepare(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                 torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                 torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                 torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                 torch::Tensor fill,
                 torch::Tensor slot_mapping, int64_t n_tok, torch::Tensor live, int64_t n_live,
                 int64_t step);

void hbm_write(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
               torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
               torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
               torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
               torch::Tensor fill,
               torch::Tensor key, torch::Tensor value, torch::Tensor slot_mapping,
               int64_t layer);

void hbm_encode(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill,
                torch::Tensor rank_of, int64_t layer, int64_t arena, int64_t arena_stride,
                int64_t max_close);

void hbm_commit(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill);

void hbm_gather(torch::Tensor src_blocks, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor plut, torch::Tensor slut, torch::Tensor wk, torch::Tensor wv,
                int64_t kv_heads);

void hbm_export(torch::Tensor blocks, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor out, int64_t stride, torch::Tensor lengths, int64_t digest);

void hbm_import(torch::Tensor counters, torch::Tensor wslot, torch::Tensor tags,
                torch::Tensor free_stack, torch::Tensor ovf_bits, torch::Tensor claim_list,
                torch::Tensor close_list, torch::Tensor rec_addr, torch::Tensor rec_len,
                torch::Tensor staging, torch::Tensor scratch, torch::Tensor overflow,
                torch::Tensor fill,
                torch::Tensor src, torch::Tensor offsets, torch::Tensor blocks,
                torch::Tensor status, torch::Tensor arena_base, torch::Tensor arena_stride,
                int64_t digest);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hbm_export", &hbm_export);
  m.def("hbm_import", &hbm_import);
  m.def("fused_decode", &fused_decode);
  m.def("hbm_prepare", &hbm_prepare);
  m.def("hbm_write", &hbm_write);
  m.def("hbm_encode", &hbm_encode);
  m.def("hbm_commit", &hbm_commit);
  m.def("hbm_gather", &hbm_gather);
}
