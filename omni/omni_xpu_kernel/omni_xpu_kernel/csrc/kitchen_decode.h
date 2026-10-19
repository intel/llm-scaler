#pragma once
#include <torch/extension.h>
#include <vector>

namespace omni_xpu::kitchen {
std::vector<torch::Tensor> decode_attention(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor lengths, bool causal);
void decode_step_merge(torch::Tensor out, torch::Tensor lse,
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor merged);
std::vector<torch::Tensor> decode_quantized(torch::Tensor q,torch::Tensor qi,torch::Tensor qs,
    torch::Tensor k,torch::Tensor v,torch::Tensor ks,torch::Tensor vs,
    torch::Tensor length,int64_t capacity);
void int8_decode_update(torch::Tensor key,torch::Tensor value,torch::Tensor k,torch::Tensor v,
    torch::Tensor ks,torch::Tensor vs,torch::Tensor length,int64_t capacity,bool initialize);
std::vector<torch::Tensor> int8_decode(torch::Tensor query,torch::Tensor k,torch::Tensor v,
    torch::Tensor ks,torch::Tensor vs,torch::Tensor length,int64_t capacity);
void deltanet_conv_deferred(torch::Tensor proj,torch::Tensor state,torch::Tensor weight,
    std::optional<torch::Tensor> bias,torch::Tensor proj_buf,torch::Tensor qkv_buf,torch::Tensor ctl);
torch::Tensor gated_delta_decode_deferred(torch::Tensor x,torch::Tensor wa,torch::Tensor wb,
    torch::Tensor dt,torch::Tensor gd,torch::Tensor state,int64_t key_dim,int64_t hk,double scale,
    torch::Tensor z,torch::Tensor nw,double eps,torch::Tensor qkv,torch::Tensor gates,torch::Tensor sums,torch::Tensor ctl);
}
