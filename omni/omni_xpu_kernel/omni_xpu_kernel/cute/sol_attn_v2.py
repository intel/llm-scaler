"""Quantized Sol-Attn orchestration over the native CUTE/SYCL operators."""
from __future__ import annotations

import math
import torch


def _ops():
    from . import _ensure_loaded
    _ensure_loaded()
    return torch.ops.omni_xpu_sol_attn


def _prepare_chunked(qkv_chunks,t,h,rope_freqs,qk_norm_weights,kmean,vscale,
                     scale,tau,rope_eps,block_len):
    """Keep normalized Q/K inside each chunk and full carriers in INT8."""
    from . import _prepare_bmg_policy_dispatch
    from .. import rotary
    if isinstance(t,bool) or isinstance(h,bool) or not isinstance(t,int) or not isinstance(h,int) or t<=0 or h<=0:
        raise ValueError('sol_attn_chunked requires positive integer T and H')
    if not isinstance(rope_freqs,torch.Tensor) or rope_freqs.device.type!='xpu' or rope_freqs.ndim<3:
        raise ValueError('rope_freqs must be a floating XPU tensor of per-token 2x2 transforms')
    rot=rope_freqs.shape[-3]*2
    if rope_freqs.shape[-2:]!=(2,2) or rot%8 or not 0<rot<=128 or rope_freqs.numel()!=t*rot*2:
        raise ValueError('rope_freqs must cover T tokens, with rot_dim a multiple of 8 in (0,128]')
    if rope_freqs.dtype not in (torch.float32,torch.float16,torch.bfloat16):
        raise ValueError('rope_freqs must have FP32, FP16 or BF16 dtype')
    if not math.isfinite(rope_eps) or rope_eps<0:
        raise ValueError('rope_eps must be finite and nonnegative')
    dev=rope_freqs.device
    _prepare_bmg_policy_dispatch(rope_freqs)
    if len(qk_norm_weights)!=2 or any(not isinstance(w,torch.Tensor) or w.shape!=(128,) or not w.is_floating_point() for w in qk_norm_weights):
        raise ValueError('qk_norm_weights must contain two floating [128] tensors')
    weights=tuple(w.to(device=dev,dtype=torch.bfloat16).contiguous() for w in qk_norm_weights)
    freqs=rope_freqs.reshape(1,t,1,rot//2,2,2)
    n=(t+63)//64
    if block_len is None: lengths=torch.empty(0,device=dev,dtype=torch.int32)
    elif block_len.device!=dev or block_len.dtype!=torch.int32 or block_len.shape!=(n,):
        raise ValueError('block_len must be int32 [ceil(T/64)] on the input XPU')
    else:lengths=block_len.contiguous()
    for name,value in (('kmean',kmean),('vscale',vscale)):
        if value is not None and (not isinstance(value,torch.Tensor) or value.shape!=(h,128) or not value.is_floating_point()):
            raise ValueError(f'{name} must be a floating [H,128] tensor')
    ops=_ops()
    if not all(hasattr(ops,name) for name in ('producer_begin','producer_chunk','producer_finish')):
        raise RuntimeError('the installed Sol sidecar does not provide the native chunk producer')
    factory=qkv_chunks if callable(qkv_chunks) else None
    if factory is None and (kmean is None or vscale is None):
        retained=list(qkv_chunks)
        factory=lambda:iter(retained)

    def produce(km,vs):
        c=ops.producer_begin(rope_freqs,t,h,vs)
        offset=0
        for chunk in (factory() if factory is not None else qkv_chunks):
            if not isinstance(chunk,torch.Tensor) or chunk.ndim!=2 or chunk.shape[1]!=3*h*128 or chunk.dtype!=torch.bfloat16 or chunk.device!=dev:
                raise ValueError(f'projection chunks must be [M,{3*h*128}] BF16 on {dev}')
            m=chunk.shape[0]
            if not m:continue
            if offset%64 or offset+m>t or (m%64 and offset+m!=t):
                raise ValueError('chunks must cover whole 64-token blocks, except the final ragged block')
            # A private bounded copy preserves the caller's projection and keeps
            # the maintained packed H3 RMS/RoPE fast-path strides available.
            packed=chunk.clone(memory_format=torch.contiguous_format).view(1,m,3,h,128)
            q,k,v=packed.unbind(2)
            rotary.rms_kitchen_rope_split_half_(q,k,freqs[:,offset:offset+m],
                *weights,epsilon=float(rope_eps),rot_dim=rot)
            ops.producer_chunk(q,k,v,c,km,offset,lengths)
            offset+=m
        if offset!=t:raise ValueError(f'projection chunks cover {offset} tokens, expected T={t}')
        ops.producer_finish(c,float(scale),float(tau),lengths)
        return c

    if kmean is None or vscale is None:
        bootstrap=produce(torch.zeros((h,128),device=dev),torch.ones((h,128),device=dev))
        kmean,vscale=bootstrap[17][0],bootstrap[18][0]
        del bootstrap
    kmean=kmean.to(device=dev,dtype=torch.float32).contiguous()
    vscale=vscale.to(device=dev,dtype=torch.float32).clamp_min(1e-8).contiguous()
    return produce(kmean,vscale),lengths


def _finish(c,scale,sinks,sink_q,topk,tail,token_aug,lengths,key_bias=None,fp16=False):
    ops=_ops()
    scores=ops.centroid_scores(c[6],c[8],c[7],c[9],scale)
    routes,state,refs,common=ops.pooled_routes(scores,c[15],c[14],c[5],lengths,c[0].shape[1],
        *sinks,*sink_q,topk,tail,bool(token_aug))
    indices=counts=None
    if token_aug:
        gq,gqs,gref=ops.token_group_centroids(c[10],refs)
        hist=ops.token_histogram(gq,gqs,gref,c[1],c[4],common,scale)
        cut=ops.token_bin_cutoff(hist,token_aug)
        indices,counts,group=ops.token_remainder(gq,gqs,gref,c[1],c[4],c[2],common,cut,scale,token_aug,tail)
        indices=ops.sort_token_indices(indices,counts)
        state=ops.merge_token_tail(state,group)
        selected=ops.forward_cute_selected(*c[:6],routes,state,scale,indices,counts,key_bias)
        return ops.forward_cute_prepared_split(*c[:6],routes,state,selected,scale,key_bias,fp16)
    return ops.forward_cute_prepared(*c[:6],routes,state,scale,indices,counts,key_bias,fp16)


def is_available():
    """Whether the loaded sidecar provides the complete quantized Sol API."""
    try:
        ops=_ops()
        return all(hasattr(ops,name) for name in (
            'prepare_carriers','centroid_scores','pooled_routes','token_group_centroids',
            'token_histogram','token_bin_cutoff','token_remainder','sort_token_indices',
            'merge_token_tail','forward_cute_prepared','forward_cute_selected',
            'forward_cute_prepared_split','producer_begin','producer_chunk',
            'producer_finish','coarse_output','add_coarse_'))
    except (ImportError,OSError,RuntimeError):
        return False


def _controls(tokens,tau,scale,sink_blocks,sink_q,topk_ratio,token_aug):
    n=(tokens+63)//64
    tau=float(tau);scale=128**-0.5 if scale is None else float(scale)
    ratio=float(topk_ratio)
    if not math.isfinite(tau) or not math.isfinite(scale):
        raise ValueError('tau and scale must be finite')
    if ratio!=0.0 and not 0.0<ratio<1.0:
        raise ValueError('topk_ratio must be zero or in (0,1)')
    if not isinstance(token_aug,int) or token_aug<0 or token_aug>256 or token_aug%64:
        raise ValueError('token_aug must be zero or a multiple of 64 through 256')
    def pair(value,name):
        if value is None:return (0,0)
        if len(value)!=2 or any(not isinstance(x,int) or x<0 for x in value) or value[1]<value[0]:
            raise ValueError(f'{name} must contain nonnegative integer [start,end) block indices')
        return (min(value[0],n),min(value[1],n))
    sb,sq=pair(sink_blocks,'sink_blocks'),pair(sink_q,'sink_q')
    selectable=n-(sb[1]-sb[0])
    topk=max(0,min(selectable-1,max(1,round(ratio*selectable)))) if ratio else -1
    return tau,scale,sb,sq,topk,token_aug


def _key_bias(value,batch,tokens,device):
    if value is None:return None
    if not isinstance(value,torch.Tensor) or value.device!=device:
        raise ValueError('key_bias must be on the input XPU')
    if value.ndim==4:
        if value.shape[1:3]!=(1,1):raise ValueError('key_bias must not vary over heads or queries')
        value=value[:,0,0]
    if value.ndim==1:value=value.unsqueeze(0)
    if value.ndim!=2 or value.shape[0] not in (1,batch) or value.shape[1]!=tokens:
        raise ValueError('key_bias must be (T,), (B,T), or (B|1,1,1,T)')
    if value.dtype==torch.bool:value=torch.where(value,0.0,float('-inf'))
    elif not value.is_floating_point():raise ValueError('key_bias must be bool or floating point')
    return value.float().mul(math.log2(math.e)).expand(batch,tokens).contiguous()


def _gate(value,shape,device):
    if value is not None and (not isinstance(value,torch.Tensor) or value.device!=device or
            tuple(value.shape)!=tuple(shape) or value.dtype not in (torch.float32,torch.float16,torch.bfloat16)):
        raise ValueError('coarse_gate must have the input shape/device and FP32, FP16 or BF16 dtype')
    return value


def _add_coarse(output,c,scale,lengths,gate):
    if gate is not None:
        ops=_ops()
        coarse=ops.coarse_output(c[10],c[11],c[14],lengths,output.shape[1],scale)
        ops.add_coarse_(output,coarse,gate.contiguous())
    return output


def sol_attn(q,k,v,tau=1.0,scale=None,sink_blocks=None,sink_q=None,key_bias=None,
             topk_ratio=0.0,tail=True,block_len=None,coarse_gate=None,token_aug=0):
    """BF16/FP16 BTHD D128 Sol attention with native token augmentation."""
    from . import _prepare_bmg_policy_dispatch
    if not isinstance(q,torch.Tensor) or q.device.type!='xpu' or q.ndim!=4 or q.shape[-1]!=128 or min(q.shape[:3])<=0:
        raise ValueError('Q/K/V must be nonempty BTHD XPU tensors with D128')
    if q.dtype not in (torch.bfloat16,torch.float16):raise ValueError('Q/K/V must be BF16 or FP16')
    for x in (q,k,v):
        if not isinstance(x,torch.Tensor) or x.device!=q.device or x.dtype!=q.dtype or x.shape!=q.shape or x.stride(-1)!=1:
            raise ValueError('Q/K/V must share shape, dtype, XPU and a contiguous D axis')
    b,t,h,_=q.shape
    tau,scale,sb,sq,topk,token_aug=_controls(t,tau,scale,sink_blocks,sink_q,topk_ratio,token_aug)
    bias=_key_bias(key_bias,b,t,q.device)
    gate=_gate(coarse_gate,q.shape,q.device)
    if block_len is None:lengths=torch.empty(0,device=q.device,dtype=torch.int32)
    elif block_len.device!=q.device or block_len.dtype!=torch.int32 or block_len.shape!=((t+63)//64,):
        raise ValueError('block_len must be int32 [ceil(T/64)] on the input XPU')
    else:lengths=block_len.contiguous()
    _prepare_bmg_policy_dispatch(q)
    c=_ops().prepare_carriers(q,k,v,scale,tau,lengths)
    output=_finish(c,scale,sb,sq,topk,bool(tail),token_aug,lengths,bias,q.dtype==torch.float16)
    return _add_coarse(output,c,scale,lengths,gate)


def sol_attn_chunked(qkv_chunks,t,h,rope_freqs,qk_norm_weights,kmean=None,vscale=None,
                     tau=1.0,topk_ratio=0.0,scale=None,sink_blocks=None,sink_q=None,
                     rope_eps=1e-6,tail=True,block_len=None,coarse_gate=None,token_aug=0):
    """Return output and next K/V statistics from bounded BF16 projection chunks."""
    if not isinstance(t,int) or not isinstance(h,int) or t<=0 or h<=0:
        raise ValueError('sol_attn_chunked requires positive integer T and H')
    tau,scale,sb,sq,topk,token_aug=_controls(t,tau,scale,sink_blocks,sink_q,topk_ratio,token_aug)
    if not isinstance(rope_freqs,torch.Tensor):raise ValueError('rope_freqs must be an XPU tensor')
    gate=_gate(coarse_gate,(1,t,h,128),rope_freqs.device)
    c,lengths=_prepare_chunked(qkv_chunks,t,h,rope_freqs,qk_norm_weights,kmean,vscale,
                              scale,tau,float(rope_eps),block_len)
    output=_finish(c,scale,sb,sq,topk,bool(tail),token_aug,lengths)
    return _add_coarse(output,c,scale,lengths,gate),c[17][0],c[18][0]
