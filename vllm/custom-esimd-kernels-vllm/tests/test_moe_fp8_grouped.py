import math, time, torch
import custom_esimd_kernels_vllm
import custom_esimd_kernels_vllm.moe_int4_prefill_ops
dev=torch.device("xpu")
def gelu_tanh(t): return 0.5*t*(1.0+torch.tanh(math.sqrt(2/math.pi)*(t+0.044715*t**3)))
def ref_moe(x,w_,idx_,w13,w2,gu_s,dn_s,top_k,inter):
    out=torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(top_k):
            eid=idx_[t,k].item()
            g=(w13[eid].to(torch.float16)*float(gu_s[eid]))@x[t]
            gate,up=g.split(inter); mid=gelu_tanh(gate.float()).to(torch.float16)*up
            out[t]+=((w2[eid].to(torch.float16)*float(dn_s[eid]))@mid)*w_[t,k]
    return out
E,K,H,I=128,8,2816,704; sc=0.3
for T in [256]:
    torch.manual_seed(42)
    x=torch.randn(T,H,dtype=torch.float16,device=dev)*0.05
    logits=torch.randn(T,E,dtype=torch.float16,device=dev)
    probs=torch.softmax(logits.float(),-1); pw,pi=torch.topk(probs,K,-1)
    pw=(pw/pw.sum(-1,keepdim=True)).to(torch.float16); pi=pi.to(torch.int32)
    w13=(torch.randn(E,2*I,H,device=dev)*0.05).clamp(-sc,sc).div(sc).to(torch.float8_e4m3fn)
    w2=(torch.randn(E,H,I,device=dev)*0.05).clamp(-sc,sc).div(sc).to(torch.float8_e4m3fn)
    gu_s=torch.full((E,),sc,dtype=torch.float32,device=dev); dn_s=torch.full((E,),sc,dtype=torch.float32,device=dev)
    r=ref_moe(x,pw,pi,w13,w2,gu_s,dn_s,K,I)
    # gather
    go=torch.ops.moe_int4_prefill_ops.moe_prefill_gather_forward_v2(pi.contiguous(),E)
    expert_offsets, expert_tokens = go[0], go[1]
    # flat routing weights pair-indexed: pair = token*K + k
    rw_flat = pw.reshape(-1).contiguous()  # [T*K], pair-indexed
    w13u=w13.view(torch.uint8); w2u=w2.view(torch.uint8)
    def run():
        inter=torch.ops.moe_ops.moe_up_fp8_grouped(x.contiguous(),w13u,gu_s,expert_offsets,expert_tokens,K,E)
        outp=torch.ops.moe_ops.moe_down_fp8_grouped(inter,w2u,dn_s,rw_flat,expert_offsets,expert_tokens,K,E)
        return outp.view(T,K,H).sum(1)
    o=run(); torch.xpu.synchronize()
    diff=(o.float()-r.float()).abs().max().item(); isnan=bool(torch.isnan(o).any())
    for _ in range(3): run()
    torch.xpu.synchronize(); t0=time.perf_counter()
    for _ in range(10): run()
    torch.xpu.synchronize(); dt=(time.perf_counter()-t0)/10*1000
    print("GROUPED T=%d max_diff=%.3e nan=%s %.2fms/call %s"%(T,diff,isnan,dt,"OK" if diff<3e-2 else "FAIL"))
