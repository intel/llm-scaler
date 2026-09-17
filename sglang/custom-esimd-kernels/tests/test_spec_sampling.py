"""Differential test for the XPU tree_speculative_sampling_target_only.

The reference below is a line-by-line transcription of the CUDA kernel
(sgl-kernel/csrc/speculative/speculative_sampling.cuh ::
TreeSpeculativeSamplingTargetOnly) in plain python. The XPU implementation
splits the same algorithm into a SYCL walk plus a torch residual draw, so the
two must agree exactly on the accept/reject decisions and on every token
written.

Run inside the runtime container:
    python tests/test_spec_sampling.py
"""

import sys

import torch

from custom_esimd_kernels_sglang.spec_sampling import (
    top_k_renorm_prob,
    top_p_renorm_prob,
    tree_speculative_sampling_target_only,
)

DEV = "xpu"


def reference(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
):
    bs, num_draft_tokens, d = target_probs.shape
    num_spec_step = accept_index.shape[1]
    tp = target_probs.reshape(-1, d)
    dp = draft_probs.reshape(-1, d)

    for bx in range(bs):
        prob_acc = 0.0
        cur_row = bx * num_draft_tokens
        coin = float(uniform_samples[bx, 0])
        last_accepted = int(retrive_index[bx, 0])
        accept_index[bx, 0] = last_accepted
        num_accepted = 0
        cur_index = 0

        for _ in range(1, num_spec_step):
            cur_index = int(retrive_next_token[bx, cur_index])
            while cur_index != -1:
                draft_index = int(retrive_index[bx, cur_index])
                draft_token_id = int(candidates[bx, cur_index])
                target_prob_single = float(tp[cur_row, draft_token_id])
                prob_acc += target_prob_single
                if coin <= prob_acc / threshold_acc or target_prob_single >= threshold_single:
                    prob_acc = 0.0
                    cur_row = bx * num_draft_tokens + cur_index
                    coin = float(uniform_samples[bx, cur_index])
                    predicts[last_accepted] = draft_token_id
                    num_accepted += 1
                    accept_index[bx, num_accepted] = draft_index
                    last_accepted = draft_index
                    break
                else:
                    dp[cur_row, draft_token_id] = tp[cur_row, draft_token_id]
                    cur_index = int(retrive_next_sibling[bx, cur_index])
            if cur_index == -1:
                break

        accept_token_num[bx] = num_accepted

        # final draw from relu(target - draft)
        coin = float(uniform_samples_for_final_sampling[bx])
        q = tp[cur_row].double()
        p = dp[cur_row].double()
        if num_accepted == num_spec_step - 1:
            p = torch.zeros_like(p)
        resid = (q - p).clamp_min(0)
        u = coin * float(resid.sum())
        agg = 0.0
        sampled_id = d
        last_valid = -1
        for i in range(d):
            v = float(resid[i])
            agg += v
            if v > 0:
                last_valid = i
                if agg > u and sampled_id == d:
                    sampled_id = i
        if sampled_id == d:
            sampled_id = last_valid if last_valid != -1 else d - 1
        predicts[last_accepted] = sampled_id


def make_chain(bs, num_draft_tokens):
    """topk=1: every node has exactly one child, no siblings."""
    ri = torch.arange(bs * num_draft_tokens, dtype=torch.int64).reshape(bs, num_draft_tokens)
    nt = torch.full((bs, num_draft_tokens), -1, dtype=torch.int64)
    nt[:, : num_draft_tokens - 1] = torch.arange(1, num_draft_tokens, dtype=torch.int64)
    ns = torch.full((bs, num_draft_tokens), -1, dtype=torch.int64)
    parent = [-1] + list(range(num_draft_tokens - 1))
    return ri, nt, ns, parent


def make_tree(bs):
    """A 7-node binary tree: 0 -> (1,2); 1 -> (3,4); 2 -> (5,6)."""
    n = 7
    ri = torch.arange(bs * n, dtype=torch.int64).reshape(bs, n)
    nt = torch.tensor([1, 3, 5, -1, -1, -1, -1], dtype=torch.int64).repeat(bs, 1)
    ns = torch.tensor([-1, 2, -1, 4, -1, 6, -1], dtype=torch.int64).repeat(bs, 1)
    parent = [-1, 0, 0, 1, 1, 2, 2]
    return ri, nt, ns, parent


def run_case(name, bs, num_draft_tokens, num_spec_step, d, builder, seed,
             threshold_single=1.0, threshold_acc=1.0, sharpness=1.0, likely=False):
    torch.manual_seed(seed)
    ri, nt, ns, parent = builder

    logits = torch.randn(bs, num_draft_tokens, d) * sharpness
    target_probs = torch.softmax(logits, dim=-1).float()
    if likely:
        # Draft the tokens the parent node actually considers probable, so the
        # rejection test accepts often. Uniformly random candidates have
        # vanishing target mass and would only ever exercise the reject path.
        candidates = torch.randint(0, d, (bs, num_draft_tokens), dtype=torch.int64)
        for i, par in enumerate(parent):
            if par < 0:
                continue
            top = target_probs[:, par].topk(4, dim=-1).indices
            candidates[:, i] = top[torch.arange(bs), i % 4]
    else:
        candidates = torch.randint(0, d, (bs, num_draft_tokens), dtype=torch.int64)
    uniform = torch.rand(bs, num_draft_tokens, dtype=torch.float32)
    uniform_final = torch.rand(bs, dtype=torch.float32)

    tot = bs * num_draft_tokens
    ref_pred = torch.zeros(tot, dtype=torch.int32)
    ref_ai = torch.full((bs, num_spec_step), -1, dtype=torch.int32)
    ref_atn = torch.zeros(bs, dtype=torch.int32)
    reference(
        ref_pred, ref_ai, ref_atn, candidates, ri, nt, ns, uniform, uniform_final,
        target_probs.clone(), torch.zeros_like(target_probs),
        threshold_single, threshold_acc,
    )

    xpu_pred = torch.zeros(tot, dtype=torch.int32, device=DEV)
    xpu_ai = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=DEV)
    xpu_atn = torch.zeros(bs, dtype=torch.int32, device=DEV)
    tree_speculative_sampling_target_only(
        predicts=xpu_pred,
        accept_index=xpu_ai,
        accept_token_num=xpu_atn,
        candidates=candidates.to(DEV),
        retrive_index=ri.to(DEV),
        retrive_next_token=nt.to(DEV),
        retrive_next_sibling=ns.to(DEV),
        uniform_samples=uniform.to(DEV),
        uniform_samples_for_final_sampling=uniform_final.to(DEV),
        target_probs=target_probs.to(DEV).contiguous(),
        draft_probs=torch.zeros_like(target_probs, device=DEV),
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
    )
    torch.xpu.synchronize()

    ok = True
    for label, got, want in (
        ("accept_token_num", xpu_atn.cpu(), ref_atn),
        ("accept_index", xpu_ai.cpu(), ref_ai),
    ):
        if not torch.equal(got, want):
            ok = False
            print(f"  [FAIL] {label}\n    xpu={got.tolist()}\n    ref={want.tolist()}")

    # Only slots the walk actually wrote are defined; the CUDA op documents the
    # rest as undefined, so compare exactly those positions.
    written = set()
    for b in range(bs):
        n = int(ref_atn[b])
        for j in range(n + 1):
            written.add(int(ref_ai[b, j]))
    written = sorted(written)
    g = xpu_pred.cpu()[written]
    w = ref_pred[written]
    if not torch.equal(g, w):
        ok = False
        bad = [(i, int(a), int(b)) for i, a, b in zip(written, g, w) if a != b]
        print(f"  [FAIL] predicts mismatch at {bad[:8]}")

    print(f"{'PASS' if ok else 'FAIL'}  {name}  (accepted={ref_atn.tolist()})")
    return ok, int(ref_atn.sum())


def test_renorm():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(6, 512), dim=-1).to(DEV)
    ok = True

    k = torch.tensor([1, 3, 8, 50, 512, 2], device=DEV)
    out = top_k_renorm_prob(probs.clone(), k).cpu()
    for i in range(6):
        nz = int((out[i] > 0).sum())
        if nz != min(int(k[i]), 512):
            ok = False
            print(f"  [FAIL] top_k row {i}: kept {nz}, want {int(k[i])}")
        want_idx = torch.topk(probs[i].cpu(), min(int(k[i]), 512)).indices.sort().values
        got_idx = (out[i] > 0).nonzero().flatten().sort().values
        if not torch.equal(want_idx, got_idx):
            ok = False
            print(f"  [FAIL] top_k row {i}: wrong index set")
    if not torch.allclose(out.sum(-1), torch.ones(6), atol=1e-5):
        ok = False
        print(f"  [FAIL] top_k rows do not sum to 1: {out.sum(-1)}")

    p = torch.tensor([0.1, 0.5, 0.9, 0.95, 1.0, 0.01], device=DEV)
    out = top_p_renorm_prob(probs.clone(), p).cpu()
    for i in range(6):
        sv = probs[i].cpu().sort(descending=True).values
        excl = sv.cumsum(0) - sv
        want = int((excl < float(p[i])).sum())
        nz = int((out[i] > 0).sum())
        if nz != want:
            ok = False
            print(f"  [FAIL] top_p row {i}: kept {nz}, want {want}")
    if not torch.allclose(out.sum(-1), torch.ones(6), atol=1e-5):
        ok = False
        print(f"  [FAIL] top_p rows do not sum to 1: {out.sum(-1)}")

    print(f"{'PASS' if ok else 'FAIL'}  renorm (top_k / top_p)")
    return ok


if __name__ == "__main__":
    results = [test_renorm()]
    accepts = 0

    def case(*a, **kw):
        global accepts
        ok, n = run_case(*a, **kw)
        accepts += n
        results.append(ok)

    # Reject-dominated: uniformly random candidates almost never match.
    for seed in range(4):
        case(f"chain reject-heavy seed={seed}", 4, 4, 4, 128, make_chain(4, 4), seed)
    for seed in range(3):
        case(f"tree  reject-heavy seed={seed}", 3, 7, 3, 96, make_tree(3), seed)

    # Accept-dominated: production config (--speculative-eagle-topk 1 => chain
    # of 4) with peaked distributions and plausible drafts. This is the path
    # that actually runs in decode, and it exercises accept, prob-row descent
    # and the all-accepted bonus-token branch.
    for seed in range(6):
        case(f"chain accept-heavy seed={seed}", 4, 4, 4, 128, make_chain(4, 4), seed,
             sharpness=6.0, likely=True)
    for seed in range(4):
        case(f"tree  accept-heavy seed={seed}", 3, 7, 3, 96, make_tree(3), seed,
             sharpness=6.0, likely=True)

    # threshold_single forces accepts regardless of the coin; threshold_acc < 1
    # loosens the rejection test.
    case("chain thresholds", 4, 4, 4, 128, make_chain(4, 4), 11,
         threshold_single=0.05, threshold_acc=0.5, sharpness=4.0, likely=True)
    # Large-ish vocab closer to production.
    case("chain d=4096 accept-heavy", 2, 4, 4, 4096, make_chain(2, 4), 21,
         sharpness=6.0, likely=True)
    case("chain d=4096 reject-heavy", 2, 4, 4, 4096, make_chain(2, 4), 21)

    print(f"\n{sum(results)}/{len(results)} passed; total tokens accepted across "
          f"cases = {accepts} (0 would mean the accept path was never exercised)")
    sys.exit(0 if all(results) else 1)
