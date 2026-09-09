"""Metadata-only contracts for the upstream sparse device eligibility adapter."""
import ast
import importlib.util
import inspect
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH=Path(__file__).parents[1]/"ComfyUI-OmniXPU/adapters/sparse_attention.py"
spec=importlib.util.spec_from_file_location("sparse_adapter_test",_PATH)
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def generic(q,k,v,dim_head):
    if q.device.type != "cuda":
        return "not on CUDA"
    if dim_head != 128:return "dimension"
    return None


def eligible(attn,x,rope_freqs,transformer_options,patch,block_index):
    if rope_freqs is None or x.device.type != "cuda" or attn.head_dim != 128:
        return False
    return patch.dense_reason(transformer_options,x.shape[0],block_index) is None


def unsupported(q,k,v,dim_head):
    return q.device.type == "cuda"


def duplicate(q,k,v,dim_head):
    if q.device.type != "cuda":return False
    if q.device.type != "cuda":return False
    return True


@pytest.mark.parametrize("device",["xpu","cuda","cpu"])
def test_generic_preserves_other_conditions(device):
    changed=adapter._extend_device_guard(generic,"q")
    q=SimpleNamespace(device=SimpleNamespace(type=device))
    assert changed(q,q,q,128)==(None if device in ("xpu","cuda") else "not on CUDA or XPU")
    assert changed(q,q,q,64)==("dimension" if device in ("xpu","cuda") else "not on CUDA or XPU")
    assert generic(q,q,q,128)==(None if device=="cuda" else "not on CUDA")
    assert inspect.signature(changed)==inspect.signature(generic)
    assert changed.__globals__ is generic.__globals__


@pytest.mark.parametrize("device",["xpu","cuda","cpu"])
@pytest.mark.parametrize("dense",[False,True])
def test_h3_keeps_schedule_and_shape_rules(device,dense):
    changed=adapter._extend_device_guard(eligible,"x")
    x=SimpleNamespace(device=SimpleNamespace(type=device),shape=(15787,5376))
    calls=[]
    def dense_reason(options,tokens,block):
        calls.append((options,tokens,block))
        return "scheduled dense" if dense else None
    patch=SimpleNamespace(dense_reason=dense_reason)
    options={"uuids":("test",)};attn=SimpleNamespace(head_dim=128)
    assert changed(attn,x,object(),options,patch,3)==(device in ("xpu","cuda") and not dense)
    assert len(calls)==int(device in ("xpu","cuda"))
    assert changed(attn,x,None,options,patch,3) is False
    attn.head_dim=64
    assert changed(attn,x,object(),options,patch,3) is False


@pytest.mark.parametrize("function",[unsupported,duplicate])
def test_unrecognized_guard_fails_closed(function):
    with pytest.raises(ValueError,match="exactly one"):
        adapter._extend_device_guard(function,"q")


def test_apply_atomic_and_idempotent(monkeypatch):
    monkeypatch.setitem(sys.modules,"nodes",SimpleNamespace(NODE_CLASS_MAPPINGS={}))
    ck=types.ModuleType("comfy_kitchen");ck.sol_attn_is_available=lambda:True
    module=types.ModuleType("comfy_extras.nodes_sparse_attention")
    module._ineligible=generic;module.h3_eligible=unsupported
    untouched=object();module.h3_sparse_attention=untouched;module.SparseAttnPatch=untouched
    package=types.ModuleType("comfy_extras");package.nodes_sparse_attention=module
    monkeypatch.setitem(sys.modules,"comfy_kitchen",ck)
    monkeypatch.setitem(sys.modules,"comfy_extras",package)
    assert adapter.apply()[0] is False
    assert module._ineligible is generic and module.h3_eligible is unsupported
    module.h3_eligible=eligible
    assert adapter.apply()==(True,"")
    first=module._ineligible,module.h3_eligible
    assert adapter.apply()==(True,"already patched")
    assert (module._ineligible,module.h3_eligible)==first
    assert module.h3_sparse_attention is untouched and module.SparseAttnPatch is untouched


def test_apply_targets_registered_file_loader_module(monkeypatch,tmp_path):
    ck=types.ModuleType("comfy_kitchen");ck.sol_attn_is_available=lambda:True
    module=types.ModuleType("nodes_sparse_attention")
    module.__file__=str(tmp_path/"comfy_extras/nodes_sparse_attention.py")
    node=type("BlockSparseAttention",(),{"__module__":module.__name__})
    module.BlockSparseAttention=node
    module._ineligible=generic;module.h3_eligible=eligible
    canonical=types.ModuleType("comfy_extras.nodes_sparse_attention")
    canonical._ineligible=generic;canonical.h3_eligible=eligible
    package=types.ModuleType("comfy_extras");package.nodes_sparse_attention=canonical
    nodes=SimpleNamespace(__file__=str(tmp_path/"nodes.py"),NODE_CLASS_MAPPINGS={"BlockSparseAttention":node})
    for name,value in (("nodes",nodes),(module.__name__,module),("comfy_extras",package),("comfy_kitchen",ck)):
        monkeypatch.setitem(sys.modules,name,value)
    assert adapter.apply()==(True,"")
    q=SimpleNamespace(device=SimpleNamespace(type="xpu"))
    assert module._ineligible(q,q,q,128) is None
    assert canonical._ineligible is generic and canonical.h3_eligible is eligible
    assert adapter.apply()==(True,"already patched")


def test_registered_node_ownership_mismatch_fails_closed(monkeypatch):
    ck=types.ModuleType("comfy_kitchen");ck.sol_attn_is_available=lambda:True
    node=type("BlockSparseAttention",(),{"__module__":"missing_sparse_module"})
    monkeypatch.setitem(sys.modules,"nodes",SimpleNamespace(NODE_CLASS_MAPPINGS={"BlockSparseAttention":node}))
    monkeypatch.setitem(sys.modules,"comfy_kitchen",ck)
    assert adapter.apply()==(False,"upstream sparse eligibility is unsupported: registered sparse node has no owning module")


def test_missing_native_leaves_upstream_unmodified(monkeypatch):
    ck=types.ModuleType("comfy_kitchen");ck.sol_attn_is_available=lambda:False
    monkeypatch.setitem(sys.modules,"comfy_kitchen",ck)
    assert adapter.apply()==(False,"complete native XPU Sol API is unavailable")
