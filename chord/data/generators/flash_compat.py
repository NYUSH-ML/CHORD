"""PyTorch SDPA fallback for official DLM repos that hard-import flash-attn."""

from __future__ import annotations

import importlib.util
import sys
import types


def install_if_missing() -> bool:
    if importlib.util.find_spec("flash_attn") is not None:
        return False
    import torch
    import torch.nn.functional as F

    package = types.ModuleType("flash_attn")
    package.__path__ = []
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    layers = types.ModuleType("flash_attn.layers")
    layers.__path__ = []
    rotary = types.ModuleType("flash_attn.layers.rotary")

    def rotate_half(value):
        left, right = value.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def apply_rotary_emb_qkv_(qkv, cos, sin):
        full_cos = torch.cat((cos, cos), dim=-1)[None, :, None, None, :]
        full_sin = torch.cat((sin, sin), dim=-1)[None, :, None, None, :]
        qkv[:, :, :2] = qkv[:, :, :2] * full_cos + rotate_half(qkv[:, :, :2]) * full_sin
        return qkv

    def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p, causal=False):
        del max_seqlen
        outputs = []
        for index in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[index].item())
            stop = int(cu_seqlens[index + 1].item())
            q, k, v = qkv[start:stop].unbind(dim=1)
            value = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                v.transpose(0, 1)[None],
                dropout_p=float(dropout_p),
                is_causal=bool(causal),
            )
            outputs.append(value[0].transpose(0, 1))
        return torch.cat(outputs, dim=0)

    rotary.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_
    interface.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    layers.rotary = rotary
    package.layers = layers
    package.flash_attn_interface = interface
    # transformers probes flash_attn via importlib.util.find_spec, which raises
    # on a module whose __spec__ is None — give every stub a real spec.
    from importlib.machinery import ModuleSpec

    for name, module in [
        ("flash_attn", package),
        ("flash_attn.flash_attn_interface", interface),
        ("flash_attn.layers", layers),
        ("flash_attn.layers.rotary", rotary),
    ]:
        module.__spec__ = ModuleSpec(name, loader=None, is_package=hasattr(module, "__path__"))
    # advertise a version older than 2.x so transformers does not select the
    # flash-attention kernels from the stub
    package.__version__ = "0.0.0-sdpa-stub"
    sys.modules["flash_attn"] = package
    sys.modules["flash_attn.flash_attn_interface"] = interface
    sys.modules["flash_attn.layers"] = layers
    sys.modules["flash_attn.layers.rotary"] = rotary
    return True
