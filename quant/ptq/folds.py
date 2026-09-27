"""Fold map for GR00T N1.7 (Qwen3-VL backbone): where per-input-channel AWQ
scales can be folded EXACTLY by editing only weights/norm params.

Verified implementations (2026-09-28):
  - Qwen3VLTextRMSNorm: y = w * rms(x)          -> w *= s   (exact)
  - LayerNorm (vision/merger/vlln): y = g*x+b   -> g,b *= s (exact)
  - gated SiLU MLP (lang): down(silu(gate)*up)  -> s folds into up cols (exact)
  - attention out: y = softmax(QK)V @ Wo        -> s folds into V cols (exact,
    linear in V); GQA: v-channels shared by 2 q-heads (repeat_interleave)
  - vision MLP act_fn(fc1)->fc2 & merger fc2: NOT foldable (elementwise act)
  - lm_head: NO fold (final norm also feeds action-head vlln path; scaling it
    would perturb the DiT context stream)
Backbone = Qwen3-VL (user: \"qwen3.5-vl\"): lang 12L h2048 GQA16/8 head128,
vision 24L h1024 fused qkv, merger + 3 deepstack mergers.
"""
import re
import torch

LANG = r"^backbone\.model\.model\.language_model\.layers\.(\d+)\."
VIS  = r"^backbone\.model\.model\.visual\.blocks\.(\d+)\."

def build_sites(linear_names):
    """Return dict site_id -> spec. Members reference Linear names."""
    sites = {}
    def add(sid, kind, **kw):
        sites[sid] = {"kind": kind, **kw}
    langs = set()
    viss = set()
    for n in linear_names:
        m = re.match(LANG, n)
        if m: langs.add(int(m.group(1)))
        m = re.match(VIS, n)
        if m: viss.add(int(m.group(1)))
    L = "backbone.model.model.language_model.layers.%d."
    V = "backbone.model.model.visual.blocks.%d."
    for i in sorted(langs):
        p = L % i
        add(f"lang{i}.in", "norm", norm=p+"input_layernorm", rms=True,
            members=[p+"self_attn.q_proj", p+"self_attn.k_proj", p+"self_attn.v_proj"])
        add(f"lang{i}.post", "norm", norm=p+"post_attention_layernorm", rms=True,
            members=[p+"mlp.gate_proj", p+"mlp.up_proj"])
        add(f"lang{i}.vo", "producer_cols",
            producer=p+"self_attn.v_proj", pcols="out",
            consumer=p+"self_attn.o_proj", gqa_groups=2, head_dim=128)
        add(f"lang{i}.updown", "producer_cols",
            producer=p+"mlp.up_proj", pcols="out",
            consumer=p+"mlp.down_proj", gqa_groups=1, head_dim=None)
    for i in sorted(viss):
        p = V % i
        add(f"vis{i}.1", "norm", norm=p+"norm1", rms=False,
            members=[p+"attn.qkv"])
        add(f"vis{i}.2", "norm", norm=p+"norm2", rms=False,
            members=[p+"mlp.linear_fc1"])
        add(f"vis{i}.vp", "producer_cols",
            producer=p+"attn.qkv", pcols="out", pcol_slice=(2048, 3072),
            consumer=p+"attn.proj", gqa_groups=1, head_dim=None)
    # merger + deepstack mergers: fc1 fed by LayerNorm
    for base in ["backbone.model.model.visual.merger"] + [
            f"backbone.model.model.visual.deepstack_merger_list.{k}" for k in range(3)]:
        add(base.split("visual.")[-1].replace(".", "_"), "norm", norm=base+".norm",
            rms=False, members=[base+".linear_fc1"])
    return sites

def consumer_col_to_producer_col(c, gqa_groups, head_dim):
    """Map consumer input-channel index -> producer output-channel index."""
    if gqa_groups == 1:
        return c
    h, d = divmod(c, head_dim)
    return (h // gqa_groups) * head_dim + d

def broadcast_s_to_consumer(s_prod, Kc, gqa_groups, head_dim, device):
    """s defined on producer channels -> (Kc,) vector at consumer input."""
    if gqa_groups == 1:
        return s_prod
    idx = torch.arange(Kc, device=device)
    h, d = divmod_idx(idx, head_dim)
    pcol = (h // gqa_groups) * head_dim + d
    return s_prod[pcol]

def divmod_idx(idx, n):
    return idx // n, idx % n
