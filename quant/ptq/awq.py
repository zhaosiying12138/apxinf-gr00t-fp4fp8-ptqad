"""AWQ-style activation-aware per-channel scale search (Lin et al. 2023).

For every foldable site (see folds.py) we search s = absmean(x)^alpha over an
alpha grid, judging candidates by EXACT layer output MSE via the calibration
Hessian identity  sum_i ||x_i dW^T||^2 = tr(dW H dW^T).

Site kinds:
  norm          q/k/v (or gate/up) share one input; scale folds into the
                preceding RMSNorm/LayerNorm gain. Runtime input becomes x*s,
                weights quantized as quant(W / s):  dW_eff = s * quant(W/s) - W
  producer_cols o_proj folds into v_proj output cols (GQA-shared), down_proj
                into up_proj cols, vision proj into the fused-qkv v-slice.
                Producer: dW_eff = quant(Wp*s) - Wp*s   (its H unchanged)
                Consumer: dW_eff = sb*quant(Wc/sb) - Wc (H unchanged).
                The sum is a local producer/consumer proxy, not the exact
                end-to-end error of their jointly quantized composition.

s is normalized to mean 1 (any constant factor cancels in the fold).
"""
import torch
from quantizers import nvfp4_dequant, layer_mse_tr

ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)


def _gqa_bidx(Kc, gqa_groups, head_dim, device):
    c = torch.arange(Kc, device=device)
    if gqa_groups == 1:
        return c
    h, d = c // head_dim, c % head_dim
    return (h // gqa_groups) * head_dim + d


@torch.no_grad()
def search_site(site, get_W, calib, clip=1.0, alphas=ALPHAS, verbose=False):
    """get_W(name) -> f32 (N,K) original weight on CPU or CUDA. Returns dict result."""
    kind = site["kind"]
    if kind == "norm":
        m0 = site["members"][0]
        first_W = get_W(m0)
        dev = first_W.device
        a = (calib[m0]["abs"] / calib[m0]["n"]).to(dev)
        K = a.shape[0]
        layers = []
        for m in site["members"]:
            W = first_W if m == m0 else get_W(m)
            H = calib[m]["H"].to(dev)
            layers.append((m, W, H))
        def err_for(s):
            tot = 0.0
            for m, W, H in layers:
                Wq = nvfp4_dequant(W / s.unsqueeze(0), clip=clip)
                dW = Wq * s.unsqueeze(0) - W
                tot += layer_mse_tr(dW, H).item()
            return tot
        n_ch = K
    else:  # producer_cols
        pc, cc = site["producer"], site["consumer"]
        Wp, Wc = get_W(pc), get_W(cc)
        dev = Wp.device
        Hp, Hc = calib[pc]["H"].to(dev), calib[cc]["H"].to(dev)
        sl = site.get("pcol_slice")
        n_prod = (sl[1] - sl[0]) if sl else Wp.shape[0]
        # s lives on producer OUTPUT channels (the shared representation)
        ac = (calib[cc]["abs"] / calib[cc]["n"]).to(dev)   # consumer input absmean
        Kc = ac.shape[0]
        gg, hd = site.get("gqa_groups", 1), site.get("head_dim") or 0
        if gg > 1:
            bidx = _gqa_bidx(Kc, gg, hd, dev)
            # protect-more: max over the q-head copies sharing a kv channel
            a_prod = torch.zeros(n_prod, device=dev)
            a_prod.scatter_reduce_(0, bidx, ac, reduce="amax", include_self=True)
        else:
            a_prod = ac
        def _apply_p(W, s):
            if sl is not None:
                W = W.clone(); W[sl[0]:sl[1], :] = W[sl[0]:sl[1], :] * s.unsqueeze(1)
                return W
            return W * s.unsqueeze(1)
        def _apply_c(W, sb):
            return W / sb.unsqueeze(0)
        def err_for(s):
            sb = s[_gqa_bidx(Kc, gg, hd, dev)] if gg > 1 else s
            Wpq = nvfp4_dequant(_apply_p(Wp, s), clip=clip)
            dWp = Wpq - _apply_p(Wp, s)
            e_p = layer_mse_tr(dWp, Hp).item()
            Wcq = nvfp4_dequant(_apply_c(Wc, sb), clip=clip)
            dWc = Wcq * sb.unsqueeze(0) - Wc
            e_c = layer_mse_tr(dWc, Hc).item()
            return e_p + e_c
        n_ch = n_prod
        a = a_prod
    # Folding requires invertible scales, including channels absent in calibration.
    a = a.clamp_min(1e-12)
    best = (None, None)
    for alpha in alphas:
        s = a.pow(alpha)
        s = s / s.mean().clamp_min(1e-12)
        e = err_for(s)
        if best[1] is None or e < best[1]:
            best = (alpha, e)
    alpha, err = best
    s = a.pow(alpha)
    s = s / s.mean().clamp_min(1e-12)
    return {"alpha": alpha, "err": err, "s": s.cpu(), "n_ch": n_ch}
