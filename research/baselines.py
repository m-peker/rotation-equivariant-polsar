"""Comparison baselines, re-implemented in PyTorch under a common interface.

The set follows the comparison used by Alkhatib (2025) so that our table is
directly readable against the literature: CV-MLP, CV-2DCNN, CV-3DCNN,
CV-2D-3D, a complex ViT, and CV-MsAtViT (in cvmsatvit.py). SVM is handled
separately in the experiment script.

All of them take the 7-channel equivariance-preserving representation, are
trained with the same schedule, and are evaluated on the same splits, so the
comparison isolates the model rather than the pipeline. Where a method's
official code exists we followed it; every model here is a re-implementation
and is labelled as such in the paper.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from polsar_lib import CConv, CLin, crelu, cpool


class CVMLP(nn.Module):
    """Per-pixel complex MLP: no spatial context (centre pixel only)."""
    def __init__(s, ncl, cin=7, w=15):
        super().__init__()
        s.m = w // 2
        s.f1 = CLin(cin, 128); s.f2 = CLin(128, 64); s.f3 = CLin(64, ncl)
        s.do = nn.Dropout(0.3)

    def forward(s, xr, xi):
        a = xr[:, :, s.m, s.m]; b = xi[:, :, s.m, s.m]
        a, b = crelu(*s.f1(a, b)); a, b = s.do(a), s.do(b)
        a, b = crelu(*s.f2(a, b)); a, b = s.f3(a, b)
        return torch.sqrt(a ** 2 + b ** 2 + 1e-9)


class CV2DCNN(nn.Module):
    """Spatial complex CNN; no explicit polarimetric-axis processing."""
    def __init__(s, ncl, cin=7):
        super().__init__()
        s.c1 = CConv(cin, 32); s.c2 = CConv(32, 64)
        s.f1 = CLin(64 * 3 * 3, 128); s.f2 = CLin(128, ncl); s.do = nn.Dropout(0.3)

    def forward(s, xr, xi):
        xr, xi = crelu(*s.c1(xr, xi)); xr, xi = cpool(xr, xi)      # 15->7
        xr, xi = crelu(*s.c2(xr, xi)); xr, xi = cpool(xr, xi)      # 7->3
        xr = s.do(xr.flatten(1)); xi = s.do(xi.flatten(1))
        xr, xi = crelu(*s.f1(xr, xi)); xr, xi = s.f2(xr, xi)
        return torch.sqrt(xr ** 2 + xi ** 2 + 1e-9)


class CConv3d(nn.Module):
    def __init__(s, i, o, k):
        super().__init__()
        p = tuple(x // 2 for x in k)
        s.r = nn.Conv3d(i, o, k, padding=p); s.i = nn.Conv3d(i, o, k, padding=p)

    def forward(s, a, b):
        return s.r(a) - s.i(b), s.r(b) + s.i(a)


class CV3DCNN(nn.Module):
    """Joint spatial-polarimetric complex 3-D CNN."""
    def __init__(s, ncl, cin=7):
        super().__init__()
        s.cin = cin
        s.c1 = CConv3d(1, 8, (3, 3, 3)); s.c2 = CConv3d(8, 16, (3, 3, 3))
        s.f1 = CLin(16 * 3 * 3 * cin, 128); s.f2 = CLin(128, ncl)
        s.do = nn.Dropout(0.3)

    def _p(s, t):                       # (B,C,H,W) -> (B,1,H,W,C)
        return t.permute(0, 2, 3, 1).unsqueeze(1)

    def forward(s, xr, xi):
        a, b = s._p(xr), s._p(xi)
        a, b = crelu(*s.c1(a, b))
        a = F.avg_pool3d(a, (2, 2, 1)); b = F.avg_pool3d(b, (2, 2, 1))
        a, b = crelu(*s.c2(a, b))
        a = F.avg_pool3d(a, (2, 2, 1)); b = F.avg_pool3d(b, (2, 2, 1))
        a = s.do(a.flatten(1)); b = s.do(b.flatten(1))
        a, b = crelu(*s.f1(a, b)); a, b = s.f2(a, b)
        return torch.sqrt(a ** 2 + b ** 2 + 1e-9)


class CV2D3D(nn.Module):
    """Hybrid: a 3-D stage over the polarimetric axis, then a 2-D stage."""
    def __init__(s, ncl, cin=7):
        super().__init__()
        s.c3 = CConv3d(1, 8, (3, 3, 3))
        s.c2 = CConv(8 * cin, 64)
        s.f1 = CLin(64 * 3 * 3, 128); s.f2 = CLin(128, ncl); s.do = nn.Dropout(0.3)

    def forward(s, xr, xi):
        B, C, H, W = xr.shape
        a = xr.permute(0, 2, 3, 1).unsqueeze(1)
        b = xi.permute(0, 2, 3, 1).unsqueeze(1)
        a, b = crelu(*s.c3(a, b))                       # (B,8,H,W,C)
        a = a.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        b = b.permute(0, 1, 4, 2, 3).reshape(B, -1, H, W)
        a, b = crelu(*s.c2(a, b))
        a, b = cpool(a, b); a, b = cpool(a, b)          # 15->7->3
        a = s.do(a.flatten(1)); b = s.do(b.flatten(1))
        a, b = crelu(*s.f1(a, b)); a, b = s.f2(a, b)
        return torch.sqrt(a ** 2 + b ** 2 + 1e-9)


class CVViT(nn.Module):
    """Complex ViT on 3x3 patches, real/imag attention as in the reference work."""
    def __init__(s, ncl, cin=7, ws=15, patch=3, dim=64, heads=4, layers=4):
        super().__init__()
        s.ws, s.p, s.np = ws, patch, (ws // patch) ** 2
        s.proj = CLin(patch * patch * cin, dim)
        s.pos = nn.Embedding(s.np, dim)
        s.nr = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers * 2)])
        s.ni = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers * 2)])
        s.ar = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=0.1,
                                                    batch_first=True) for _ in range(layers)])
        s.ai = nn.ModuleList([nn.MultiheadAttention(dim, heads, dropout=0.1,
                                                    batch_first=True) for _ in range(layers)])
        s.m1 = nn.ModuleList([CLin(dim, dim * 2) for _ in range(layers)])
        s.m2 = nn.ModuleList([CLin(dim * 2, dim) for _ in range(layers)])
        s.layers = layers
        s.head = CLin(s.np * dim, ncl); s.do = nn.Dropout(0.3)

    def forward(s, xr, xi):
        B = xr.shape[0]
        f = lambda t: (t.unfold(2, s.p, s.p).unfold(3, s.p, s.p)
                        .permute(0, 2, 3, 1, 4, 5).reshape(B, s.np, -1))
        a, b = s.proj(f(xr), f(xi))
        a = a + s.pos(torch.arange(s.np, device=a.device))[None]
        for l in range(s.layers):
            na, nb = s.nr[2 * l](a), s.ni[2 * l](b)
            oa, _ = s.ar[l](na, na, na); ob, _ = s.ai[l](nb, nb, nb)
            a, b = a + oa, b + ob
            na, nb = s.nr[2 * l + 1](a), s.ni[2 * l + 1](b)
            ha, hb = crelu(*s.m1[l](na, nb)); ha, hb = s.m2[l](ha, hb)
            a, b = a + ha, b + hb
        a = s.do(a.flatten(1)); b = s.do(b.flatten(1))
        a, b = s.head(a, b)
        return torch.sqrt(a ** 2 + b ** 2 + 1e-9)


# --------------------------------------------------------------------------
# Complexity accounting.
# One complex multiply-accumulate is four real multiplies and two adds; our
# complex layers are built from two real kernels, so counting the real kernels
# and doubling gives the complex MAC count. We count Conv2d, Conv3d and Linear
# by hooking actual output shapes, which avoids the failure mode of the
# reference implementation's estimator, where 3-D convolutions matched no
# branch and contributed zero.
# --------------------------------------------------------------------------
def count_macs(model, xr, xi):
    """Real multiply-accumulates for one sample.

    Dispatch-based, not hook-based. Our equivariant layers call F.conv2d with
    explicit weight tensors rather than nn.Conv2d modules, and a module hook
    misses those entirely -- it reported 2.3e5 MACs for a network that actually
    performs 4.5e7, which would have flattered our own method in the complexity
    table. FlopCounterMode intercepts the operation itself, so functional and
    modular calls are counted alike.
    """
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    ctr = FlopCounterMode(display=False)
    with ctr, torch.no_grad():
        model(xr[:1], xi[:1])
    return ctr.get_total_flops() // 2       # counter reports 2 flops per MAC


REGISTRY = {
    "CV-MLP": CVMLP,
    "CV-2DCNN": CV2DCNN,
    "CV-3DCNN": CV3DCNN,
    "CV-2D-3D": CV2D3D,
    "CV-ViT": CVViT,
}
