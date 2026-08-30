"""Steerable network with the Clebsch-Gordan products carried to k_max = 6 or 8.

The network of steerable.py truncates at k_max = 4, which is where the input
lives. A reviewer is entitled to ask what that costs, and whether the gap
between the steerable and the discrete network is a consequence of it. This
module reproduces SteerNet with the higher weight classes carried explicitly:

  k_max = 6 adds a weight-6 field, fed by z_2 z_4  (2 + 4)
  k_max = 8 adds a weight-8 field as well, fed by z_4 z_4 and z_2 z_6

The added fields start empty -- the coherency matrix has no k > 4 content -- so
they are populated only from products, which is precisely the expressiveness the
truncation gives up. Equivariance is unaffected either way: it holds weight class
by weight class, and every operation below stays inside one.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from steerable import cmul, cmulc, CConv, decompose, SteerLift, pool


class SteerLayerK(nn.Module):
    """One block, with weight classes {0, 2, 4}, {0, 2, 4, 6} or {0, ..., 8}.

    Every product that lands inside the retained set is formed; products that
    would leave it are dropped, which is what k_max means.
    """

    def __init__(s, c0, c, kmax=4, k=3):
        super().__init__()
        s.kmax = kmax
        s.ks = [2, 4] + ([6] if kmax >= 6 else []) + ([8] if kmax >= 8 else [])
        # invariant stream sees |z_k|^2 for every retained harmonic field
        s.k0 = nn.Conv2d(c0 + len(s.ks) * c, c0, k, padding=k // 2)
        # each harmonic field gets its own field plus the products landing on it
        s.conv = nn.ModuleDict()
        s.nin = {}
        for kk in s.ks:
            n = 1 + len(s._sources(kk))
            s.nin[kk] = n
            s.conv[str(kk)] = CConv(n * c, c, k)
        s.gate = nn.ModuleDict({str(kk): nn.Conv2d(c0, c, 1) for kk in s.ks})

    def _sources(s, target):
        """Products (a, b, conj) of weight `target` whose inputs are retained.

        conj=False is a*b and carries weight a+b; conj=True is a*conj(b) and
        carries a-b. Sums are taken once, with a <= b, since a*b = b*a.
        """
        out = []
        for a in s.ks:
            for b in s.ks:
                if a <= b and a + b == target:
                    out.append((a, b, False))
                if a - b == target:
                    out.append((a, b, True))
        return out

    def forward(s, w0, Z):
        nrm = [Z[k][0] ** 2 + Z[k][1] ** 2 for k in s.ks]
        h0 = F.relu(s.k0(torch.cat([w0] + nrm, 1)))
        out = {}
        for kk in s.ks:
            parts_r, parts_i = [Z[kk][0]], [Z[kk][1]]
            for a, b, conj in s._sources(kk):
                f = cmulc if conj else cmul
                pr, pi = f(Z[a][0], Z[a][1], Z[b][0], Z[b][1])
                parts_r.append(pr); parts_i.append(pi)
            hr, hi = s.conv[str(kk)](torch.cat(parts_r, 1), torch.cat(parts_i, 1))
            g = torch.sigmoid(s.gate[str(kk)](h0))
            out[kk] = (hr * g, hi * g)
        return h0, out


def invariants_k(w0, Z, ks):
    """Magnitudes of every field, plus the relative phase between weights 2
    and 4 that the k_max = 4 network already uses."""
    feats = [w0]
    for k in ks:
        feats.append(torch.sqrt(Z[k][0] ** 2 + Z[k][1] ** 2 + 1e-12))
    z2r, z2i = Z[2]; z4r, z4i = Z[4]
    sqr, sqi = cmul(z2r, z2i, z2r, z2i)
    xr, xi = cmulc(z4r, z4i, sqr, sqi)
    d = torch.sqrt(sqr ** 2 + sqi ** 2 + 1e-12) * torch.sqrt(
        z4r ** 2 + z4i ** 2 + 1e-12) + 1e-12
    return torch.cat(feats + [xr / d, xi / d], 1)


class SteerNetK(nn.Module):
    def __init__(s, ncl, c=24, c0=16, kmax=4, stats=None):
        super().__init__()
        s.kmax = kmax
        s.ks = [2, 4] + ([6] if kmax >= 6 else []) + ([8] if kmax >= 8 else [])
        s.lift = SteerLift(c0, c, *(stats if stats is not None
                                    else (None, None, 1.0, 1.0)))
        # the higher fields have no input content, so they are lifted from zero
        s.l1 = SteerLayerK(c0, c, kmax); s.l2 = SteerLayerK(c0, c, kmax)
        s.l3 = SteerLayerK(c0, c, kmax)
        # readout: w0, one magnitude per harmonic field, and the real and
        # imaginary parts of the weight-2/weight-4 relative-phase invariant
        nin = (c0 + len(s.ks) * c + 2 * c) * 3 * 3
        s.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                               nn.Linear(nin, 128), nn.ReLU(),
                               nn.Linear(128, ncl))

    def forward(s, xr, xi):
        w0, z2, z4 = decompose(xr, xi)
        w0, z2, z4 = s.lift(w0, z2, z4)
        Z = {2: z2, 4: z4}
        for k in s.ks[2:]:
            Z[k] = (torch.zeros_like(z4[0]), torch.zeros_like(z4[1]))
        w0, Z = s.l1(w0, Z)
        w0, Z = s.l2(w0, Z)
        w0 = F.avg_pool2d(w0, 2)
        Z = {k: (F.avg_pool2d(v[0], 2), F.avg_pool2d(v[1], 2)) for k, v in Z.items()}
        w0, Z = s.l3(w0, Z)
        w0 = F.avg_pool2d(w0, 2)
        Z = {k: (F.avg_pool2d(v[0], 2), F.avg_pool2d(v[1], 2)) for k, v in Z.items()}
        return s.head(invariants_k(w0, Z, s.ks))
