"""Cok sahneli GPU veri hatti (esdegisir normalizasyon destekli)."""
import numpy as np, torch
from polsar_data import load_scene, bdist, SCENES
from polsar_lib import rot6_torch
from eqnorm import EqNorm

class MSPipe:
    def __init__(s, scene, dev="cuda", W=15, norm="equivariant", eval_cap=200_000, seed=0):
        s.scene=scene; s.dev=dev; s.W=W; s.M=W//2
        X, s.gt, s.ncl = load_scene(scene)
        s.R, s.C = s.gt.shape
        s.dist = bdist(s.gt)
        s.norm = norm
        s.eqn = EqNorm(X).to(dev) if norm=="equivariant" else None
        if norm in ("channel", "channel7"):
            mu=np.array([X[...,k].mean() for k in range(6)],np.complex64)
            sd=np.array([X[...,k].std()  for k in range(6)],np.float32)
            s.MUr=torch.tensor(mu.real,device=dev).view(1,6,1,1)
            s.MUi=torch.tensor(mu.imag,device=dev).view(1,6,1,1)
            s.SD =torch.tensor(sd,device=dev).view(1,6,1,1)
        if norm=="channel7":
            # Per-channel standardisation PLUS the invariant log-power channel.
            # The equivariant normalisation carries seven channels, plain
            # per-channel standardisation six, so comparing them directly would
            # confound the broken equivariance with the missing channel. This
            # variant matches the channel count so the ablation isolates the
            # property under test.
            T=X[...,0].real+X[...,1].real+X[...,2].real
            L=np.log(np.clip(T,1e-20,None))
            m=float(np.median(L)); v=float(np.median(np.abs(L-m)))*1.4826+1e-12
            s.LM=torch.tensor(m,device=dev,dtype=torch.float32)
            s.LS=torch.tensor(v,device=dev,dtype=torch.float32)
        Xp=np.pad(X,((s.M,s.M),(s.M,s.M),(0,0)),mode="constant")
        s.Pr=torch.from_numpy(np.ascontiguousarray(Xp.real)).to(dev)
        s.Pi=torch.from_numpy(np.ascontiguousarray(Xp.imag)).to(dev)
        s.off=torch.arange(W,device=dev)
        s.lr,s.lc = np.nonzero(s.gt>0); s.y = s.gt[s.lr,s.lc]-1
        # buyuk sahnelerde degerlendirme alt-orneklemi (istatistiksel olarak fazlasiyla yeterli)
        s.eval_cap=eval_cap; s._rng=np.random.default_rng(seed)
        del X, Xp
    def grab(s,r,c,th=None):
        rr=r[:,None,None]+s.off[None,:,None]; cc=c[:,None,None]+s.off[None,None,:]
        xr=s.Pr[rr,cc].permute(0,3,1,2).contiguous(); xi=s.Pi[rr,cc].permute(0,3,1,2).contiguous()
        if th is not None: xr,xi=rot6_torch(xr,xi,th)
        if s.eqn is not None: return s.eqn(xr,xi)
        a=(xr-s.MUr)/s.SD; b=(xi-s.MUi)/s.SD
        if s.norm=="channel7":
            span=torch.clamp(xr[:,0]+xr[:,1]+xr[:,2], min=1e-20)
            L=torch.clamp((torch.log(span)-s.LM)/s.LS, -8.0, 8.0).unsqueeze(1)
            a=torch.cat([a,L],1); b=torch.cat([b,torch.zeros_like(L)],1)
        return a,b
    def cap_eval(s, idx):
        """Subsample the evaluation indices, DETERMINISTICALLY.

        This used to draw from a shared, stateful generator, so the subset
        depended on how many models had already been evaluated: every model was
        scored on a different 150k subset, both across tables and between rows
        of the same table. That is a methodological error -- competing methods
        must be scored on the same test set -- and it is why two runs of the
        same configuration disagreed by up to 0.6 pp.

        A fresh generator with a fixed seed makes the subset a property of the
        scene and the cap alone, so every model in every run sees exactly the
        same evaluation pixels.
        """
        if len(idx) <= s.eval_cap: return idx
        g = np.random.default_rng(20260827)
        return np.sort(g.choice(idx, s.eval_cap, replace=False))
    def info(s):
        return (f"{s.scene}: {s.R}x{s.C}, {s.ncl} sinif, {len(s.y)} etiketli, "
                f"sinir<=7 %{100*(s.dist[s.lr,s.lc]<=7).mean():.1f}")
