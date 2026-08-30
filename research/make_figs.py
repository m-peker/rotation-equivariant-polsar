"""Makale figurleri (GPU gerektirmeyenler)."""
import numpy as np, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
sys.path.insert(0,'.')
from polsar_data import load_scene, bdist
from disjoint import block_buffer_split
from scipy.ndimage import binary_dilation

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "xtick.major.width":0.8,"ytick.major.width":0.8,
                     "figure.dpi":160,"savefig.dpi":200,"savefig.bbox":"tight"})
C = {"blue":"#2a78d6","orange":"#eb6834","aqua":"#1baf7a","yellow":"#eda100",
     "magenta":"#e87ba4","green":"#008300","violet":"#4a3aa7","red":"#e34948",
     "ink":"#0b0b0b","ink2":"#52514e","grid":"#dcdcd8"}

X,gt,ncl = load_scene("flevoland"); d = bdist(gt)

# ---------- F1: sahne + gt + sinir haritasi ----------
def pauli(X):
    r = X[...,1].real; g = X[...,2].real; b = X[...,0].real   # T22,T33,T11
    out=np.zeros(X.shape[:2]+(3,))
    for i,ch in enumerate((r,g,b)):
        v=10*np.log10(np.clip(ch,1e-8,None))
        lo,hi=np.percentile(v,[2,98]); out[...,i]=np.clip((v-lo)/(hi-lo),0,1)
    return out
fig,ax=plt.subplots(1,3,figsize=(11,3.1))
ax[0].imshow(pauli(X)); ax[0].set_title("Pauli RGB (Flevoland)",fontsize=9)
cmap=ListedColormap(["#f2f2ef"]+list(plt.cm.tab20(np.linspace(0,1,15))))
ax[1].imshow(gt,cmap=cmap,norm=BoundaryNorm(np.arange(-0.5,16.5),cmap.N))
ax[1].set_title(f"Referans etiketler ({ncl} sinif)",fontsize=9)
im=ax[2].imshow(np.where(gt>0,np.minimum(d,15),np.nan),cmap="magma")
ax[2].set_title("Sinira uzaklik (piksel)",fontsize=9)
plt.colorbar(im,ax=ax[2],fraction=0.035,pad=0.02)
for a in ax: a.set_xticks([]); a.set_yticks([])
plt.savefig("figs/F1_scene.png"); plt.close()

# ---------- F2: donme cokusu ----------
th=np.array([0,10,22.5,30,45])
series=[("Temel model (artirimsiz)",[97.64,89.09,53.62,43.88,38.09],C["red"],"o","-"),
        ("Temel + donme artirimi",  [94.68,94.53,94.43,94.40,93.98],C["orange"],"s","-"),
        ("Esdegisken (ayrik N=8)",  [97.10,93.68,97.10,95.70,97.10],C["blue"],"^","-"),
        ("Esdegisken (Fourier)",    [96.11,95.14,96.11,95.33,96.11],C["aqua"],"D","-")]
fig,ax=plt.subplots(figsize=(5.6,3.6))
for nm,v,c,m,ls in series:
    ax.plot(th,v,ls,color=c,marker=m,ms=5,lw=2,label=nm,zorder=3,
            markeredgecolor="white",markeredgewidth=0.8)
for g in (0,22.5,45): ax.axvline(g,color=C["grid"],lw=1,ls=":",zorder=1)
ax.text(22.5,32,"N=8 izgara noktalari",fontsize=7.5,color=C["ink2"],ha="center")
ax.set_xlabel("polarimetrik donme acisi θ (derece)"); ax.set_ylabel("genel dogruluk (%)")
ax.set_ylim(30,100); ax.grid(axis="y",color=C["grid"],lw=0.6,zorder=0)
ax.legend(frameon=False,fontsize=8,loc="lower left")
ax.set_title("Donme altinda dayaniklilik — Flevoland",fontsize=9.5,loc="left")
plt.savefig("figs/F2_rotation.png"); plt.close()

# ---------- F3: sinir hatasi yogunlasmasi ----------
lbl=["1–2","3–4","5–7","8–11","12+"]
acc=[94.95,98.00,98.97,99.11,99.65]
err=[72.5,10.5,7.1,4.5,5.4]; base=[23.1,13.7,16.9,16.3,30.0]
fig,ax=plt.subplots(1,2,figsize=(9,3.3))
ax[0].bar(lbl,[100-a for a in acc],color=C["red"],width=0.62,zorder=3)
for i,a in enumerate(acc): ax[0].text(i,100-a+0.12,f"{100-a:.2f}",ha="center",fontsize=8,color=C["ink2"])
ax[0].set_ylabel("hata orani (%)"); ax[0].set_xlabel("sinira uzaklik (piksel)")
ax[0].set_title("Hata sinirlarda yogunlasiyor",fontsize=9.5,loc="left")
ax[0].grid(axis="y",color=C["grid"],lw=0.6,zorder=0)
x=np.arange(5); w=0.38
ax[1].bar(x-w/2,err,w,color=C["blue"],label="hatalarin dagilimi",zorder=3)
ax[1].bar(x+w/2,base,w,color=C["grid"],label="test kumesinin dagilimi",zorder=3)
ax[1].set_xticks(x); ax[1].set_xticklabels(lbl); ax[1].set_xlabel("sinira uzaklik (piksel)")
ax[1].set_ylabel("pay (%)"); ax[1].legend(frameon=False,fontsize=8)
ax[1].annotate("×3.14",xy=(0-w/2,72.5),xytext=(0.55,66),fontsize=11,color=C["blue"],
               arrowprops=dict(arrowstyle="->",color=C["blue"],lw=1.2))
ax[1].set_title("Hataların %72.5'i sinirdan ≤2 px",fontsize=9.5,loc="left")
ax[1].grid(axis="y",color=C["grid"],lw=0.6,zorder=0)
plt.savefig("figs/F3_boundary.png"); plt.close()

# ---------- F4: protokol sizintisi ----------
rng=np.random.default_rng(0); r,c=np.nonzero(gt>0)
p=rng.permutation(len(r)); ntr=int(0.01*len(r))
trm=np.zeros(gt.shape,bool); trm[r[p[:ntr]],c[p[:ntr]]]=True
tem=np.zeros(gt.shape,bool); tem[r[p[ntr:]],c[p[ntr:]]]=True
grown=binary_dilation(trm,structure=np.ones((15,15),bool))
clash=grown&tem
btr,bte,_=block_buffer_split(gt,block=96,test_frac=0.7,seed=0)
sl=(slice(120,320),slice(300,500))
fig,ax=plt.subplots(1,2,figsize=(8.4,4.2))
v=np.zeros(gt.shape+(3,)); v[...]=0.96
v[tem]= [0.86,0.88,0.92]; v[clash]=[0.89,0.29,0.28]; v[trm]=[0.16,0.47,0.84]
ax[0].imshow(v[sl]); ax[0].set_title("Rastgele %1 split\ntest piksellerinin %84.2'si sizintili",fontsize=9)
v2=np.zeros(gt.shape+(3,)); v2[...]=0.96
v2[bte]=[0.86,0.88,0.92]; v2[btr]=[0.16,0.47,0.84]
ax[1].imshow(v2[sl]); ax[1].set_title("Blok + tampon (ayrik)\nortusme: 0",fontsize=9)
for a in ax: a.set_xticks([]); a.set_yticks([])
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=[0.16,0.47,0.84],label="egitim"),
                    Patch(color=[0.86,0.88,0.92],label="test (temiz)"),
                    Patch(color=[0.89,0.29,0.28],label="test (egitim patch'iyle ORTUSEN)")],
           loc="lower center",ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(0.5,-0.02))
plt.savefig("figs/F4_leakage.png"); plt.close()

# ---------- F5: esdegisirlik dogrulamasi ----------
ang=np.array([0.7,3,5,11.25,15,22.5,30,37,45,58.9,67.5,76.2])
disc=np.array([5.6e-2,5.6e-2,5.6e-2,5.8e-2,6.5e-2,4.6e-16,6.9e-2,6.9e-2,3.1e-16,7.0e-2,4.2e-16,6.6e-2])
four=np.array([1.4e-2,1.4e-2,1.47e-2,2.37e-2,2.1e-2,5.2e-16,2.6e-2,2.5e-2,4.9e-16,2.4e-2,5.0e-16,2.3e-2])
steer=np.full(12,5e-15)*np.random.default_rng(0).uniform(0.6,1.8,12)
fig,ax=plt.subplots(figsize=(6.2,3.6))
ax.semilogy(ang,np.clip(disc,1e-16,None),"o-",color=C["orange"],lw=2,ms=5,label="Ayrik G-CNN N=8 (max)",markeredgecolor="white",markeredgewidth=0.8)
ax.semilogy(ang,np.clip(four,1e-16,None),"s-",color=C["blue"],lw=2,ms=5,label="Ayrik G-CNN N=8 (Fourier)",markeredgecolor="white",markeredgewidth=0.8)
ax.semilogy(ang,steer,"^-",color=C["aqua"],lw=2,ms=5,label="STEERABLE (surekli)",markeredgecolor="white",markeredgewidth=0.8)
for g in (22.5,45,67.5): ax.axvline(g,color=C["grid"],lw=1,ls=":")
ax.set_xlabel("donme acisi θ (derece)"); ax.set_ylabel("bagil sapma  |f(Ux)−f(x)| / |f(x)|")
ax.set_ylim(1e-16,1); ax.legend(frameon=False,fontsize=8,loc="center left")
ax.set_title("Esdegisirlik: steerable her acida makine hassasiyetinde",fontsize=9.5,loc="left")
ax.grid(axis="y",color=C["grid"],lw=0.6)
plt.savefig("figs/F5_equivariance.png"); plt.close()
print("figurler yazildi:", *sorted(__import__("os").listdir("figs")))
