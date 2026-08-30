import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8,
                     "axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":160,"savefig.dpi":200,"savefig.bbox":"tight"})
C={"blue":"#2a78d6","orange":"#eb6834","aqua":"#1baf7a","red":"#e34948",
   "ink2":"#52514e","grid":"#dcdcd8"}
th=np.array([0,10,22.5,30,45])
S=[("Temel model (artirimsiz)",[97.64,89.09,53.62,43.88,38.09],C["red"],"o"),
   ("Temel + donme artirimi",  [94.68,94.53,94.43,94.40,93.98],C["orange"],"s"),
   ("Esdegisken, ayrik N=8",   [97.10,93.68,97.10,95.70,97.10],C["blue"],"^"),
   ("Esdegisken, Fourier",     [96.11,95.14,96.11,95.33,96.11],C["aqua"],"D")]
fig,ax=plt.subplots(1,2,figsize=(9.6,3.5),gridspec_kw={"width_ratios":[1,1],"wspace":0.28})
for a,(ylo,yhi,ttl) in zip(ax,[(30,100,"Tam olcek: temel model cokuyor"),
                               (92,98.5,"Yakinlastirilmis: dayanikli yontemler")]):
    for nm,v,c,m in S:
        if a is ax[1] and nm.startswith("Temel model"): continue
        a.plot(th,v,color=c,marker=m,ms=6,lw=2,label=nm,zorder=3,
               markeredgecolor="white",markeredgewidth=0.9)
    for g in (0,22.5,45): a.axvline(g,color=C["grid"],lw=1,ls=":",zorder=1)
    a.set_ylim(ylo,yhi); a.grid(axis="y",color=C["grid"],lw=0.6,zorder=0)
    a.set_xlabel("polarimetrik donme acisi θ (derece)")
    a.set_title(ttl,fontsize=9.5,loc="left",color=C["ink2"])
    a.set_xticks([0,10,22.5,30,45]); a.set_xticklabels(["0","10","22.5","30","45"])
ax[0].set_ylabel("genel dogruluk (%)")
ax[0].annotate("−59.6 puan",xy=(45,38.09),xytext=(28,52),fontsize=10,color=C["red"],
               arrowprops=dict(arrowstyle="->",color=C["red"],lw=1.3))
ax[1].annotate("izgara noktalarinda\nBIREBIR ayni",xy=(45,97.10),xytext=(24,93.2),fontsize=8,
               color=C["blue"],ha="center",arrowprops=dict(arrowstyle="->",color=C["blue"],lw=1.1))
ax[1].annotate("izgara DISI kayip",xy=(10,93.68),xytext=(6,92.4),fontsize=8,color=C["ink2"])
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=4,frameon=False,fontsize=8.5,bbox_to_anchor=(0.5,-0.10))
fig.suptitle("Polarimetrik donme altinda dayaniklilik — Flevoland, %1 etiket",
             fontsize=10.5,x=0.02,ha="left",y=1.03)
plt.savefig("figs/F2_rotation.png"); plt.close()
print("F2 yenilendi")
