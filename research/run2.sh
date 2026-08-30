#!/bin/sh
PY="C:/Users/musa.peker/AppData/Local/anaconda3/envs/last-env/python.exe"
cd "c:/Users/musa.peker/Desktop/CV-MsAtViT-main/research" || exit 1
echo "== exp38: roll-invariant temel, 6 sahne =="
"$PY" -u exp38_rollinv.py        > exp38.log 2>&1
echo "== exp37: yonlendirilebilir ag ablasyonu =="
"$PY" -u exp37_steer_ablation.py > exp37.log 2>&1
echo "HEPSI BITTI"
