#!/bin/sh
# TEK kuyruk. Ikinci bir kuyruk ASLA baslatilmayacak; calisan betik duzenlenmeyecek.
# Hepsi devam ettirilebilir: kesilirse ayni komut kaldigi yerden surer.
PY="C:/Users/musa.peker/AppData/Local/anaconda3/envs/last-env/python.exe"
cd "c:/Users/musa.peker/Desktop/CV-MsAtViT-main/research" || exit 1
echo "== exp31: ana tablo, 10 tohum, Flevoland =="
"$PY" -u exp31_seed10.py     > exp31.log 2>&1
echo "== exp34: AIR-PolSAR-Seg-2.0, 3 bolge =="
"$PY" -u exp34_air.py        > exp34.log 2>&1
echo "== exp30: kmax=8 =="
"$PY" -u exp30_kmax.py       >> exp30.log 2>&1
echo "== exp32: yama taramasi, 10 tohum =="
"$PY" -u exp32_patch10.py    > exp32.log 2>&1
echo "HEPSI BITTI"
