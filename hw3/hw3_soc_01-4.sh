#!/bin/bash

#SBATCH -J "hw3_013"              # isin adi

#SBATCH -A orttdf                         # account / proje adi
#SBATCH -p a100q                          # kuyruk (partition/queue) adi

#SBATCH -n 64                             # cekirdek / islemci sayisi
#SBATCH -N 1                             # bilgisayar sayisi

source ~/miniconda3/bin/activate
conda init --all
conda activate cmpe591_env

#calisacak gpu isi
python homework3_soc_uhem_01.py &
python homework3_soc_uhem_02.py &
python homework3_soc_uhem_03.py &
python homework3_soc_uhem_04.py &
wait

