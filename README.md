
# Tabular AIL

This repository is the official implementation of **Provably Efficient Adversarial Imitation Learning with Unknown Transitions** which is accepted as oral presentation in UAI 2023. The code contains the implementation of BC, FEM, GTAL, OAL and MBTAIL.

##  Install

- Python 3.10

```
conda create -n tabular_ail python=3.10
conda activate tabular_ail
```


- Python Packages


```
pip install numpy
pip install gym==0.15.6
pip install termcolor==2.3.0
pip install json-tricks==3.15.5
pip install Pyyaml
```


## Run

- run mbtail

```
bash scripts/run_mbtail.sh
```

- run oal

```
bash scripts/run_oal.sh
```

