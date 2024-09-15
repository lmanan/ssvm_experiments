# ssvm experiments

Fitting _motile_ weights in the _Fluo-N2DL-HeLa_ dataset using SSVM.


#### Download HeLa dataset

Download and uncompress the data from the v0.0.1 [release](https://github.com/lmanan/ssvm_experiments/releases/download/v0.0.1/Fluo-N2DL-HeLa.zip), and put at the same location as `main.py`.

#### Installation


```bash
conda create -n motile_ssvm python==3.10 
conda activate motile_ssvm
conda install -c conda-forge -c funkelab -c gurobi ilpy
pip install -e git+https://github.com/lmanan/motile.git
pip install -e git+https://github.com/lmanan/motile_toolbox.git
pip install -e git+https://github.com/lmanan/traccuracy.git
pip install natsort imagecodecs jsonargparse
conda install scip==9.0.0
```


