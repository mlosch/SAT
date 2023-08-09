import os
import numpy as np
from torchvision.datasets import CIFAR100
import random


with open('./template.yaml', 'r') as f:
    template = f.read()
name_template = 'preactresnet18bn_mpO1_A{nA}.PGDs0.1i7e0.5_B{nB}.none_it{it:02d}'

f_run = open('run_all.sh', 'w')

AuB = list(range(10))
A = []

for idx in range(10):
    A = [idx]
    B = [i for i in range(len(AuB)) if i not in A]

    nA = len(A)
    nB = len(AuB) - nA

    weight = np.full((10,), 5./nB)
    for i in A:
        weight[i] = 5./nA
    cfg = template.format(A=str(A), B=str(B), weight=str(weight.tolist()))
    filename = name_template.format(nA=nA, nB=nB, it=idx)

    assert not os.path.exists(filename)
    with open(filename+'.yaml', 'w') as f:
        f.write(cfg)

    f_run.write('python -m tool.train --config config/cifar10/imbalanced_pgd7_A1/{}.yaml\n'.format(filename))

f_run.close()
