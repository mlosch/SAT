import os
import numpy as np
from torchvision.datasets import CIFAR100
import random

with open('./template.yaml', 'r') as f:
    template = f.read()
name_template = 'preactresnet18bn_unevensets_A{nA}.PGDs0.1i7e0.5_B{nB}.none'

# avg on training:
ranking = [55, 72, 44, 4, 65, 50, 27, 3, 45, 74, 80, 67, 35, 11, 73, 98, 2, 46, 38, 93, 10, 32, 78, 64, 18, 33, 40, 25, 26, 77, 96, 19, 59, 51, 42, 99, 63, 83, 30, 79, 14, 84, 7, 92, 12, 29, 90, 31, 6, 15, 34, 16, 95, 37, 97, 71, 62, 47, 81, 91, 87, 70, 66, 5, 61, 52, 57, 21, 23, 54, 22, 89, 13, 36, 86, 69, 43, 88, 1, 85, 49, 24, 75, 39, 76, 9, 28, 60, 53, 41, 17, 48, 0, 56, 8, 94, 58, 82, 20, 68]

f_run = open('run_all.sh', 'w')
f_eval = open('eval_all.sh', 'w')

N = 100
AuB = list(range(N))
A = []

min_dt = 6
max_dt = 24

for idx in range(10,N,10):
    A = [ranking[i] for i in range(idx)]
    B = [i for i in range(len(AuB)) if i not in A]

    nA = len(A)
    nB = len(AuB) - nA

    A_factor = 1.0 + idx//10
    weight = np.full((N,), 1.0)
    for i in A:
        weight[i] = A_factor * (N*N - N*nA) / (N*nA)
    print(nA, weight[A[0]])

    cfg = template.format(A=str(A), B=str(B), weight=str(weight.tolist()))
    filename = name_template.format(nA=nA, nB=nB)

    assert not os.path.exists(filename)
    with open(filename+'.yaml', 'w') as f:
        f.write(cfg)

    if nA <= 50:
       f_run.write('python -m tool.train --config config/cifar100/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{}.yaml\n'.format(filename))
    else:
        f_run.write('python -m tool.train --config config/cifar100/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{}.yaml\n'.format(filename))

    f_eval.write("python -m tool.train --config config/cifar100/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{name}.yaml \
training.resume='exp/cifar100/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{name}/model/best_rob_acc.pth' training.evaluate_only=True \
evaluations.auto_attack_l2.eval_freq=1 evaluations.auto_attack_l2.eval_mode=True \
evaluations.auto_attack_l2.n_samples=9000 evaluations.auto_attack_l2.batch_size=1000 \
evaluations.auto_attack_l2.save_path='exp/cifar100/weightedincreasing2to10_csat_pgd7_decreasing_entropy/{name}/evaluations/auto_attack_l2'\n".format(
            name=filename))

f_run.close()
f_eval.close()
