import os
import numpy as np
import torch

run_tool = 'python -m tool.train --config'

with open('./template.yaml', 'r') as f:
    template = f.read()
name_template = 'preactresnet18bn_mpO1_10bins_A{nA}.PGDs0.1i7e0.5_B{nB}.none'
ranking_filep = 'exp/cifar100/vanilla/preactresnet18bn_vanilla/FCdump/train/instancewise_decreasing_entropy_ranking.pth'
indices = torch.load('../../../'+ranking_filep).tolist()

f_run = open('run_all.sh', 'w')
f_eval = open('eval_all.sh', 'w')

# label_to_idx = dict([(name, idx) for idx, name in enumerate(labels)])
ninstances = 50000
nbins = 10
bin_width = ninstances // nbins
AuB = list(range(nbins))
As = AuB

for i in range(1,nbins):
    A = As[:i]
    A = sorted(A)
    B = [j for j in range(nbins) if j not in A]

    nA = len(A)
    nB = nbins - nA

    niA = nA*bin_width
    niB = nB*bin_width

    A_factor = 1.0 + i
    weight_value_A = A_factor * ((ninstances**2 - ninstances*niA) / (ninstances*niA))
    weight_value_B = 1.0

    print(nA, niA, weight_value_A, weight_value_B)

    cfg = template.format(A=str(A), B=str(B), nbins=nbins, ranking_filep=ranking_filep, loss_indices=indices[:niA], weight_value_A=weight_value_A, weight_value_B=weight_value_B, num_instances=ninstances) #weight=str(weight.tolist()))
    filename = name_template.format(nA=nA, nB=nB)

    assert not os.path.exists(filename)
    with open(filename+'.yaml', 'w') as f:
        f.write(cfg)

    if len(A) <= 5:
        f_run.write('python -m tool.train --config config/cifar100/weightedincreasing2to10_esat_pgd7_decreasing_entropy/{}.yaml\n'.format(filename))
    else:
        f_run.write('python -m tool.train --config config/cifar100/weightedincreasing2to10_esat_pgd7_decreasing_entropy/{}.yaml\n'.format(filename))

    f_eval.write("python -m tool.train --config config/cifar100/weightedincreasing2to10_esat_pgd7_decreasing_entropy/{name}.yaml \
training.resume='exp/cifar100/weightedincreasing2to10_esat_pgd7_decreasing_entropy/{name}/model/best_rob_acc.pth' training.evaluate_only=True \
evaluations.auto_attack_l2.eval_freq=1 evaluations.auto_attack_l2.eval_mode=True \
evaluations.auto_attack_l2.n_samples=9000 evaluations.auto_attack_l2.batch_size=1000 \
evaluations.auto_attack_l2.save_path='exp/cifar100/weightedincreasing2to10_esat_pgd7_decreasing_entropy/{name}/evaluations/auto_attack_l2'\n".format(
            name=filename))

f_run.close()
f_eval.close()
