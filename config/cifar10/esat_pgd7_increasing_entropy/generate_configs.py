import os
import numpy as np

run_tool = 'python -m tool.train --config'

with open('./template.yaml', 'r') as f:
    template = f.read()
name_template = 'preactresnet18bn_mpO1_10bins_A{nA}.PGDs0.1i7e0.5_B{nB}.none'
ranking_filep = 'exp/cifar10/vanilla/preactresnet18bn_vanilla/FCdump/train/increasing_entropy_ranking.pth'


f_run = open('run_all.sh', 'w')
f_eval = open('eval_all.sh', 'w')

# label_to_idx = dict([(name, idx) for idx, name in enumerate(labels)])
N = 10
AuB = list(range(N))
As = AuB

for i in range(1,N):
    A = As[:i]
    A = sorted(A)
    B = [j for j in range(N) if j not in A]

    nA = len(A)
    nB = N - nA

    # weight = np.full((10,), 5./nB)
    # for i in A:
    #     weight[i] = 5./nA
    cfg = template.format(A=str(A), B=str(B), nbins=N, ranking_filep=ranking_filep) #weight=str(weight.tolist()))
    filename = name_template.format(nA=nA, nB=nB)

    assert not os.path.exists(filename)
    with open(filename+'.yaml', 'w') as f:
        f.write(cfg)

    f_run.write('python -m tool.train --config config/cifar10/esat_pgd7_decreasing_entropy/{}.yaml\n'.format(filename))

    f_eval.write("python -m tool.train --config config/cifar10/esat_pgd7_decreasing_entropy/{name}.yaml \
training.resume='exp/cifar10/esat_pgd7_decreasing_entropy/{name}/model/best_rob_acc.pth' training.evaluate_only=True \
evaluations.auto_attack_l2.eval_freq=1 evaluations.auto_attack_l2.eval_mode=True \
evaluations.auto_attack_l2.n_samples=9000 evaluations.auto_attack_l2.batch_size=1000 \
evaluations.auto_attack_l2.save_path='exp/cifar10/esat_pgd7_decreasing_entropy/{name}/evaluations/auto_attack_l2'\n".format(
            name=filename))

f_run.close()
f_eval.close()
