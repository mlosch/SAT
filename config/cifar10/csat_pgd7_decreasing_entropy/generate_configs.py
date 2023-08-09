import os
import numpy as np

run_tool_half = 'python -m tool.train --config'
run_tool_full = 'python -m tool.train --config'

with open('./template.yaml', 'r') as f:
	template = f.read()
name_template = 'preactresnet18bn_mpO1_A{nA}.PGDs0.1i7e0.5_B{nB}.none'


f_run = open('run_all.sh', 'w')
f_eval = open('eval_all.sh', 'w')

labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

As = [3, 5, 2, 4, 0, 6, 7, 8, 9, 1]  # decreasing class entropy

for i in range(1,10):
	A = As[:i]
	A = sorted(A)
	B = [j for j in range(len(labels)) if j not in A]

	nA = len(A)
	nB = len(labels) - nA

	cfg = template.format(A=str(A), B=str(B))
	filename = name_template.format(nA=nA, nB=nB)

	assert not os.path.exists(filename)
	with open(filename+'.yaml', 'w') as f:
		f.write(cfg)

	if len(A) <= 5:
		f_run.write('{} config/cifar10/csat_pgd7_decreasing_entropy/{}.yaml\n'.format(run_tool_half, filename))
	else:
		f_run.write('{} config/cifar10/csat_pgd7_decreasing_entropy/{}.yaml\n'.format(run_tool_full, filename))

	f_eval.write("python -m tool.train --config config/cifar10/csat_pgd7_decreasing_entropy/{name}.yaml \
training.resume='exp/cifar10/csat_pgd7_decreasing_entropy/{name}/model/best_rob_acc.pth' training.evaluate_only=True \
evaluations.auto_attack_l2.eval_freq=1 evaluations.auto_attack_l2.eval_mode=True \
evaluations.auto_attack_l2.n_samples=9000 evaluations.auto_attack_l2.batch_size=1000 \
evaluations.auto_attack_l2.save_path='exp/cifar10/csat_pgd7_decreasing_entropy/{name}/evaluations/auto_attack_l2'\n".format(
            name=filename))

f_run.close()
