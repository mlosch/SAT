import os
import numpy as np

with open('./template.yaml', 'r') as f:
    template = f.read()
name_template = 'preactresnet18bn_mpO1_onlyCE_fromATi{order_abbrv}A{nA}.B{nB}'


f_run = open('run_all.sh', 'w')
f_eval = open('eval_all.sh', 'w')

labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# label_to_idx = dict([(name, idx) for idx, name in enumerate(labels)])
# AuB = list(range(10))


for order, order_abbrv in zip(['decreasing', 'increasing'], ['dec', 'inc']):
    for nA in range(10,100,10):
        nB = 100 - nA

        cfg = template.format(order=order, nA=nA, nB=nB)
        filename = name_template.format(order_abbrv=order_abbrv, nA=nA, nB=nB)

        assert not os.path.exists(filename)
        with open(filename+'.yaml', 'w') as f:
            f.write(cfg)

        f_run.write('python -m tool.train --config config/cifar10/transferred_from_cifar100_SESAT_weightedincreasing2to10/{}.yaml\n'.format(filename))

        f_eval.write("python -m tool.train --config config/cifar10/transferred_from_cifar100_SESAT_weightedincreasing2to10/{name}.yaml \
            training.resume='exp/cifar10/transferred_from_cifar100_SESAT_weightedincreasing2to10/{name}/model/best_rob_acc.pth' training.evaluate_only=True \
            evaluations.auto_attack_l2.eval_freq=1 evaluations.auto_attack_l2.eval_mode=True \
            evaluations.auto_attack_l2.n_samples=9000 evaluations.auto_attack_l2.batch_size=1000 \
            evaluations.auto_attack_l2.save_path='exp/cifar10/transferred_from_cifar100_SESAT_weightedincreasing2to10/{name}/evaluations/auto_attack_l2'\n".format(
                        name=filename))

f_run.close()
f_eval.close()
