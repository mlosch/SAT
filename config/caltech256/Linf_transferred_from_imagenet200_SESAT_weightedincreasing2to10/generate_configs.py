import os
import numpy as np

with open('./template.yaml', 'r') as f:
    template = f.read()
name_template = 'resnet50_onlyCE_fromATi{order_abbrv}A{nA}.B{nB}'


f_run = open('run_all.sh', 'w')
f_eval = open('eval_all.sh', 'w')


for order, order_abbrv in zip(['decreasing', 'increasing'], ['dec', 'inc']):
    for nA in range(1,10,1):
        nB = 10 - nA

        cfg = template.format(order=order, nA=nA, nB=nB)
        filename = name_template.format(order_abbrv=order_abbrv, nA=nA, nB=nB)

        assert not os.path.exists(filename)
        with open(filename+'.yaml', 'w') as f:
            f.write(cfg)

        f_run.write('python -m tool.train --config config/caltech256/Linf_transferred_from_imagenet200_SESAT_weightedincreasing2to10a/{}.yaml\n'.format(filename))

        f_eval.write("python -m tool.train --config config/caltech256/Linf_transferred_from_imagenet200_SESAT_weightedincreasing2to10a/{name}.yaml \
            training.resume='exp/caltech256/Linf_transferred_from_imagenet200_SESAT_weightedincreasing2to10a/{name}/model/best_rob_acc.pth' training.evaluate_only=True \
            evaluations.auto_attack_linf.eval_freq=1 evaluations.auto_attack_linf.eval_mode=True \
            evaluations.auto_attack_linf.n_samples=5122 evaluations.auto_attack_linf.batch_size=100 \
            evaluations.auto_attack_linf.save_path='exp/caltech256/Linf_transferred_from_imagenet200_SESAT_weightedincreasing2to10a/{name}/evaluations/auto_attack_linf' \
            training.train_gpu=[0]\n".format(
                        name=filename))

f_run.close()
f_eval.close()
