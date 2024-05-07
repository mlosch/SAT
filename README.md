# On Adversarial Training without Perturbing all Examples (SAT)
Official repository for "On Adversarial Training without Perturbing all Examples", Accepted at ICLR 2024.
<BR>Paper [PDF](https://openreview.net/pdf?id=pE6gWrASQm), [Reviews](https://openreview.net/forum?id=pE6gWrASQm)
<BR>Poster: [poster.pdf](https://github.com/mlosch/SAT/blob/main/poster.pdf)

| ![Teaser figure](https://github.com/mlosch/SAT/blob/main/teaser_fig.png) | 
|:--:| 
| **In a Nutshell:** Vanilla adversarial training (AT) and most its variants perturb **every training example**. To what extent is that necessary? We split the training set into subsets *A* and *B*, train on *AuB* but construct adv. examples only for examples in *A*. |

## Requirements
- python 3.8
- pytorch 1.6.0
- autoattack
- tensorboard
- apex


## Reproducing Results
All our experiments are represented by yaml config files. They can be found in the directory 'config/'.
To train on multiple GPUs, make sure to update the value for the config key `train_gpu`, e.g. `train_gpu: [0,1,2,3]`.

### Training
To train, e.g. ESAT on ImageNet-200, run `bash config/imagenet200/weightedincreasing2to10_esat_pgd7_decreasing_entropy/run_all.sh`. This trains 10 models, according to the config files defined at the same location: `config/imagenet200/weightedincreasing2to10_esat_pgd7_decreasing_entropy/*.yaml`.

### Evaluation
After training completion, adversarial robustness is evaluated via AutoAttack. To start, call `bash config/imagenet200/weightedincreasing2to10_esat_pgd7_decreasing_entropy/eval_all.sh` 

### Additional Experiments
Results on Wide-ResNets (discussed [here](https://openreview.net/forum?id=aS2Yl8s5OG&noteId=H1XIxWAYSC)) training script can be found here: 'config/cifar10/wrn70-16_esat_pgd7_decreasing_entropy/'.