"""
Heavily borrowed from github.com/MadryLab/robustness
Specifically, attacker.py and attack_steps.py
Main changes:
- Joined functionality into one file
- Reduced constraints to {2,inf}
- Renamed constraint to norm_p to keep consistency with Lipschitz and PGDAttack code
"""

import os
from collections import OrderedDict

import torch.nn as nn
import torch
import numpy as np
# torch.autograd.set_detect_anomaly(True)

try:
    import apex.amp as amp
except:
    pass

from lib import util
from model import BaseModel, LipschitzModel
import model.losses as losses
from lib.scheduler import ScheduledModule

class AttackerStep(object):
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (torch.tensor): the original input
        '''
        super(AttackerStep, self).__init__()
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            torch.tensor x : the input to project back into the feasible set.

        Returns:
            A `torch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p`
        norms).

        Parameters:
            g (torch.tensor): the raw gradient

        Returns:
            The new input, a torch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x, eps, noise_type):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

    def to_image(self, x):
        '''
        Given an input (which may be in an alternative parameterization),
        convert it to a valid image (this is implemented as the identity
        function by default as most of the time we use the pixel
        parameterization, but for alternative parameterizations this functino
        must be overriden).
        '''
        return x

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        return torch.clamp(diff + self.orig_input, 0, 1)

    def step(self, x, g):
        """
        """
        step = torch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x, eps=None, noise_type='uniform'):
        """
        """
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, 0, 1)

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return torch.clamp(self.orig_input + diff, 0, 1)

    def step(self, x, g):
        """
        """
        l = len(x.shape) - 1
        g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x, eps=1.0, noise_type='normal'):
        """
        """
        if noise_type=='normal':
            delta = torch.empty_like(x).normal_()*eps
        elif noise_type=='uniform':
            delta = 2 * (torch.empty_like(x).uniform_()-0.5) * eps
        else:
            raise NotImplementedError

        d_flat = delta.view(delta.size(0),-1)
        n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r/n*self.eps
        return torch.clamp(x + delta, 0, 1)

        # l = len(x.shape) - 1
        # rp = torch.randn_like(x)
        # rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        # return torch.clamp(x + self.eps * rp / (rp_norm + 1e-10), 0, 1)

STEPS = {
    'inf': LinfStep,
    '2': L2Step,
}

def lossclass_from_string(name):
    if name.startswith('nn.'):
        loss_class = nn.__dict__[name.replace('nn.','')]
    else:
        loss_class = losses.__dict__[name]
    return loss_class

class AttackerModel(BaseModel, ScheduledModule):

    def __init__(self, p, eps, step_size, 
        iterations, loss_name, use_best=True,
        random_start=False, random_noise_eps=1.0, random_noise_type=None,
        data_mean=[0,0,0], data_std=[1,1,1]):
        super(AttackerModel, self).__init__()
        self.norm_p = p
        self.eps = eps
        self.step_size = step_size
        self.iterations = iterations
        # self.data_mean = data_mean
        # self.data_std = data_std
        self.adv_loss_name = loss_name
        self.additional_losses = dict()
        # self.loss = losses.construct_loss(loss_cfg)
        if p is not None:
            self.step_class = STEPS[str(p)]
        self.use_best = use_best
        self.random_start = random_start
        self.random_eps = random_noise_eps
        self.random_noise_type = random_noise_type

        self.data_mean = torch.Tensor(data_mean)
        self.data_std = torch.Tensor(data_std)
        self.optimizer = None  # automaticall set when using amp mixed precision

    def _normalize_input(self, x):
        if x.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(x.device).to(x.dtype)
            self.data_std = self.data_std.to(x.device).to(x.dtype)
        mean = self.data_mean[None, :, None, None]
        std = self.data_std[None, :, None, None]
        return (x - mean) / std

    def _calc_loss(self, x, target):
        outputs = self._forward_pass(x)
        if self.adv_loss_name not in self.loss:
            if self.adv_loss_name not in self.additional_losses:
                loss = lossclass_from_string(self.adv_loss_name)()
                self.additional_losses[self.adv_loss_name] = loss
            else:
                loss = self.additional_losses[self.adv_loss_name]
        else:
            loss = self.loss[self.adv_loss_name]
        loss_reduction = loss.reduction
        loss.reduction = 'none'
        losses = loss(outputs['prediction'], target)
        loss.reduction = loss_reduction
        return losses

    # A function that updates the best loss and best input
    def _replace_best(self, loss, bloss, x, bx, m=1):
        if bloss is None:
            bx = x.clone().detach()
            bloss = loss.clone().detach()
        else:
            replace = m * bloss < m * loss
            bx[replace] = x[replace].clone().detach()
            bloss[replace] = loss[replace]

        return bloss, bx

    def attack(self, x, target, **kwargs):
        orig_input = x.detach()

        # print('attack', x.shape, x.min().item(), x.max().item())

        step = self.step_class(orig_input=orig_input, eps=self.eps, step_size=self.step_size)
        best_loss = None
        best_x = None

        if self.random_start:
            if self.random_noise_type is None:
                if isinstance(step, L2Step):
                    self.random_noise_type = 'normal'
                elif isinstance(step, LinfStep):
                    self.random_noise_type = 'uniform'
                else:
                    raise NotImplementedError
            with torch.no_grad():
                # print('FGSM-RS eps={}, type={}'.format(self.random_eps, self.random_noise_type))
                x = step.random_perturb(x.clone(), self.random_eps, self.random_noise_type)

        for iteration in range(self.iterations):
            x = x.clone().detach().requires_grad_(True)

            losses = self._calc_loss(step.to_image(x), target)
            # assert losses.shape[0] == x.shape[0], \
            #         'Shape of losses must match input!'

            loss = torch.mean(losses)
            if self.optimizer is not None:
                with amp.scale_loss(loss, []) as scaled_loss:
                    # print('attack-loss', iteration, loss.item(), scaled_loss.item())
                    if torch.any(torch.isnan(loss)):
                        # print('attack-loss is nan. Skipping attack this iteration')
                        if best_x is None:
                            best_x = x
                        break
                    elif loss.item() > 10:
                        # print('attack-loss is unexpectedly large. Skipping attack this iteration')
                        if best_x is None:
                            best_x = x
                        break
                    scaled_loss.backward()
                grad = x.grad.detach()
                # x.grad.zero_()
            else:
                # print('attack-loss', iteration, loss.item())
                grad, = torch.autograd.grad(loss, [x])

            with torch.no_grad():
                args = [losses, best_loss, x, best_x]
                best_loss, best_x = self._replace_best(*args) if self.use_best else (losses, x)
                
                x = step.step(x, grad)
                x = step.project(x)

        # losses, out = self._calc_loss(step.to_image(x), target)
        # loss = torch.mean(losses)

        if self.use_best:
            return best_x
        else:
            return x
        
    def _forward_pass(self, x, y=None, **kwargs):
        if kwargs.get('normalize_input', True):
            x = self._normalize_input(x)
        return super(AttackerModel, self)._forward_pass(x, y=y, **kwargs)

    def forward(self, x, y=None, **kwargs):
        if y is not None and self.training and self.iterations > 0:
            is_training = self.training
            self.eval()
            kwargs['x_original'] = x.detach().clone()
            x = self.attack(x, y, **kwargs)
            if is_training:
                self.train()

        return super(AttackerModel, self).forward(x,y,**kwargs)

    def __repr__(self):
        return 'AttackerModel(p={}, eps={}, step={}, it={}, use_best={}, rnd_start={})({})'.format(self.norm_p, self.eps, self.step_size, self.iterations, self.use_best, self.random_start, BaseModel.__repr__(self))


class DomainNoiseGeneratingModel(BaseModel, ScheduledModule):
    def __init__(self, *args, 
        attacked_domain1=[], attacked_domain2=[], 
        sigmas_domain1=[.08, .12, 0.18, 0.26, 0.38],
        sigmas_domain2=[],
        data_mean=[0,0,0], data_std=[1,1,1], **kwargs):
        super(DomainNoiseGeneratingModel, self).__init__()
        assert len(attacked_domain1) > 0
        self.sigmas_domain1 = sigmas_domain1
        self.sigmas_domain2 = sigmas_domain2

        self.data_mean = torch.Tensor(data_mean)
        self.data_std = torch.Tensor(data_std)

        self.attacked_domain1 = attacked_domain1
        self.attacked_domain2 = attacked_domain2

    def _normalize_input(self, x):
        if x.device != self.data_mean.device:
            self.data_mean = self.data_mean.to(x.device)
            self.data_std = self.data_std.to(x.device)
        mean = self.data_mean[None, :, None, None]
        std = self.data_std[None, :, None, None]
        return (x - mean) / std

    def perturb(self, x, sigmas):
        with torch.no_grad():
            coinflip = (torch.rand(x.shape[0], device=x.device) > 0.5).float()
            sigma = torch.from_numpy(np.random.choice(sigmas, (x.shape[0],))).float().to(x.device)
            noise = torch.randn_like(x) * sigma[:,None, None, None]
            img = torch.clamp(x + noise*coinflip[:,None, None, None], 0, 1)
        return img

        
    def _forward_pass(self, x, y=None, **kwargs):
        if kwargs.get('normalize_input', True):
            x = self._normalize_input(x)
        return super(DomainNoiseGeneratingModel, self)._forward_pass(x, y=y, **kwargs)

    def forward(self, x, y=None, **kwargs):
        if y is not None and self.training:
            is_training = self.training
            self.eval()
            kwargs['x_original'] = x.detach().clone()

            # perturb first domain
            attacked_domain1 = util.domain_mask(self.attacked_domain1, y)
            x[attacked_domain1] = self.perturb(x[attacked_domain1], self.sigmas_domain1)

            if len(self.attacked_domain2) > 0:
                # perturb second domain
                attacked_domain2 = util.domain_mask(self.attacked_domain2, y)
                x[attacked_domain2] = self.perturb(x[attacked_domain2], self.sigmas_domain2)

            if is_training:
                self.train()

        return super(DomainNoiseGeneratingModel, self).forward(x,y,**kwargs) 


class DomainAttackerModel(AttackerModel):
    def __init__(self, *args, 
        attacked_domain1=[], attacked_domain2=[], 
        p_domain1=None, p_domain2=None, 
        eps_domain1=None, eps_domain2=None, 
        step_size_domain1=None, step_size_domain2=None,
        it_domain1=None, it_domain2=None, 
        random_targeted_attacks=False, 
        flip_likelihood_based_logit_drop=False,
        ignore_flips_for_classes=None,
        debug_save_path=None, **kwargs):
        super(DomainAttackerModel, self).__init__(*args, p=None, eps=None, step_size=None, iterations=None, **kwargs)
        assert len(attacked_domain1) > 0
        self.register_buffer('attacked_domain1', torch.Tensor(attacked_domain1))
        self.register_buffer('attacked_domain2', torch.Tensor(attacked_domain2))
        self.p_domain1 = p_domain1
        self.p_domain2 = p_domain2
        self.eps_domain1 = eps_domain1
        self.eps_domain2 = eps_domain2
        self.step_size_domain1 = step_size_domain1
        self.step_size_domain2 = step_size_domain2
        self.it_domain1 = it_domain1
        self.it_domain2 = it_domain2
        self.random_targeted_attacks = random_targeted_attacks
        self.flip_likelihood_based_logit_drop = flip_likelihood_based_logit_drop
        self.ignore_flips_for_classes = ignore_flips_for_classes
        self.debug_save_path = debug_save_path
        self.debug_data = dict()

        self.flip_stats = None
        self.debug_flip_count = np.zeros((10,))
        self.debug_iter = 0

        # if self.ignore_flips_for_classes is not None:
        #     for key in self.ignore_flips_for_classes.keys():
        #         values = self.ignore_flips_for_classes[key]
        #         values = torch.Tensor(values)

    def dump(self, tensor, fileprefix):
        if self.training:
            filep = os.path.join(self.debug_save_path, 'train', '{}.bin'.format(fileprefix))
        else:
            filep = os.path.join(self.debug_save_path, 'val', '{}.bin'.format(fileprefix))
        if not os.path.exists(os.path.dirname(filep)):
            os.makedirs(os.path.dirname(filep))

        util.write_bin_data(filep, tensor)

    def _calc_loss(self, x, target):
        outputs = self._forward_pass(x)

        if self.debug_save_path is not None:
            self.debug_data['target'] = target.detach()
            self.debug_data['output'] = outputs['prediction'].detach()
            if not 'initial_output' in self.debug_data:
                self.debug_data['initial_output'] = outputs['prediction'].detach()

        if self.adv_loss_name not in self.loss:
            if self.adv_loss_name not in self.additional_losses:
                loss = lossclass_from_string(self.adv_loss_name)()
                self.additional_losses[self.adv_loss_name] = loss
            else:
                loss = self.additional_losses[self.adv_loss_name]
        else:
            loss = self.loss[self.adv_loss_name]
        loss_reduction = loss.reduction
        loss.reduction = 'none'
        if self.random_targeted_attacks:
            # we maximize the random target loss
            assert outputs['prediction'].shape[-1] == 10, 'Model setup for 10 classes. For different settings, update implementation (see nlogits)'
            # losses = -loss(outputs['prediction'], self.rnd_target)
            losses = -loss(outputs['prediction'], self.rnd_target)# + loss(outputs['prediction'], target)

        elif self.flip_likelihood_based_logit_drop:
            if self.flip_stats is None:
                nclasses = outputs['prediction'].shape[-1]
                self.flip_stats = torch.eye(nclasses)

            if torch.rand(1)[0] > 0.5:
                # drop
                raise NotImplementedError
            else:
                # default: we minimize the target loss
                losses = loss(outputs['prediction'], target)

                # aggregate flip stats

        else:
            # we minimize the target loss
            losses = loss(outputs['prediction'], target)
        loss.reduction = loss_reduction

        if self.ignore_flips_for_classes is not None:
            preds = torch.argmax(outputs['prediction'], dim=1)
            N = 0
            for classidx, ignore_flips in self.ignore_flips_for_classes.items():
                for flip_idx in ignore_flips:
                    mask = (target == classidx) & (preds == flip_idx)
                    N += torch.sum(mask).item()
                    losses[mask] = 0
            # print('disabled entries: {} (n dog: {})'.format(N, torch.sum(preds==5).item()))

                # mask = x.new(*losses.shape).fill_(0)
                # mask.scatter_(index=torch.Tensor(ignore_flips, device=x.device, dtype=target.dtype), dim=0, value=1)
                # mask &= (target==classidx)
                # losses[mask]


        return losses


    def attack(self, x, y, **kwargs):
        if self.random_targeted_attacks:
            nlogits = 10 #outputs['prediction'].shape[1]
            # rnd_target = torch.randint(0, nlogits, y.shape, dtype=y.dtype, device=y.device)
            # # rnd_target = torch.full(y.shape, 4, dtype=y.dtype, device=y.device)
            # mask = rnd_target == y
            # while mask.sum() > 0:
            #     # ensure all rnd_targets are unequal the target class
            #     rnd_target[mask] = torch.randint(0, nlogits, rnd_target[mask].shape, dtype=y.dtype, device=y.device)
            #     mask = rnd_target == y

            if self.debug_flip_count.sum() == 0:
                probs = np.ones((nlogits,))
                probs[y.cpu().numpy()] = 0
                probs /= probs.sum()
            else:
                probs = self.debug_flip_count
                probs = probs/probs.sum()
                probs = 1-probs
                probs[y.cpu().numpy()] = 0
                probs /= probs.sum()
                # print(probs, probs.sum())
            rnd_target = np.random.choice(nlogits, size=y.shape, p=probs)
            # print(probs)
            # print(rnd_target[:10])
            self.rnd_target = torch.from_numpy(rnd_target).to(y.device).long()

        return super(DomainAttackerModel, self).attack(x,y,**kwargs)


    def attack_inputs_(self, x, y, **kwargs):
        # attack first domain
        # set environment variables
        self.norm_p = self.p_domain1
        self.eps = self.eps_domain1
        self.step_size = self.step_size_domain1
        self.iterations = self.it_domain1
        self.step_class = STEPS[str(self.norm_p)]
        attacked_domain1 = util.domain_mask(self.attacked_domain1, y)
        if attacked_domain1.sum() == 0:
            # no instances to attack
            return attacked_domain1
            # if is_training:
            #     self.train()
            # return super(AttackerModel, self).forward(x,y,**kwargs)

        x[attacked_domain1] = self.attack(x[attacked_domain1], y[attacked_domain1], **kwargs).detach().clone()
        x_mask = attacked_domain1

        if self.debug_save_path is not None:
            pred = torch.argmax(self.debug_data['output'], dim=1)
            target = self.debug_data['target']
            initial_pred = torch.argmax(self.debug_data['initial_output'])
            debug_flips = [torch.sum((initial_pred == target) & (pred == classidx)).item() for classidx in range(10)]
            self.debug_flip_count += np.array(debug_flips)
            self.debug_iter += 1

            if self.debug_iter % 100 == 0:
                print('Flipped predictions:\n\t{}'.format(self.debug_flip_count)) 

            self.dump(self.debug_data['target'].cpu(), 'target')
            self.dump(self.debug_data['output'].cpu(), 'output')
            self.dump(self.debug_data['initial_output'].cpu(), 'initial_output')
            with torch.no_grad():
                x_adv = x[attacked_domain1]
                x_ori = kwargs['x_original'][attacked_domain1]
                x_norms = torch.norm((x_adv-x_ori).view(x_adv.shape[0], -1), p=self.p_domain1, dim=1)
            self.dump(x_norms.cpu(), 'x_norms')
            del self.debug_data['initial_output']

        if len(self.attacked_domain2) > 0:
            # attack second domain
            # set environment variables
            self.norm_p = self.p_domain2
            self.eps = self.eps_domain2
            self.step_size = self.step_size_domain2
            self.iterations = self.it_domain2
            self.step_class = STEPS[str(self.norm_p)]
            attacked_domain2 = util.domain_mask(self.attacked_domain2, y)

            x[attacked_domain2] = self.attack(x[attacked_domain2], y[attacked_domain2], **kwargs)
            x_mask = attacked_domain1 | attacked_domain2

        return x_mask

    def forward(self, x, y=None, **kwargs):
        if y is not None and self.training:
            is_training = self.training
            self.eval()
            kwargs['x_original'] = x.detach().clone()

            self.attack_inputs_(x, y, **kwargs)

            if is_training:
                self.train()

        result = super(AttackerModel, self).forward(x,y,**kwargs)
        # if self.random_targeted_attacks:
        #     print('Predictions')
        #     # print(result['prediction'])
        #     attacked_domain1 = util.domain_mask(self.attacked_domain1, y)
        #     print(torch.argmax(result['prediction'][attacked_domain1], dim=1).cpu().numpy())
        return result


class InstanceDomainAttackerModel(DomainAttackerModel):
    def __init__(self, *args, 
        ranking=None,
        bins=5,
        attacked_domain1_bins=[], attacked_domain2_bins=[], 
        random_selection=False, random_select_fraction=0.5,
        onthefly_selection=False, onthefly_select_fraction=0.5, **kwargs):

        if ranking is not None:
            if not os.path.exists(ranking):
                raise RuntimeError('{} does not exist'.format(ranking))
            ranking = torch.load(ranking)
            N = len(ranking)
            if N%bins != 0:
                print('InstanceDomainAttackerModel :: Warning :: bins={} does not divide domain ranking list of length {}'.format(bins, N))
            bin_width = N//bins
            attacked_domain1 = []
            attacked_domain2 = []
            for i in range(0,N,bin_width):
                bin_idx = i//bin_width
                if bin_idx in attacked_domain1_bins:
                    attacked_domain1 += ranking[i:i+bin_width].tolist()
                elif bin_idx in attacked_domain2_bins:
                    attacked_domain2 += ranking[i:i+bin_width].tolist()
        else:
            assert len(attacked_domain1_bins) > 0
            attacked_domain1 = attacked_domain1_bins
            attacked_domain2 = attacked_domain2_bins
        self.random_selection = random_selection
        self.random_select_fraction = random_select_fraction

        self.onthefly_selection = onthefly_selection
        self.onthefly_select_fraction = onthefly_select_fraction
        assert self.random_selection==False or self.onthefly_selection==False

        super(InstanceDomainAttackerModel, self).__init__(*args,  attacked_domain1=attacked_domain1, attacked_domain2=attacked_domain2, **kwargs)

        if self.onthefly_selection:
            N = 50_000
            self.register_buffer('sum_instance_entropy', torch.zeros(N))
            self.register_buffer('n_instance_entropy', torch.zeros(N))

    def forward(self, x, y=None, **kwargs):
        if y is not None and self.training:
            if 'imids' not in kwargs:
                raise RuntimeError('imids attribute required for InstanceDomainAttackerModel')
            
            is_training = self.training
            self.eval()
            kwargs['x_original'] = x.detach().clone()

            # attack first domain
            # set environment variables
            self.norm_p = self.p_domain1
            self.eps = self.eps_domain1
            self.step_size = self.step_size_domain1
            self.iterations = self.it_domain1
            self.step_class = STEPS[str(self.norm_p)]

            attacked_domain_inds = self.attacked_domain1
            if self.random_selection:
                rand_inds = torch.randperm(len(kwargs['imids']))
                rand_inds = rand_inds[:int(len(rand_inds)*self.random_select_fraction)]
                # select fraction of image indices in current batch
                attacked_domain_inds = kwargs['imids'][rand_inds]

            attacked_domain1 = util.domain_mask(attacked_domain_inds, kwargs['imids'])

            # if self.random_selection:
            #     n_sel = int(len(kwargs['imids'])*self.random_select_fraction)
            #     assert attacked_domain1.sum() == n_sel, 'should be selected n={}, but selected were {}'.format(n_sel, attacked_domain1.sum())

            if attacked_domain1.sum() == 0:
                # no instances to attack
                if is_training:
                    self.train()
                return super(AttackerModel, self).forward(x,y,**kwargs)

            x[attacked_domain1] = self.attack(x[attacked_domain1], y[attacked_domain1], **kwargs)

            if self.debug_save_path is not None:
                self.dump(self.debug_data['target'].cpu(), 'target')
                self.dump(self.debug_data['output'].cpu(), 'output')

            if len(self.attacked_domain2) > 0:
                # attack second domain
                # set environment variables
                self.norm_p = self.p_domain2
                self.eps = self.eps_domain2
                self.step_size = self.step_size_domain2
                self.iterations = self.it_domain2
                self.step_class = STEPS[str(self.norm_p)]
                attacked_domain2 = util.domain_mask(self.attacked_domain2, kwargs['imids'])

                x[attacked_domain2] = self.attack(x[attacked_domain2], y[attacked_domain2], **kwargs)

            if is_training:
                self.train()

        output = super(AttackerModel, self).forward(x,y,**kwargs)

        if self.onthefly_selection and self.training:
            # accumulate entropy stats
            log_probs = F.log_softmax(output['prediction'], dim=1)
            entropy = -torch.sum(log_probs*(log_probs.exp()), dim=1, keepdim=False)
            imids = kwargs['imids']
            self.sum_instance_entropy.scatter_add_(dim=0, index=imids, src=entropy)
            self.n_instance_entropy.scatter_add_(dim=0, index=imids, src=entropy.new(len(imids)).fill_(1))

            if torch.all(self.n_instance_entropy>0) and torch.rand(1)[0] >= 0.99:
                print('updating indices')
                n_sel = int(len(self.sum_instance_entropy)*self.onthefly_select_fraction)
                # rank by avg entropy
                avg_entropy = self.sum_instance_entropy / self.n_instance_entropy
                ranked_inds = torch.argsort(avg_entropy, descending=True)[:n_sel]
                self.attacked_domain1 = ranked_inds

        return output


from model.adversarial_model_functional import *

class GradientMaskingWrapper(nn.Module):
    def __init__(self, module):
        super(GradientMaskingWrapper, self).__init__()
        self.module = module
        self.masked_modules = []
        self.recursive_linear_replacement(self.module)
        self.recursive_conv_replacement(self.module)

        if isinstance(self.module, nn.Linear):
            self.module = self._from_linear(module)
            self.masked_modules.append(self.module)
        elif isinstance(self.module, nn.Conv2d):
            self.module = self._from_conv2d(module)
            self.masked_modules.append(self.module)

    @property
    def mask_update(self):
        return self.__mask_update
    
    @mask_update.setter
    def mask_update(self, mask):
        self.__mask_update = mask

    def _from_linear(self, m):
        new_m = GradientMaskingLinear(m.in_features, m.out_features)
        with torch.no_grad():
            new_m.weight = m.weight
            new_m.bias = m.bias
        return new_m

    def _from_conv2d(self, m):
        new_m = GradientMaskingConv2d(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, m.bias)
        with torch.no_grad():
            new_m.weight = m.weight
            new_m.bias = m.bias
        return new_m

    def recursive_linear_replacement(self, module):
        for name, m in module.named_children():
            if isinstance(m, nn.Linear):
                new_m = self._from_linear(m)
                setattr(module, name, new_m)
                self.masked_modules.append(new_m)
            else:
                self.recursive_linear_replacement(m)

    def recursive_conv_replacement(self, module):
        for name, m in module.named_children():
            if isinstance(m, nn.Conv2d):
                new_m = self._from_conv2d(m)
                setattr(module, name, new_m)
                self.masked_modules.append(new_m)
            else:
                self.recursive_conv_replacement(m)

    def _set_masks(self):
        for m in self.masked_modules: #self.module.modules():
            #if isinstance(m, GradientMaskingLinear) or isinstance(m, GradientMaskingConv2d):
            m.mask_update = self.__mask_update

    def forward(self, x):
        if self.training:
            self._set_masks()
        o = self.module(x)
        return o

    def __repr__(self):
        return 'MW{'+str(self.module)+'}'


class GradientMaskedDomainAttackerModel(DomainAttackerModel):
    def __init__(self, *args, masked_layers=[], **kwargs):
        super(GradientMaskedDomainAttackerModel, self).__init__(*args, **kwargs)
        self.masked_layers = dict()
        self.unmasked_layers = dict()
        for name in masked_layers:
            self.masked_layers[name] = None

    def __setattr__(self, name, value):
        if isinstance(value, nn.Module):
            if name in self.masked_layers:
                value = GradientMaskingWrapper(value)
                self.masked_layers[name] = value
            else:
                if not name in ['loss', 'metrics', 'evaluations', 'metric_argspecs', 'loss_argspecs']:
                    value = GradientMaskingWrapper(value)
                    self.unmasked_layers[name] = value
        
        super(GradientMaskedDomainAttackerModel, self).__setattr__(name, value)

    def forward(self, x, y=None, **kwargs):
        if y is not None and self.training:
            is_training = self.training
            self.eval()
            kwargs['x_original'] = x.detach().clone()

            mask = self.attack_inputs_(x, y, **kwargs)

            # masked layers still require signal for the non-attacked samples
            #  we append these samples to the end of the batch
            N = x.shape[0]
            x = torch.cat([x, kwargs['x_original'][mask]], dim=0)
            y = torch.cat([y, y[mask]], dim=0)
            Next = x.shape[0]

            # pass mask to all wrapper modules
            #  all wrapper modules pass masks onto individual modules which carry weights
            masked_update = torch.cat([~mask, mask.new(Next-N).fill_(True)])
            for name, module in self.masked_layers.items():
                module.mask_update = None #masked_update

            unmasked_update = mask.new(Next).fill_(True)
            unmasked_update[Next-N:] = False #torch.cat([mask, mask.new(Next-N).fill_(False)])
            for name, module in self.unmasked_layers.items():
                module.mask_update = None #unmasked_update


            if is_training:
                self.train()

        return super(AttackerModel, self).forward(x,y,**kwargs)  


class AttackerLipschitzModel(AttackerModel, LipschitzModel):
    def __init__(self, attacker_kwargs, lipmodel_kwargs):
        AttackerModel.__init__(self, **attacker_kwargs)
        LipschitzModel.__init__(self, **lipmodel_kwargs)

class DomainAttackerLipschitzModel(DomainAttackerModel, LipschitzModel):
    def __init__(self, attacker_kwargs, lipmodel_kwargs):
        DomainAttackerModel.__init__(self, **attacker_kwargs)
        LipschitzModel.__init__(self, **lipmodel_kwargs)
        # re-register buffers from attack model

        self.register_buffer('attacked_domain1', torch.Tensor(attacker_kwargs['attacked_domain1']))
        self.register_buffer('attacked_domain2', torch.Tensor(attacker_kwargs.get('attacked_domain2', [])))
