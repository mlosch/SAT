import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.scheduler as scheduler
from model.lipschitz_model import LipschitzLayerComputer, LipschitzLayerContainer, ClassifierLipschitzComputer
import model.lipschitz_model as lip

def construct_loss(loss_cfg, default_reduction='sum'):
    if isinstance(loss_cfg, dict):
        loss_type = loss_cfg.pop('type')
    else:
        loss_type = loss_cfg
        loss_cfg = {}

    if loss_type.startswith('nn.'):
        loss_class = nn.__dict__[loss_type.replace('nn.','')]
    else:
        # print('__name__', __name__, type(__name__))
        current_module = __import__(__name__)
        # print('current_module', current_module, type(current_module))
        # print(current_module.__dict__.keys())
        loss_class = current_module.losses.__dict__[loss_type]
    if 'reduction' not in loss_cfg:
        loss_cfg['reduction'] = default_reduction
    return loss_class(**loss_cfg)


class CosineSimilarityLoss(nn.Module):
    def __init__(self, lambda_=1.0, temperature=1.0, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.temp = temperature
        self.reduction = reduction
        self.lambda_ = lambda_

    def forward(self, prediction, target):
        normed_preds = prediction / prediction.norm(dim=1, keepdim=True)

        cos_sim = (1./self.temp) * normed_preds @ normed_preds.t()

        targets = (target[None,...]==target[...,None]).long()
        
        # targets = targets*2 - 1
        # loss = - (cos_sim * targets)

        # loss = (-cos_sim*targets).mean() + (cos_sim*(1.0-targets)).mean()
        loss = -cos_sim.mean()
        return self.lambda_*loss
        # if self.reduction == 'mean':
        #     return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # elif self.reduction == 'none':
        #     return loss.view(-1)
        # else:
        #     raise RuntimeError('Unknown reductino')


class CutOffCrossEntropyLoss(nn.Module):
    def __init__(self, prob_cutoff, reduction='mean', weight=None):
        super(CutOffCrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.log_cutoff = -np.log(prob_cutoff)
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', weight)

    def forward(self, prediction, target):
        loss = self.loss(prediction, target)
        loss[loss < self.log_cutoff] = 0

        if self.reduction == 'mean':
            if self.weight is not None:
                weights = self.weight.gather(0, target)
                weights[loss < self.log_cutoff] = 0
                return loss.sum() / (weights.sum() + 1.e-8)
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss.squeeze()
        else:
            raise AttributeError('Unkown reduction {}'.format(self.reduction))



class EntropyLoss(scheduler.ScheduledModule):
    def __init__(self, beta=1.0, reduction='mean'):
        super(EntropyLoss, self).__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, prediction, target=None):
        log_p = F.log_softmax(prediction, dim=1)
        p = F.softmax(prediction, dim=1)
        entropy = -torch.sum(log_p*p, dim=1)

        if self.reduction == 'mean':
            return -entropy.mean() * self.beta
        elif self.reduction == 'sum':
            return -entropy.sum() * self.beta
        elif self.reduction == 'none':
            return -entropy.squeeze() * self.beta
        else:
            raise AttributeError('Unkown reduction {}'.format(self.reduction))


class EntropyMaximizingCrossEntropyLoss(nn.Module):
    def __init__(self, beta=1.0, reduction='mean', weight=None, entropy_weighting=True):
        super(EntropyMaximizingCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.entropy_weighting = entropy_weighting
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = weight
        self.beta = beta

    def forward(self, prediction, target):
        log_p = F.log_softmax(prediction, dim=1)
        p = F.softmax(prediction, dim=1)
        if self.weight is not None and self.entropy_weighting:
            entropy = -torch.sum(log_p*p, dim=1)
            ent_weights = self.weight.gather(0, target)
            entropy = (entropy * ent_weights).sum() / self.weight.gather(0, target).sum()
        else:
            entropy = -torch.sum(log_p*p, dim=1)
            entropy = torch.mean(entropy)
        xentropy = F.nll_loss(log_p, target, reduction='mean', weight=self.weight)

        loss = xentropy - self.beta * entropy
        return loss
        # if self.reduction == 'mean':
        #     if self.weight is not None:
        #         return loss.sum() / self.weight.gather(0, target).sum()
        #     else:
        #         return loss.mean()
        # elif self.reduction == 'sum':
        #     return loss.sum()
        # elif self.reduction == 'none':
        #     return loss.squeeze()
        # else:
        #     raise AttributeError('Unkown reduction {}'.format(self.reduction))


class InstanceWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None, weighted_instances=None, weight_value_indices=None, weight_value_remainder=None, num_instances=None, reduction='mean'):
        super(InstanceWeightedCrossEntropyLoss, self).__init__()
        if weighted_instances is not None:
            assert weights is None, 'weights cannot be set when weighted_instances is set'
            assert weight_value_indices is not None
            assert weight_value_remainder is not None
            assert num_instances is not None, 'num_instances must be defined if weighted_instances is set'
            weights = torch.full((num_instances,), weight_value_remainder)

            if isinstance(weighted_instances, str):
                print('InstanceWeightedCrossEntropyLoss: Loading indices from {}'.format(weighted_instances))
                indices = torch.from_numpy(torch.load(weighted_instances)).long()
            else:
                indices = weighted_instances.long()
            weights.scatter_(dim=0, index=indices, value=weight_value_indices)

        self.register_buffer('weight', weights)
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, prediction, target, imids):
        loss = self.loss(prediction, target)

        if self.training:
            # weight loss
            weights = torch.gather(self.weight, index=imids, dim=0)
            wloss = loss * weights
            # print(weights)
            # print(loss)
            # print(wloss)
            # print(loss.shape, wloss.shape, weights.shape)
        else:
            weights = prediction.new(1).fill_(float(prediction.shape[0]))
            wloss = loss

        if self.reduction == 'mean':
            return wloss.sum() / weights.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss.squeeze()
        else:
            raise AttributeError('Unkown reduction {}'.format(self.reduction))


class EntropyMaximizingInstanceWeightedCrossEntropyLoss(InstanceWeightedCrossEntropyLoss):
    def __init__(self, beta=1.5, entropy_weighting=True, **kwargs):
        super(EntropyMaximizingInstanceWeightedCrossEntropyLoss, self).__init__(**kwargs)
        self.beta = beta
        self.entropy_weighting = entropy_weighting

    def __entropy(self, prediction):
        log_p = F.log_softmax(prediction, dim=1)
        p = F.softmax(prediction, dim=1)
        entropy = -torch.sum(log_p*p, dim=1)

        return entropy

    def forward(self, prediction, target, imids):
        loss = self.loss(prediction, target)

        if self.training:
            weights = torch.gather(self.weight, index=imids, dim=0)
            wloss = loss * weights
        else:
            weights = prediction.new(1).fill_(float(prediction.shape[0]))
            wloss = loss

        wentropy = self.__entropy(prediction) * weights

        if self.reduction == 'mean':
            return (wloss.sum() + self.beta*wentropy.sum()) / weights.sum()
        elif self.reduction == 'sum':
            return loss.sum() + self.beta*wentropy.sum()
        elif self.reduction == 'none':
            return loss.squeeze() + self.beta*wentropy.squeeze()
        else:
            raise AttributeError('Unkown reduction {}'.format(self.reduction))

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0, reduction='mean', weight=None):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.reduction = reduction
        if weight is not None:
            weight = weight / weight.sum()
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, prediction, target):
        pred = F.log_softmax(prediction, dim=1)
        if self.weight is not None:
            pred = pred*self.weight.unsqueeze(0)

        with torch.no_grad():
            N = pred.shape[-1]
            true_dist = torch.full_like(pred, self.smoothing / (N-1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0-self.smoothing)

        loss = -torch.sum(true_dist*pred, -1)

        if self.reduction == 'mean':
            if self.weight is not None:
                return loss.sum() / self.weight.gather(0, target).sum()
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss.squeeze()
        else:
            raise AttributeError('Unkown reduction {}'.format(self.reduction))


class LinearProbesLoss(nn.Module):
    def __init__(self, num_classes, layers, loss_cfg, input_shape, model, reduction='mean', fix_index_errors=False):
        super(LinearProbesLoss, self).__init__()

        probes = dict()

        with torch.no_grad():
            model.eval()
            x = torch.rand(*input_shape).unsqueeze(0) #.cuda()
            for name, module in model.named_children():
                x = module(x)
                if name in layers:
                    in_feats = np.prod(x.shape[1:])
                    probes[name] = nn.Linear(in_feats, num_classes)

        self.probes = nn.ModuleDict(probes)
        self.loss = construct_loss(loss_cfg, default_reduction=reduction)
        self.last_outputs = dict()
        self.fix_index_errors = fix_index_errors

    def forward(self, target, layer_output):
        loss_per_probe = dict()
        for name, layer in layer_output.items():
            if name not in self.probes:
                continue
            x = layer
            x = x.view(x.shape[0], -1)
            # print(x.shape, self.probes[name].weight.shape)
            pred = self.probes[name](x)
            self.last_outputs[name] = pred.detach()
            if self.fix_index_errors:
                target = target.clamp(max=pred.shape[1]-1)
            loss_per_probe[name] = self.loss(pred, target)

        loss = 0
        for name, probeloss in loss_per_probe.items():
            loss = probeloss + loss

        return loss


class GloroLoss(scheduler.ScheduledModule):
    def __init__(self, output_module, epsilon, num_iter, lipschitz_computer, reduction='mean', detach_lipschitz_computer=False):
        super(GloroLoss, self).__init__()
        # raise DeprecatedError('Use layers.AddGloroAbsendLogit instead.')
        self.W = lambda : output_module.parent.weight
        self.fc_lip_estimate = lambda : output_module.lip_estimate
        self.lc = lipschitz_computer
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.reduction = reduction
        self.detach_lipschitz_computer = detach_lipschitz_computer

    def forward(self, prediction, target):
        eps = self.epsilon
        if self.detach_lipschitz_computer:
            with torch.no_grad():
                K = self.lc(num_iter=self.num_iter)
                K = K / self.fc_lip_estimate()
        else:
            K = self.lc(num_iter=self.num_iter)
            K = K / self.fc_lip_estimate()

        return gloro_loss(prediction, target, self.W(), K, epsilon=eps, reduction=self.reduction)

    def __repr__(self):
        if isinstance(self.epsilon, scheduler.Scheduler):
            return 'GloroLoss(eps={}, num_iter={})'.format(self.epsilon, self.num_iter)
        else:
            return 'GloroLoss(eps={:.2f}, num_iter={})'.format(self.epsilon, self.num_iter)


class MarginRatioLoss(scheduler.ScheduledModule):
    def __init__(self, lipschitz_computer, num_iter, output_module, norm_p, margin_lambda, detach_lipschitz_computer=True, reduction='mean'):
        super(MarginRatioLoss, self).__init__()
        self.lc = lipschitz_computer
        self.num_iter = num_iter
        self.detach_lipschitz_computer = detach_lipschitz_computer
        if not detach_lipschitz_computer:
            assert num_iter > 0
        self.norm_p = norm_p
        self.margin_lambda = margin_lambda
        assert isinstance(output_module, ClassifierLipschitzComputer) or isinstance(output_module, lip.ClassifierLipschitzConstraint)
        self.fc_lip_estimate = lambda : output_module.lip_estimate
        self.W = lambda : output_module.parent.weight

    @property
    def reduction(self):
        return self.loss.reduction

    @reduction.setter
    def reduction(self, value):
        self.loss.reduction = value

    def forward(self, prediction, target):
        W = self.W()
        Kmin = 1.0
        if not self.training:
            num_iter = 0
        else:
            num_iter = self.num_iter
        if self.detach_lipschitz_computer:
            with torch.no_grad():
                K = self.lc(num_iter=num_iter) / self.fc_lip_estimate()
                kW = torch.norm(W, p=self.norm_p, dim=1)
                Ki = kW * K
        else:
            K = self.lc(num_iter=num_iter) / self.fc_lip_estimate()
            kW = torch.norm(W, p=self.norm_p, dim=1)
            Ki = kW * K

        # Ki \in 10
        # prediction \in Nx10
        # target \in N
        # target_vals \in Nx1

        # prediction = F.log_softmax(prediction, dim=1)

        # target_vals = torch.gather(prediction, index=target.unsqueeze(1), dim=1)
        # margin = (prediction - target_vals)**2
        # margin[margin == 0] = 1.0
        # inv_margin_ratio = Ki.unsqueeze(0) / margin

        y_j, j = torch.topk(prediction, k=2, dim=1)
        already_correct = (j[:,0] == target)
        j[already_correct,0] = j[already_correct,1]
        i = target.unsqueeze(1)
        j = j[:,0].unsqueeze(1) # take the (second) best


        margin = 0.5*(torch.gather(prediction, index=i, dim=1) - torch.gather(prediction, index=j, dim=1))**2


        return self.margin_lambda * -margin.mean() #inv_margin_ratio.mean()


class LipschitzCrossEntropyLoss(scheduler.ScheduledModule):
    def __init__(self, lipschitz_computer, num_iter, output_module, norm_p, K_scale=1.0, alpha=1.0, detach_lipschitz_computer=True, use_Kmax=False, grad_scale=False, reduction='mean'):
        super(LipschitzCrossEntropyLoss, self).__init__()
        self.lc = lipschitz_computer
        self.num_iter = num_iter
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        self.detach_lipschitz_computer = detach_lipschitz_computer
        if not detach_lipschitz_computer:
            assert num_iter > 0
        self.K_scale = K_scale
        self.use_Kmax = use_Kmax
        self.grad_scale = grad_scale
        self.alpha = alpha
        self.norm_p = norm_p
        assert isinstance(output_module, ClassifierLipschitzComputer) or isinstance(output_module, lip.ClassifierLipschitzConstraint)
        self.fc_lip_estimate = lambda : output_module.lip_estimate
        self.W = lambda : output_module.parent.weight

    @property
    def reduction(self):
        return self.loss.reduction

    @reduction.setter
    def reduction(self, value):
        self.loss.reduction = value
    

    def forward(self, prediction, target):
        W = self.W()
        Kmin = 1.0
        if not self.training:
            num_iter = 0
        else:
            num_iter = self.num_iter
        if self.detach_lipschitz_computer:
            with torch.no_grad():
                K = self.lc(num_iter=num_iter) / self.fc_lip_estimate()
                kW = torch.norm(W, p=self.norm_p, dim=1)
                Kmin = torch.min(kW)
                if self.use_Kmax:
                    kW = torch.max(kW)
                # Kmin = torch.min(kW).detach()
                Ki = kW * K
                Kmin = torch.min(Ki).detach()
        else:
            K = self.lc(num_iter=num_iter) / self.fc_lip_estimate()
            kW = torch.norm(W, p=self.norm_p, dim=1)
            if self.use_Kmax:
                kW = torch.max(kW)
            Ki = kW * K
            Kmin = torch.min(kW) #.detach()

        Ki = Ki * self.K_scale
        # print(Ki.detach().min().item(), Ki.detach().max().item())
        if self.use_Kmax:
            blend_factor = (self.alpha * (1.0/Ki)) + (1.0 - self.alpha)
        else:
            blend_factor = (self.alpha * (1.0/Ki[None, :])) + (1.0 - self.alpha)
        weighted_prediction = blend_factor*prediction

        if self.grad_scale:
            grad_scale = (self.alpha * Kmin) + (1.0 - self.alpha)
        else:
            grad_scale = 1.0
        return grad_scale * self.loss(weighted_prediction, target)


class MarginLipschitzCrossEntropyLoss(scheduler.ScheduledModule):
    def __init__(self, lipschitz_computer, num_iter, output_module, norm_p, K_scale=1.0, alpha=1.0, detach_lipschitz_computer=True, grad_scale=False, reduction='mean'):
        super(MarginLipschitzCrossEntropyLoss, self).__init__()
        self.lc = lipschitz_computer
        self.num_iter = num_iter
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        self.detach_lipschitz_computer = detach_lipschitz_computer
        self.K_scale = K_scale
        self.grad_scale = grad_scale
        self.alpha = alpha
        self.norm_p = norm_p
        assert isinstance(output_module, ClassifierLipschitzComputer)
        self.fc_lip_estimate = lambda : output_module.lip_estimate
        self.W = lambda : output_module.parent.weight

    @property
    def reduction(self):
        return self.loss.reduction

    @reduction.setter
    def reduction(self, value):
        self.loss.reduction = value
    

    def forward(self, prediction, target):
        W = self.W()
        if not self.training:
            num_iter = 0
        else:
            num_iter = self.num_iter

        if self.detach_lipschitz_computer:
            with torch.no_grad():
                K = self.lc(num_iter=num_iter) / self.fc_lip_estimate()
                kWij = torch.cdist(W, W, p=self.norm_p) * K

                y_j, j = torch.topk(prediction, k=2, dim=1)
                already_correct = (j[:,0] == target)
                j[already_correct,0] = j[already_correct,1]
                i = target
                j = j[:,0] # take the (second) best
                Ki = torch.gather(kWij[i], index=j.unsqueeze(1), dim=1)
        else:
            raise RuntimeError

        Ki = Ki * self.K_scale
        # print(Ki.detach().min().item(), Ki.detach().max().item())
        blend_factor = (self.alpha * (1.0/Ki)) + (1.0 - self.alpha)
        weighted_prediction = blend_factor*prediction

        if self.grad_scale:
            grad_scale = (self.alpha * Kmin) + (1.0 - self.alpha)
        else:
            grad_scale = 1.0
        return grad_scale * self.loss(weighted_prediction, target)



class ScaledCrossEntropyLoss(scheduler.ScheduledModule):
    def __init__(self, T, reduction='mean'):
        super(ScaledCrossEntropyLoss, self).__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, prediction, target):
        T = self.T
        return F.cross_entropy(prediction/T, target, reduction=self.reduction)


class LoroLoss(scheduler.ScheduledModule):
    def __init__(self, output_module, epsilon):
        super(LoroLoss, self).__init__()
        self.W = lambda : output_module.parent.weight
        # self.norm_p = norm_p
        self.epsilon = epsilon

        self.loss = nn.CrossEntropyLoss()

    def _loro_absent_logit(self, predictions, last_linear_weight, lipschitz_estimate, epsilon):
        def get_Kij(pred, lc, W):
            # C x K <- W.shape
            # N x 1 <- lc.shape
            kW = lc[:, None, None] * W   # kW \in [N x C x K]
            
            # with torch.no_grad():
            y_j, j = torch.max(pred, dim=1)

            # Get the weight column of the predicted class.
            kW_j = W[j] * lc

            # Get weights that predict the value y_j - y_i for all i != j.
            #kW_j \in [N x K x 1], kW \in [N x C x K]
            #kW_ij \in [N x K x C]
            kW_ij = kW_j[:,:,None] - kW.permute(0,2,1)
            
            K_ij = torch.norm(kW_ij, dim=1, p=2)
            #K_ij \in [256 x 10]
            return y_j, K_ij

        y_j, K_ij = get_Kij(predictions, lipschitz_estimate, last_linear_weight)
        y_bot_i = predictions + epsilon * K_ij

        # `y_bot_i` will be zero at the position of class j. However, we don't 
        # want to consider this class, so we replace the zero with negative
        # infinity so that when we find the maximum component for `y_bot_i` we 
        # don't get zero as a result of all of the components we care aobut 
        # being negative.
        y_bot_i[predictions==y_j.unsqueeze(1)] = -np.infty
        y_bot = torch.max(y_bot_i, dim=1, keepdim=True)[0]
        all_logits = torch.cat([predictions, y_bot], dim=1)

        return all_logits

    def forward(self, _Klocal, prediction, target):
        eps = self.epsilon

        K_lip = _Klocal
        W = self.W()
        if self.training:
            prediction = self._loro_absent_logit(prediction, W, K_lip, eps)

        return self.loss(prediction, target)


class PatchLoss(nn.Module):
    def __init__(self, num_patches, loss_type):
        super(PatchLoss, self).__init__()
        self.num_patches = num_patches
        if isinstance(loss_type, dict):
            loss_cfg = loss_type
            loss_type = loss_cfg.pop('type')
        else:
            loss_cfg = {}

        if loss_type.startswith('nn.'):
            loss_class = nn.__dict__[loss_type.replace('nn.','')]
        else:
            loss_class = losses.__dict__[loss_type]
        if 'reduction' not in loss_cfg:
            loss_cfg['reduction'] = 'sum'
        self.loss = loss_class(**loss_cfg)


    def forward(self, prediction, target):
        target = target.unsqueeze(2).repeat(1,1,self.num_patches)
        target = target.view(-1)
        assert target.shape[0] == prediction.shape[0]
        return self.loss(prediction, target)


class SampleNormMaximizationLoss(scheduler.ScheduledModule):
    def __init__(self, lambda_, norm_p, layer_selection=None):
        super(SampleNormMaximizationLoss, self).__init__()
        self.lambda_ = lambda_
        self.norm_p = norm_p
        self.layer_selection = layer_selection

    def _norm_max_loss(self, x):
        x_shape = list(range(len(x.shape)))
        return -torch.norm(x, dim=x_shape[1:], p=self.norm_p).mean()

    def forward(self, model, layer_output):
        loss = 0
        if self.layer_selection is not None:
            keys = self.layer_selection
        else:
            keys = [key for key in layer_output.keys() if hasattr(getattr(model, key), 'weight') and getattr(model, key).weight is not None]

        if len(keys) == 0:
            raise RuntimeError('Module does not have modules with weights.')

        for key in keys:
            loss = self._norm_max_loss(layer_output[key]) + loss

        return self.lambda_ * loss


class MaxMarginClassifierLoss(nn.Module):
    def __init__(self, lambda_):
        super(MaxMarginClassifierLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, model):
        key = model.classifier_layer
        fc = getattr(model, key)
        W = fc.parent.weight
        
        K_ij = torch.cdist(W, W, p=fc.norm_p)
        inds = torch.triu_indices(*K_ij.shape, offset=1)
        K_ij = K_ij[inds[0,:], inds[1,:]]  # select triu entries

        return self.lambda_ * -K_ij.mean()


class GradientPenalty(scheduler.ScheduledModule):
    def __init__(self, lambda_, variant, norm_p, loss_type=None):
        super(GradientPenalty, self).__init__()
        assert variant in ['predicted', 'target', 'loss']

        self.lambda_ = lambda_
        self.variant = variant
        self.norm_p = norm_p

        self.loss = None
        if variant == 'loss' and loss_type is None:
            raise AttributeError('loss_type must be defined, if variant loss is chosen.')
        if loss_type is not None:
            if isinstance(loss_type, dict):
                loss_cfg = loss_type
                loss_type = loss_cfg.pop('type')
            else:
                loss_cfg = {}

            if loss_type.startswith('nn.'):
                loss_class = nn.__dict__[loss_type.replace('nn.','')]
            else:
                loss_class = losses.__dict__[loss_type]
            if 'reduction' not in loss_cfg:
                loss_cfg['reduction'] = 'none'
            self.loss = loss_class(**loss_cfg)

    def forward(self, model, input, target):
        if not torch.is_grad_enabled():
            return torch.full((1,), np.nan).to(input.device)

        input = input.detach().clone()
        input.requires_grad = True

        pred = model._forward_pass(input)['prediction']

        if self.variant == 'predicted':
            values, logits = -pred.max(dim=1)
        elif self.variant == 'target':
            values = -torch.gather(pred, index=target.unsqueeze(1), dim=1)
        elif self.variant == 'loss':
            values = self.loss(pred, target)

        gradients = torch.autograd.grad(
            values, 
            inputs=input, 
            grad_outputs=torch.ones_like(values),
            create_graph=True,
            retain_graph=True
            )[0]

        gradients = gradients.view(input.shape[0], -1)
        if self.norm_p == 2:
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1).clamp(1.e-12))
        else:
            gradients_norm = torch.norm(gradients, dim=1, p=self.norm_p)

        return self.lambda_ * gradients_norm.mean()



class BCEWithLogitsLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(*args, **kwargs)

    def forward(self, prediction, target):
        onehot = torch.zeros_like(prediction)
        onehot.scatter_(dim=1, index=target.unsqueeze(1), src=torch.ones_like(target).unsqueeze(1).float())
        
        return self.loss(prediction, onehot)


class FrobeniusNormDecay(scheduler.ScheduledModule):
    def __init__(self, lambda_):
        super(FrobeniusNormDecay, self).__init__()
        self.lambda_ = lambda_

    def forward(self, model):
        loss = 0
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                fro = torch.norm(module.weight, p='fro')
                loss = fro**2 + loss
            elif isinstance(module, LipschitzLayerComputer) and hasattr(module.parent, 'weight'):
                fro = torch.norm(module.parent.weight, p='fro')
                loss = fro**2 + loss
        return self.lambda_ * loss


class LipschitzBetaDecay(scheduler.ScheduledModule):
    def __init__(self, lambda_):
        super(LipschitzBetaDecay, self).__init__()
        self.lambda_ = lambda_

    def forward(self, model):
        loss = 0
        for name, module in model.named_modules():
            if isinstance(module, LipschitzLayerComputer) and \
                hasattr(module, 'beta'):
                loss = loss + 0.5 * module.beta**2
        return self.lambda_ * loss


class LipschitzDecay(scheduler.ScheduledModule):
    def __init__(self, lambda_, num_iter, bn_weight=1.0):
        super(LipschitzDecay, self).__init__()
        assert num_iter >= 1
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.bn_weight = bn_weight

    def forward(self, model):
        K = 0
        visited = set()
        for name, module in model.named_modules():
            if name in visited:
                # print('visited already', name)
                continue
            if isinstance(module, LipschitzLayerContainer):
                continue
            if isinstance(module, LipschitzLayerComputer):
                if hasattr(module, 'beta'):
                    continue
                if isinstance(module, lip.BatchNormLipschitzComputer):
                    Kmod = module.estimate(num_iter=self.num_iter)
                    K = (self.bn_weight * Kmod**2) + K
                else:
                    Kmod = module.estimate(num_iter=self.num_iter)
                    K = Kmod**2 + K
        return self.lambda_ * K


class InverseSpectralNormDecay(scheduler.ScheduledModule):
    def __init__(self, inv_lambda):
        super(InverseSpectralNormDecay, self).__init__()
        self.inv_lambda = inv_lambda

    def forward(self, model):
        K = 0
        for name, module in model.named_modules():
            if isinstance(module, LipschitzLayerContainer):
                continue
            if isinstance(module, LipschitzLayerComputer):
                if hasattr(module, 'beta'):
                    continue
                if isinstance(module, lip.BatchNormLipschitzComputer):
                    continue
                elif hasattr(module.parent, 'weight') and module.parent.weight is not None:
                    W = module.parent.weight.view(module.parent.weight.shape[0], -1)
                    b = torch.rand(W.shape[0], 1)
                    b /= torch.norm(b, p=2)

                    Wt = W.t()
                    WtW = torch.matmul(W, Wt)
                    WtWinv = torch.inverse(WtW)
                    b_new = torch.matmul(WtWinv, b)
                    b = b_new / torch.norm(b_new, p=2)
                    A = torch.mv(Wt,b)
                    B = torch.matmul(b.t(),b)
                    min_sigma = torch.norm(A,p=2)/B
                    # _,s,_ = torch.svd(module.parent.weight, some=True, compute_uv=False)
                    K = (1./min_sigma)**2 + K
        return self.inv_lambda * K


class DepthDependentLipschitzDecay(scheduler.ScheduledModule):
    def __init__(self, func_of_depth, lambda_, num_iter):
        super(DepthDependentLipschitzDecay, self).__init__()
        self.lambda_ = lambda_
        self.num_iter = num_iter
        self.func_of_depth = func_of_depth

    def depth_factor(self, l, L):
        start_val = self.func_of_depth['start_val']
        end_val = self.func_of_depth['end_val']
        power = self.func_of_depth['power']
        if start_val < end_val:
            scale_factor = ((float(l) / (L-1))) ** power
            factor = start_val + (end_val-start_val) * scale_factor
        else:
            scale_factor = (1.0-(float(l) / (L-1))) ** power
            factor = end_val + (start_val-end_val) * scale_factor

        return factor

    def forward(self, model):
        K = 0
        modules = list(model.modules())
        L = 0
        for module in modules:
            if isinstance(module, LipschitzLayerComputer):
                L += 1

        l = 0
        for module in modules:
            if isinstance(module, LipschitzLayerComputer):
                factor = self.depth_factor(l, L)
                # print(l, L, factor)
                K = factor * (module.estimate(num_iter=self.num_iter))**2 + K
                l += 1
        return self.lambda_ * K



# class GlobalLipschitzDecay(scheduler.ScheduledModule):
#     def __init__(self, lipschitz_computer, lambda_, num_iter):
#         super(GlobalLipschitzDecay, self).__init__()
#         self.num_iter = num_iter
#         self.lambda_ = lambda_
#         self.lc = lipschitz_computer

#     def forward(self, model):
#         K = self.lc(num_iter=self.num_iter)
#         print(K)
#         return self.lambda_ * K

class GlobalLipschitzDecay(scheduler.ScheduledModule):
    def __init__(self, lipschitz_computer, lambda_, num_iter, pow=1.0, ignore_fc=False, classifier_each_logit=False, integrate_loss=None):
        super(GlobalLipschitzDecay, self).__init__()
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.lc = lipschitz_computer
        self.ignore_fc = ignore_fc
        self.classifier_each_logit = classifier_each_logit
        self.integrate_loss = integrate_loss
        self.pow = pow
        if integrate_loss is not None:
            assert ignore_fc == False
            assert classifier_each_logit == False

    def forward(self, model):
        K = self.lc(num_iter=self.num_iter)
        return self.lambda_ * K**self.pow


class LipschitzContingent(scheduler.ScheduledModule):
    def __init__(self, lambda_, contingent, num_iter):
        super(LipschitzContingent, self).__init__()
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.contingent = contingent

    def forward(self, model):
        K = 0
        for module in model.modules():
            if isinstance(module, LipschitzLayerComputer):
                K = module.estimate(num_iter=self.num_iter) + K

        contingent = self.contingent

        loss = torch.clamp(K-contingent, min=0)
        return self.lambda_ * loss

    def __repr__(self):
        return 'LipschitzContingent(lambda={}, contingent={})'.format(self.lambda_, self.contingent)


class GlobalLipschitzTarget(scheduler.ScheduledModule):
    def __init__(self, lambda_, target, num_iter):
        super(GlobalLipschitzTarget, self).__init__()
        self.num_iter = num_iter
        self.lambda_ = lambda_
        self.target = target

    def forward(self, model):
        K = 1
        for module in model.modules():
            if isinstance(module, LipschitzLayerComputer):
                K = module.estimate(num_iter=self.num_iter) * K

        target = self.target

        loss = torch.abs(K-target)
        return self.lambda_ * loss

    def __repr__(self):
        return 'GlobalLipschitzTarget(lambda={}, target={})'.format(self.lambda_, self.target)


class CosineMaxMarginLoss(scheduler.ScheduledModule):
    def __init__(self, num_classes, epsilon, num_iter, lipschitz_computer, reduction='mean'):
        super(CosineMaxMarginLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.lc = lipschitz_computer
        self.numeric_eps = numeric_eps
        self.robust_threshold = 0
        self.reduction = reduction
        self.norm_factor = norm_factor

    def forward(self, prediction, target):
        if isinstance(self.epsilon, scheduler.Scheduler):
            epsilon = self.epsilon.val
        else:
            epsilon = self.epsilon

        prediction_norm = torch.norm(prediction, p=2, dim=1, keepdim=True)

        return None



class CosineLoss(nn.Module):
    def __init__(self, num_classes, reduction='mean', norm_factor=0, eps=1.e-9):
        super(CosineLoss, self).__init__()
        self.norm_factor = norm_factor
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction

    def forward(self, prediction, target):
        loss = 0
        normed_pred = prediction / (torch.norm(prediction, p=2, dim=1, keepdim=True) + self.eps)
        for i in range(self.num_classes):
            tomax = normed_pred[target==i][:, i]
            loss = loss + (-tomax.sum())
        norm_loss = ((1.0-torch.norm(prediction, p=2, dim=1, keepdim=True))**2)
        if self.reduction == 'mean':
            loss = (1./prediction.shape[0])*loss
            norm_loss = norm_loss.mean()
        else:
            norm_loss = norm_loss.sum()
        loss = loss + self.norm_factor * norm_loss

        return loss

    def __repr__(self):
        return 'CosineLoss(norm_factor={}, reduction={})'.format(self.norm_factor, self.reduction)


class LipschitzIntegratedCosineLoss(scheduler.ScheduledModule):
    def __init__(self, num_classes, epsilon, num_iter, lipschitz_computer, norm_factor=1.0, reduction='mean', numeric_eps=1.e-9):
        super(LipschitzIntegratedCosineLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.lc = lipschitz_computer
        self.numeric_eps = numeric_eps
        self.robust_threshold = 0
        self.reduction = reduction
        self.norm_factor = norm_factor

        if isinstance(self.epsilon, scheduler.Scheduler):
            self.register_scheduler(self.epsilon)

    def forward(self, prediction, target):
        if isinstance(self.epsilon, scheduler.Scheduler):
            epsilon = self.epsilon.val
        else:
            epsilon = self.epsilon

        prediction_norm = torch.norm(prediction, p=2, dim=1, keepdim=True)

        loss = 0
        normed_pred = prediction / (prediction_norm + self.numeric_eps)
        for i in range(self.num_classes):
            tomax = normed_pred[target==i][:, i]
            loss = loss + (-tomax.sum())

        # norm should be greater than: lipK * epsilon
        lipK = self.lc(num_iter=self.num_iter)
        target_norm = epsilon * lipK

        with torch.no_grad():
            self.robust_threshold = target_norm.detach().item()
            # print('robust threshold {:.3f}'.format(self.robust_threshold))

        norm_loss = 0.5*((target_norm-prediction_norm)**2)
        if self.reduction == 'mean':
            loss = (1./prediction.shape[0])*loss
            norm_loss = norm_loss.mean()
        else:
            norm_loss = norm_loss.sum()
        loss = loss + self.norm_factor * norm_loss

        return loss

    def __repr__(self):
        return 'LipschitzIntegratedCosineLoss(eps={}, num_iter={}, norm_factor={})'.format(self.epsilon, self.num_iter, self.norm_factor)


class LipschitzLagrangeCosineLoss(scheduler.ScheduledModule):
    def __init__(self, num_classes, epsilon, num_iter, lipschitz_computer, 
        output_lagrange_multiplier=1.0, lipschitz_lagrange_multiplier=1.0, 
        output_norm_target=1.0, lipschitz_target=1.0,
        reduction='mean', numeric_eps=1.e-9):
        """
        L(f,x,y) = <f(x)/||f(x)||_2 , y> + lambda * (K_lip - Lip(f))^2 + gamma * (K_norm - ||f(x)||_2)^2
        """
        super(LipschitzLagrangeCosineLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.lc = lipschitz_computer
        self.numeric_eps = numeric_eps
        self.robust_threshold = 0
        self.reduction = reduction
        self.lam = lipschitz_lagrange_multiplier
        self.gam = output_lagrange_multiplier
        self.lipschitz_target = lipschitz_target
        self.output_norm_target = output_norm_target

        if isinstance(self.epsilon, scheduler.Scheduler):
            self.register_scheduler(self.epsilon)

    def forward(self, prediction, target):
        if isinstance(self.epsilon, scheduler.Scheduler):
            epsilon = self.epsilon.val
        else:
            epsilon = self.epsilon

        prediction_norm = torch.norm(prediction, p=2, dim=1, keepdim=True)

        loss = 0
        normed_pred = prediction / (prediction_norm + self.numeric_eps)
        for i in range(self.num_classes):
            tomax = normed_pred[target==i][:, i]
            loss = loss + (-tomax.sum())

        # norm should be greater than: lipK * epsilon
        lipK = self.lc(num_iter=self.num_iter)

        lip_multiplier = 0.5 * self.lam * (self.lipschitz_target - lipK)**2

        desired_norm = epsilon * lipK.detach()
        # out_norm_multiplier = -0.5 * self.gam * prediction_norm
        out_norm_multiplier = 0.5 * self.gam * (desired_norm - prediction_norm)**2
        # out_norm_multiplier = 0.5 * self.gam * (self.output_norm_target - prediction_norm)**2

        with torch.no_grad():
            self.robust_threshold = desired_norm.item()
            # print('robust threshold {:.3f}'.format(self.robust_threshold))

        lagrange_multiplier = lip_multiplier + out_norm_multiplier
        if self.reduction == 'mean':
            loss = (1./prediction.shape[0])*loss
            lagrange_multiplier = lagrange_multiplier.mean()
        else:
            lagrange_multiplier = lagrange_multiplier.sum()
        
        loss = loss + lagrange_multiplier
        return loss

    def __repr__(self):
        return 'LipschitzLagrangeCosineLoss(eps={}, num_iter={}, lambda={}, gamma={}, Lip-target={})'.format(self.epsilon, self.num_iter, self.lam, self.gam, self.lipschitz_target)


class LipschitzLagrangeLoss(scheduler.ScheduledModule):
    def __init__(self, num_classes, epsilon, num_iter, lipschitz_computer, 
        output_lagrange_multiplier=1.0, lipschitz_lagrange_multiplier=1.0, 
        output_norm_target=1.0, lipschitz_target=1.0,
        reduction='mean', numeric_eps=1.e-9):
        """
        L(f,x,y) = <f(x)/||f(x)||_2 , y> + lambda * (K_lip - Lip(f))^2 + gamma * (K_norm - ||f(x)||_2)^2
        """
        super(LipschitzLagrangeLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.lc = lipschitz_computer
        self.numeric_eps = numeric_eps
        self.robust_threshold = 0
        self.reduction = reduction
        self.lam = lipschitz_lagrange_multiplier
        self.gam = output_lagrange_multiplier
        self.lipschitz_target = lipschitz_target
        self.output_norm_target = output_norm_target

        if isinstance(self.epsilon, scheduler.Scheduler):
            self.register_scheduler(self.epsilon)

    def forward(self, prediction, target):
        if isinstance(self.epsilon, scheduler.Scheduler):
            epsilon = self.epsilon.val
        else:
            epsilon = self.epsilon

        loss = 0
        prediction_norm = torch.norm(prediction, p=2, dim=1, keepdim=True)
        normed_pred = prediction / (prediction_norm + self.numeric_eps)
        for i in range(self.num_classes):
            tomax = -normed_pred[target==i][:, i]
            norm_loss = self.gam * 0.5 * (self.output_norm_target - prediction[target==i][:, i])**2
            loss = loss + tomax.sum() + norm_loss.sum()

        # norm should be greater than: lipK * epsilon
        lipK = self.lc(num_iter=self.num_iter)

        lip_multiplier = 0.5 * self.lam * (self.lipschitz_target - lipK)**2

        desired_norm = epsilon * lipK.detach()

        with torch.no_grad():
            self.robust_threshold = desired_norm.item()
            # print('robust threshold {:.3f}'.format(self.robust_threshold))

        lagrange_multiplier = lip_multiplier
        if self.reduction == 'mean':
            loss = (1./prediction.shape[0])*loss
            lagrange_multiplier = lagrange_multiplier.mean()
        else:
            lagrange_multiplier = lagrange_multiplier.sum()
        
        loss = loss + lagrange_multiplier
        return loss

    def __repr__(self):
        return 'LipschitzLagrangeLoss(eps={}, num_iter={}, lambda={}, gamma={}, Lip-target={})'.format(self.epsilon, self.num_iter, self.lam, self.gam, self.lipschitz_target)


def mmd_loss(predictions, targets, num_classes, lam=0.1):
    loss = 0
    for i in range(num_classes):
        tomax = predictions[targets==i][:, i]
        tomin = predictions[targets!=i][:, i]
        loss = loss + (-tomax.mean()) + tomin.mean()
    loss = (1./num_classes)*loss + lam*(predictions**2).mean()
    return loss

def vector_squash(v):
    vnorm = torch.norm(v, dim=1, p=2, keepdim=True) + 1.e-8
    vnorm_sq = vnorm**2
    scaling = (vnorm_sq) / (1.0 + vnorm_sq)
    return (v/vnorm) * scaling

def proj_loss(predictions, targets, num_classes):
    loss = 0
    normed_pred = predictions / (torch.norm(predictions, p=2, dim=1, keepdim=True) + 1.e-8)
#     predictions = vector_squash(predictions)
    for i in range(num_classes):
        tomax = normed_pred[targets==i][:, i]
#         tomin = predictions[targets!=i][:, i]
        loss = loss + (-tomax.sum()) # + tomin.mean()
#         loss = loss + (-tomax.mean()) + 0.1*(-predictions[targets==i][:, i].mean())
    loss = (1./predictions.shape[0])*loss
#     loss = loss + (torch.norm(predictions, p=2, dim=1, keepdim=True).mean()
#     norm_loss = 0.03 * ((1.0-torch.norm(predictions, p=2, dim=1, keepdim=True))**2).mean()
    # norm_loss = 0.01*((1-torch.norm(predictions, p=2, dim=1, keepdim=True))**2).mean()
    norm_loss = 0
    loss = loss + norm_loss
    return loss


def gloro_loss(predictions, targets, last_linear_weight, lipschitz_estimate, epsilon=0.15, reduction='mean'):
    def get_Kij(pred, lc, W):
        kW = W*lc
        
        with torch.no_grad():
            y_j, j = torch.max(pred, dim=1)

        # Get the weight column of the predicted class.
        kW_j = kW[j]

        # Get weights that predict the value y_j - y_i for all i != j.
        #kW_j \in [256 x 128 x 1], kW \in [1 x 10 x 128]
        #kW_ij \in [256 x 128 x 10]
        kW_ij = kW_j[:,:,None] - kW.transpose(1,0).unsqueeze(0)
        
        K_ij = torch.norm(kW_ij, dim=1, p=2)
        #K_ij \in [256 x 10]
        return y_j, K_ij

    #with torch.no_grad():
    y_j, K_ij = get_Kij(predictions, lipschitz_estimate, last_linear_weight)
    y_bot_i = predictions + epsilon * K_ij

    # `y_bot_i` will be zero at the position of class j. However, we don't 
    # want to consider this class, so we replace the zero with negative
    # infinity so that when we find the maximum component for `y_bot_i` we 
    # don't get zero as a result of all of the components we care aobut 
    # being negative.
    y_bot_i[predictions==y_j.unsqueeze(1)] = -np.infty
    y_bot = torch.max(y_bot_i, dim=1, keepdim=True)[0]
    all_logits = torch.cat([predictions, y_bot], dim=1)
  
    cls_output = F.log_softmax(all_logits, dim=1)
    loss = F.nll_loss(cls_output, targets, reduction=reduction)

    return loss