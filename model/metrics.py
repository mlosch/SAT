import os
import re
import itertools
from collections import OrderedDict
import struct

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib import util

class Accuracy(nn.Module):
    def __init__(self, topk=1):
        super(Accuracy, self).__init__()
        self.topk = topk

    def forward(self, prediction, target):
        with torch.no_grad():
            batch_size = target.size(0)

            _, pred = prediction.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            correct_k = correct.reshape(-1).float().sum(0)
            return correct_k.mul_(100.0 / batch_size)

    def __repr__(self):
        return 'Accuracy(topk={})'.format(self.topk)


class Entropy(nn.Module):
    def forward(self, prediction):
        with torch.no_grad():
            log_p = F.log_softmax(prediction, dim=1)
            p = prediction.exp()
            entropies = -torch.sum(p*log_p, dim=1, keepdim=False)
        return entropies.mean()

    def __repr__(self):
        return 'Entropy'


class LayerProbeAccuracies(Accuracy):
    def __init__(self, probe_loss_name, topk=1):
        super(LayerProbeAccuracies, self).__init__(topk=topk)
        self.probe_loss_name = probe_loss_name

    def forward(self, model, target):
        assert self.probe_loss_name in model.loss
        with torch.no_grad():
            predictions = model.loss[self.probe_loss_name].last_outputs
            acc_per_probe = dict()
            for name, prediction in predictions.items():
                acc = super(LayerProbeAccuracies, self).forward(prediction, target)
                acc_per_probe[name] = acc

            return acc_per_probe



class DomainAccuracy(nn.Module):
    def __init__(self, domain, topk=1, evaluate_only_during_validation=False):
        super(DomainAccuracy, self).__init__()
        self.topk = topk
        self.register_buffer('domain', torch.Tensor(domain))
        self.evaluate_only_during_validation = evaluate_only_during_validation

    def forward(self, prediction, target):
        if self.evaluate_only_during_validation and self.training:
            return prediction.new(1).fill_(0.0)

        with torch.no_grad():
            mask = util.domain_mask(self.domain, target)
            batch_size = len(target[mask])

            if batch_size == 0:
                return None

            prediction_ = prediction[mask]
            _, pred = prediction_.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[mask].view(1, -1).expand_as(pred))

            correct_k = correct.reshape(-1).float().sum(0)
            return correct_k.mul_(100.0 / batch_size)
            


class InstanceDomainAccuracy(nn.Module):
    def __init__(self, domain_cfg, topk=1, evaluate_only_during_validation=False):
        super(InstanceDomainAccuracy, self).__init__()
        self.topk = topk
        if domain_cfg['ranking'] is not None:
            ranking = torch.load(domain_cfg['ranking'])
            domain = domain_cfg['domain']
            bins = domain_cfg['bins']
            assert max(domain) < bins
            N = len(ranking)
            if N%bins != 0:
                print('InstanceDomainAccuracy :: Warning :: bins={} does not divide domain ranking list of length {}'.format(bins, N))
                # raise RuntimeError('bins={} does not divide domain ranking list of length {}'.format(bins, N))
            bin_width = N//bins
            domain_ = []
            for i in range(0,N,bin_width):
                bin_idx = i//bin_width
                if bin_idx in domain:
                    domain_ += ranking[i:i+bin_width].tolist()
        else:
            assert len(domain_cfg['domain']) > 0
            domain_ = domain_cfg['domain']
        self.register_buffer('domain', torch.Tensor(domain_))
        self.evaluate_only_during_validation = evaluate_only_during_validation

    def forward(self, prediction, target, imids):
        if self.evaluate_only_during_validation and self.training:
            return prediction.new(1).fill_(0.0)
            
        with torch.no_grad():
            mask = util.domain_mask(self.domain, imids)
            batch_size = len(target[mask])

            if batch_size == 0:
                return None

            prediction_ = prediction[mask]
            _, pred = prediction_.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[mask].view(1, -1).expand_as(pred))

            correct_k = correct.reshape(-1).float().sum(0)
            return correct_k.mul_(100.0 / batch_size)


class DomainSubsetAccuracy(nn.Module):
    def __init__(self, domain, topk=1):
        super(DomainSubsetAccuracy, self).__init__()
        self.topk = topk
        self.domain = domain

    def forward(self, prediction, target):
        with torch.no_grad():
            # print(self.domain, prediction.max(1)[1].cpu().numpy().tolist(), target.cpu().numpy().tolist())
            mask = util.domain_mask(self.domain, target)
            # mask = torch.ones(len(target)).bool()
            # print(self.domain, prediction[mask][:, self.domain].max(1)[1].cpu().numpy().tolist(), target[mask].cpu().numpy().tolist())
            batch_size = len(target[mask]) #mask.sum()

            prediction_ = prediction[mask].clone()
            prediction_[:, self.domain] += 100000.  #hack to weight domain classes higher
            _, pred = prediction_.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[mask].view(1, -1).expand_as(pred))

            correct_k = correct.reshape(-1).float().sum(0)
            return correct_k.mul_(100.0 / batch_size)

    def __repr__(self):
        return 'DomainAccuracy(topk={}, domain={})'.format(self.topk, self.domain)


class RobustAccuracy(nn.Module):
    def __init__(self, model, output_module, epsilon, data_std=[1,1,1], has_absent_logit=False, norm_p=2):
        super(RobustAccuracy, self).__init__()
        assert hasattr(output_module, 'parent'), 'output module must have attribute parent (as is defined in LipschitzLayerComputer.'
        self.W = lambda : output_module.parent.weight
        self.output_module_lc = output_module.estimate
        self.lc = model.lipschitz_estimate
        self.epsilon = epsilon
        self.norm_p = norm_p
        self.data_std = torch.Tensor(data_std)
        self.has_absent_logit = has_absent_logit

    def forward(self, prediction, target):
        if self.has_absent_logit:
            prediction = prediction[:,:-1] # remove gloro logit

        # calculate lipschitz constants per logit
        eps = self.epsilon

        with torch.no_grad():
            K_lip = 1./self.data_std.min()
            K_lip = K_lip * (self.lc(num_iter=0) / self.output_module_lc(num_iter=0))
            W = self.W()

            # def get_Kij(pred, lc, W):
            #     kW = W*lc

            #     y_j, j = torch.max(pred, dim=1)

            #     # Get the weight column of the predicted class.
            #     kW_j = kW[j]

            #     # Get weights that predict the value y_j - y_i for all i != j.
            #     #kW_j \in [256 x 128 x 1], kW \in [1 x 10 x 128]
            #     #kW_ij \in [256 x 128 x 10]
            #     kW_ij = kW_j[:,:,None] - kW.transpose(1,0).unsqueeze(0)
                
            #     K_ij = torch.norm(kW_ij, dim=1, p=self.norm_p)
            #     #K_ij \in [256 x 10]
            #     return y_j, j, K_ij

            def get_Kij(pred, lc, W):
                kW = W*lc

                y_j, j = torch.max(pred, dim=1)

                # Get the weight column of the predicted class.
                kW_j = kW[j]

                # Get weights that predict the value y_j - y_i for all i != j.
                #kW_j \in [256 x 128 x 1], kW \in [1 x 10 x 128]
                #kW_ij \in [256 x 128 x 10]
                kW_ij = kW_j[:,:,None] - kW.transpose(1,0).unsqueeze(0)
                
                K_ij = torch.norm(kW_ij, dim=1, p=self.norm_p)
                #K_ij \in [256 x 10]
                return y_j, j, K_ij

            # with torch.no_grad():
            y_j, pred_class, K_ij = get_Kij(prediction, K_lip, W)
            y_bot_i = prediction + eps * K_ij
            y_bot_i[prediction==y_j.unsqueeze(1)] = -np.infty
            y_bot = torch.max(y_bot_i, dim=1, keepdim=False)[0]

            robust = (pred_class == target) & (y_j > y_bot)

            batch_size = target.size(0)
            return robust.float().sum() * (100.0 / batch_size)


class RobustAccuracyV2(RobustAccuracy):

    def forward(self, prediction, target):
        # calculate lipschitz constants per logit
        eps = self.epsilon

        with torch.no_grad():
            K_lip = self.lc(num_iter=0) / self.output_module_lc(num_iter=0)
            W = self.W()
            K_logits = K_lip * torch.norm(W,dim=1,p=self.norm_p)  # num_classes

            y_j, j = torch.topk(prediction, k=2, dim=1)

            K_best = K_logits[j[:,0]]  # N (num_samples)
            K_second = K_logits[j[:,1]]  # N (num_samples)
            K_margin = (K_best+K_second) * eps

            y_bot = y_j[:,1] + K_margin # 2nd best logit + margin
            robust = (j[:,0] == target) & (y_j[:,0] > y_bot) # best logit still greater than y_bot?

            batch_size = target.size(0)
            return robust.float().sum() * (100.0 / batch_size)


class ConfusionMatrixMetric(nn.Module):
    def __init__(self, num_classes, robust_logit_index=None, labels=None):
        super(ConfusionMatrixMetric, self).__init__()
        self.num_classes = num_classes
        self.labels = labels
        self.robust_logit_index = robust_logit_index
        if robust_logit_index is not None:
            assert robust_logit_index == -1

    def forward(self, prediction, target):
        accuracies = []

        confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
        with torch.no_grad():
            if self.robust_logit_index is not None:
                pred = torch.max(prediction[:, :-1], dim=1)[1]
            else:
                pred = torch.max(prediction, dim=1)[1]
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
        return ConfusionMatrix(confusion_matrix, self.labels)


class GloroCleanAccuracy(Accuracy):
    def forward(self, prediction, target):
        with torch.no_grad():
            # strip last logit
            return super(GloroCleanAccuracy, self).forward(prediction[:, :-1], target)

    def __repr__(self):
        return 'GloroCleanAccuracy(topk={})'.format(self.topk)


class LipCosRobustAccuracy(nn.Module):
    def __init__(self, lipcosloss):
        super(LipCosRobustAccuracy, self).__init__()
        self.topk = 1
        # self.lipcosloss = lipcosloss
        assert hasattr(lipcosloss, 'robust_threshold')
        self.robust_threshold = lambda : lipcosloss.robust_threshold

    def forward(self, prediction, target):
        with torch.no_grad():
            batch_size = target.size(0)

            pred_vals, pred = prediction.topk(self.topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            robust = torch.norm(prediction, dim=1, p=2) > self.robust_threshold()
            robust_and_correct = correct & robust

            correct_k = robust_and_correct.reshape(-1).float().sum(0)
            return correct_k.mul_(100.0 / batch_size)



class LogitDistance(nn.Module):
    def forward(self, prediction):
        with torch.no_grad():
            N = prediction.shape[1]
            ntriu_elem = ((N*(N-1))/2)
            vals = prediction[:,:,None].repeat(1, 1, N)

            pairwise_diff = vals - vals.permute(0,2,1)
            triu = torch.triu(pairwise_diff)
            avg_diff = triu.abs().sum(dim=-1) / ntriu_elem

            return avg_diff.mean()


class MarginRatio(nn.Module):
    def __init__(self, data_std):
        super(MarginRatio, self).__init__()
        self.data_scaling = np.min(data_std)

    def forward(self, model, prediction, target):
        K = model.lipschitz_estimate(num_iter=0) * 1./self.data_scaling
        logits, _ = torch.topk(prediction, k=2, dim=1)
        margin = logits[:,0] - logits[:,1]
        ratio = margin / K
        return ratio.mean()


class WeightRanks(nn.Module):
    def forward(self, model):
        if not self.training:
            return None

        ranks = OrderedDict()

        with torch.no_grad():
            for name, module in model.named_modules():
                if not hasattr(module, 'weight'):
                    continue

                W = module.weight
                W = W.view(W.shape[0], -1)

                ranks[name] = torch.matrix_rank(W)

class SingularValues(nn.Module):
    def forward(self, model):
        if not self.training:
            return None

        values = OrderedDict()

        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, 'running_var'):
                    values[name+'.running_var'] = Histogram(module.running_var)
                    
                if not hasattr(module, 'weight'):
                    continue

                W = module.weight

                if len(W.shape) == 1:
                    s = W
                else:
                    W = W.view(W.shape[0], -1)
                    _, s, _ = torch.svd(W, some=False, compute_uv=False)

                values[name] = Histogram(s)

        return values


class LayerOutputNorms(nn.Module):
    def __init__(self, dim, p=2, name_regex=None):
        super(LayerOutputNorms, self).__init__()
        self.p = p
        self.dim = tuple(dim)
        if name_regex is not None:
            self.regex = re.compile(name_regex)
        else:
            self.regex = None

    def forward(self, input, layer_output):
        norms = OrderedDict()
        norms['input'] = torch.norm(input, p=self.p, dim=self.dim).mean()

        for name, output in layer_output.items():
            if self.regex is not None:
                if self.regex.match(name) is None:
                    continue
            dim = self.dim
            if max(dim) >= output.ndim:
                dim = [d for d in dim if d < output.ndim]
            norms[name] = torch.norm(output, p=self.p, dim=dim).mean()

        return norms

    def __repr__(self):
        return 'LayerOutputNorms(p={}, dim={})'.format(self.p, self.dim)


class OutputDumper(nn.Module):
    def __init__(self, layer, save_path, every_epoch=1):
        super(OutputDumper, self).__init__()
        self.layer = layer
        self.save_path = save_path
        self.every_epoch = every_epoch

    def dump(self, tensor, fileprefix):
        if self.training:
            filep = os.path.join(self.save_path, 'train', '{}.bin'.format(fileprefix))
        else:
            filep = os.path.join(self.save_path, 'val', '{}.bin'.format(fileprefix))
        if not os.path.exists(os.path.dirname(filep)):
            os.makedirs(os.path.dirname(filep))

        util.write_bin_data(filep, tensor)

    def forward(self, layer_output, target, imids=None):
        if self.layer == 'target':
            self.dump(target.detach().cpu(), 'target')
        elif self.layer == 'imids':
            self.dump(imids.detach().cpu(), 'imids')
        else:
            with torch.no_grad():
                output = layer_output[self.layer].detach()
            self.dump(output.cpu(), self.layer)


class ProbeOutputDumper(OutputDumper):
    def __init__(self, probe_loss_name, save_path, every_epoch=1):
        super(ProbeOutputDumper, self).__init__(probe_loss_name, save_path, every_epoch)

    def forward(self, model, target, imids=None):
        if self.layer == 'target':
            self.dump(target.detach().cpu(), 'target')
        elif self.layer == 'imids':
            self.dump(imids.detach().cpu(), 'imids')
        else:
            with torch.no_grad():
                outputs = model.loss[self.layer].last_outputs
                for name, values in outputs.items():
                    self.dump(values.cpu(), 'probe_'+name)


class OutputNormDumper(OutputDumper):
    def forward(self, layer_output, target, imids=None):
        if self.layer in layer_output:
            with torch.no_grad():
                output = layer_output[self.layer].detach()
                dims = list(output.shape)
                output = torch.norm(output, p=2, dim=dims[1:])
            self.dump(output.cpu(), self.layer)
        else:
            super(OutputNormDumper, self).forward(layer_output, target, imids)
            

class ClasswiseEntropy(nn.Module):
    def __init__(self, x_labels, save_path=None):
        super(ClasswiseEntropy, self).__init__()
        self.x_labels = x_labels
        self.save_path = save_path

    def forward(self, prediction, target):
        with torch.no_grad():
            p = F.softmax(prediction, dim=1)
            log_p = F.log_softmax(prediction, dim=1)
            entropies = -torch.sum(p*log_p, dim=1, keepdim=False).cpu()
            target_cpu = target.cpu()
            histogram = torch.zeros(prediction.shape[1], dtype=entropies.dtype)
            # print(histogram.shape, prediction.shape, target.shape, entropies.shape)
            histogram.scatter_add_(dim=0, index=target_cpu, src=entropies)
            normalization = torch.zeros(prediction.shape[1])
            normalization.scatter_add_(dim=0, index=target_cpu, src=torch.ones(prediction.shape[0]))
            return PlottableHistogram(histogram.squeeze(), normalization.squeeze(), x_labels=self.x_labels, save_path=self.save_path, during_training=self.training)



###############################################################################
## Container classes

class Histogram(object):
    def __init__(self, values, already_binned=False):
        super(Histogram, self).__init__()
        self._values = values
        self._already_binned = already_binned

    @property
    def already_binned(self):
        return self._already_binned

    @property
    def values(self):
        return self._values

class RunningHistogram(Histogram):
    def __init__(self, values, normalization=1):
        super(RunningHistogram, self).__init__(values)
        self.normalization = normalization

    def update(self, hist):
        if isinstance(hist, RunningHistogram):
            self._values += hist._values
        else:
            self._values += hist.values
        self.normalization += hist.normalization

    @property
    def values(self):
        return self._values / self.normalization


class Plottable(object):
    def plot(self, epoch=None):
        raise NotImplementedError

class PlottableHistogram(RunningHistogram, Plottable):
    def __init__(self, *args, x_labels=None, save_path=None, during_training=True, **kwargs):
        super(PlottableHistogram, self).__init__(*args, **kwargs)
        self.x_labels = x_labels
        self.save_path = save_path
        self.training = during_training

    def plot(self, epoch=None):
        if self.save_path is not None:
            save_filep = os.path.join(self.save_path, 'train' if self.training else 'val', 'epoch_{}.pth'.format(epoch))
            os.makedirs(os.path.dirname(save_filep), exist_ok=True)
            torch.save(self.values.cpu(), save_filep)

        bar_heights = self.values
        plt.style.use('seaborn')
        fig = plt.figure(1, dpi=150)
        plt.bar(np.arange(len(bar_heights))+0.5, bar_heights.cpu().numpy(), width=0.8)
        if self.x_labels is not None:
            plt.xticks(np.arange(len(bar_heights))+0.5, self.x_labels, rotation=90, fontsize=5)

        return fig


class ConfusionMatrix(Plottable):
    def __init__(self, values, labels=None):
        super(ConfusionMatrix, self).__init__()
        self._values = values
        self.labels = labels

    def update(self, mat):
        self._values += mat._values

    def plot(self, epoch=None):
        cm = self._values/self._values.sum(1)

        plt.style.use('default')
        if cm.shape[0] > 10:
            fig = plt.figure(0, figsize=(20,20))
        else:
            fig = plt.figure(0)
        _=plt.matshow(cm, fignum=0, cmap='Reds', vmin=0, vmax=1)
        _=plt.colorbar(fraction=0.046, pad=0.04)

        N = cm.shape[0]

        if self.labels is not None:
            plt.xticks(torch.arange(N), self.labels, rotation=90, fontsize=6)
            plt.yticks(torch.arange(N), self.labels, fontsize=6)

        # Print values in cells
        for i, j in itertools.product(range(N), range(N)):
            plt.text(j, i, "{:.0f}".format(cm[i, j]*100.),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="black", fontsize=6)

        plt.xlabel('Predicted')
        plt.ylabel('True')

        return fig
