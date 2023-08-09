import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize

from lib import util
from lib.scheduler import Scheduler, ScheduledModule, PolynomialScheduler
from model import BaseModel
from model.metrics import Histogram
import model.layers as layers


def wrap_into_lipschitz_layer(value, norm_p, ignore_batchnorm=False):
    # wrap conv and linear layers into LipschitzLayerComputers
    if isinstance(value, nn.Sequential):
        value = SequentialLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm)
    if isinstance(value, nn.Conv2d):
        value = Conv2dLipschitzComputer(value, norm_p)
    elif isinstance(value, nn.Linear):
        value = LinearLipschitzComputer(value, norm_p)
    if isinstance(value, layers.BasicResNetBlock):
        value = ResNetBlockLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm)
    if isinstance(value, layers.MaxNormTo1):
        value = ScalingLipschitzComputer(value, norm_p)
    if isinstance(value, nn.AvgPool2d):
        value = AvgPool2dLipschitzComputer(value, norm_p)
    if isinstance(value, nn.AdaptiveAvgPool2d):
        value = AdaptiveAvgPool2dLipschitzComputer(value, norm_p)
    if isinstance(value, layers.BNConv2d):
        value = BNConv2dLipschitzComputer(value, norm_p)
    if isinstance(value, layers.PreActBNConv2d):
        value = PreActBNConv2dLipschitzComputer(value, norm_p)
    if isinstance(value, layers.BlendBatchNorm2d):
        value = BlendBatchNormLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm)
    elif isinstance(value, layers.RunningBatchNorm2d):
        #TODO: eventually remove this constraint
        beta = 1.0
        value = BatchNormLipschitzConstraint(value, norm_p, beta)
    elif isinstance(value, nn.modules.batchnorm._BatchNorm):
        # beta = 1.5
        # value = BatchNormLipschitzConstraint(value, norm_p, beta)
        value = BatchNormLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm)
    elif isinstance(value, nn.modules.normalization.GroupNorm):
        value = GroupNormLipschitzComputer(value, norm_p, ignore_batchnorm=ignore_batchnorm)
    elif isinstance(value, layers.GradBatchNorm2d):
        value = GradBatchNorm2dLipschitzComputer(value, norm_p)
    return value


class LipschitzModel(BaseModel):
    def __init__(self, classifier_layer=None, p=2, 
        use_softmax=False, fix_last_layer_lip=False, 
        calibrate_outputs=False, ignore_batchnorm=False,
        converge_every_training_step=False):
        super(LipschitzModel, self).__init__()
        self.fix_last_layer_lip = fix_last_layer_lip
        self.norm_p = p
        self.classifier_layer = classifier_layer
        self.use_softmax = use_softmax
        self.calibrate_outputs = calibrate_outputs
        self.ignore_batchnorm = ignore_batchnorm
        # all layers are added in model_builder.build
        self.lip_estimate = 0
        self.converge_every_training_step = converge_every_training_step

    def post_training_iteration_hook(self, **kwargs):
        if self.converge_every_training_step:
            with torch.no_grad():
                self.lipschitz_estimate(num_iter=-1)

    def post_training_hook(self, **kwargs):
        data_dict = dict(lipschitz_estimate=OrderedDict())
        with torch.no_grad():
            metrics = dict(lipschitz_estimate=OrderedDict())

            # update lipschitz estimates
            # run until convergence
            Kt = self.lipschitz_estimate(num_iter=1)
            diff = np.inf
            print('Number of lipschitz estimate iterations: 1')
            print('\tLipschitz estimate: {}'.format(Kt.item()))
            eps = 1.e-9
            num_iters = 2
            while diff > eps:
                Kt1 = self.lipschitz_estimate(num_iter=1)
                diff = (Kt1-Kt).abs()
                Kt = Kt1
                num_iters += 1

            metrics['lipschitz_estimate']['global'] = Kt1 #self.lipschitz_estimate(num_iter=1000)
            print('Number of lipschitz estimate iterations: {}'.format(num_iters))
            print('\tepsilon: {}'.format((Kt1-Kt).abs().item()))
            print('\tLipschitz estimate: {}'.format(Kt1.item()))

            for name, child in self.named_children():
                if isinstance(child, LipschitzLayerComputer):
                    lip_estimate = child.estimate(num_iter=0)
                    if type(lip_estimate) is tuple:
                        metrics['lipschitz_estimate'][name+'_main'] = lip_estimate[0]
                        metrics['lipschitz_estimate'][name+'_residual'] = lip_estimate[1]
                        data_dict['lipschitz_estimate'][name+'_main'] = lip_estimate[0]
                        data_dict['lipschitz_estimate'][name+'_residual'] = lip_estimate[1]
                    else:
                        metrics['lipschitz_estimate'][name] = lip_estimate
                        data_dict['lipschitz_estimate'][name] = lip_estimate

            metrics['weight_rank'] = OrderedDict()
            metrics['singular_value'] = OrderedDict()
            for name, module in self.named_modules():
                if hasattr(module, 'running_var'):
                    metrics['singular_value'][name+'.running_var'] = Histogram(module.running_var)
                if hasattr(module, 'weight') and module.weight is not None:
                    W = module.weight
                    W = W.view(W.shape[0], -1)
                    metrics['weight_rank'][name] = torch.matrix_rank(W)
                    _, s, _ = torch.svd(W, some=False, compute_uv=False)
                    metrics['singular_value'][name] = Histogram(s)

                if name == self.classifier_layer:
                    logit_norms = torch.norm(module.parent.weight, p=2, dim=1)
                    for logit_idx, logit_norm in enumerate(logit_norms):
                        assert logit_idx <= 99, 'string construction limited to two digits.'
                        data_dict['logit_norm'] = logit_norm

        if len(data_dict) > 0 and 'save_path' in kwargs:
            # assert 'save_path' in kwargs
            assert 'epoch' in kwargs
            torch.save(data_dict, os.path.join(kwargs['save_path'], 'epoch_{}_post.pth'.format(kwargs['epoch'])))

        return metrics

    def __setattr__(self, name, value):
        # Adding a LipschitzLayer should be possible by default.
        if not isinstance(value, LipschitzLayerComputer):
            # Otherwise, if its a regular module, try to wrap it
            if isinstance(value, nn.Module):
                if name == self.classifier_layer:
                    assert isinstance(value, nn.Linear), '{} equals classifier_layer. Should be of type nn.Linear, but is: {}'.format(name, type(value))
                    value = ClassifierLipschitzComputer(value, self.norm_p, use_softmax=self.use_softmax)
                else:
                    value = wrap_into_lipschitz_layer(value, self.norm_p, self.ignore_batchnorm)
                
                if not isinstance(value, LipschitzLayerContainer):
                    value.calibrate_outputs = self.calibrate_outputs
        super(LipschitzModel, self).__setattr__(name, value)

    def lipschitz_estimate(self, num_iter):
        if num_iter == 0 and self.lip_estimate != 0:
            return self.lip_estimate
        if self.lip_estimate == 0:
            num_iter = 1

        K = 1.0
        last_weight_module = None
        has_gloro_module = False
        # check for AddGloroAbsentLogit
        for module in self.modules():
            if isinstance(module, LipschitzLayerComputer):
                if hasattr(module.parent, 'weight') and module.parent.weight is not None:
                    last_weight_module = module
            if isinstance(module, layers.AddGloroAbsentLogit):
                if not module.detach_lipschitz_computer:
                    has_gloro_module = True

        # for name, module in self.named_modules():
        for name, child in self.named_children():
            if isinstance(child, LipschitzLayerComputer):
                if has_gloro_module and (id(child) == id(last_weight_module)):
                    # skip last layer when using Gloro. It's treated differently
                    continue
                Kchild = child.estimate(num_iter)
                # print(name, K_.device)
                if self.fix_last_layer_lip and (id(child) == id(last_weight_module)):
                    # detach K_, such that the last layer is not affected by any lipschitz constraints
                    Kchild = Kchild.detach()
                
                K = child.compose(Kchild, K)

        self.lip_estimate = K.detach()

        return K

    # def forward(self, x, y=None, **kwargs):
    #     return super(LipschitzModel, self).forward(x,y,**kwargs)


class LoroModel(LipschitzModel):
    def __init__(self, penultimate_layer, *args, **kwargs):
        # self.penultimate_layer = kwargs.pop('penultimate_layer')
        super(LoroModel, self).__init__(*args, **kwargs)
        self.penultimate_layer = penultimate_layer

    def forward(self, x, y=None, **kwargs):
        return_outputs = kwargs.get('return_outputs', False)

        if self.training:
            x.requires_grad = True
        outputs = self._forward_pass(x, y, **kwargs)

        metrics = {}
        for name, metric in self.metrics.items():
            kwargs = self._assemble_function_inputs(self.metric_argspecs, name, metric, outputs, x, y)
            results = metric(**kwargs)
            if results is not None:
                metrics[name] = results
        if len(metrics) > 0:
            outputs['metric'] = metrics


        # determine local Lipschitz constant
        if self.training:
            out_penultimate = outputs['layer_output'][self.penultimate_layer]
            dims = list(range(len(out_penultimate.shape)))
            out_penultimate = torch.norm(out_penultimate, p=self.norm_p, dim=dims[1:])
            gradients = torch.autograd.grad(
                out_penultimate, 
                inputs=x,
                grad_outputs=torch.ones_like(out_penultimate),
                create_graph=True,
                retain_graph=True
                )[0]
            outputs['_Klocal'] = torch.norm(gradients, p=self.norm_p, dim=[1,2,3])
        else:
            outputs['_Klocal'] = torch.ones(x.shape[0], device=x.device)

        losses = {}
        for name, loss_f in self.loss.items():
            kwargs = self._assemble_function_inputs(self.loss_argspecs, name, loss_f, outputs, x, y)

            losses[name] = loss_f(**kwargs)
        if len(losses) > 0:
            outputs['loss'] = losses
        
        if not return_outputs:
            del outputs['layer_output']

        del outputs['_Klocal']

        return outputs


class LipschitzConstrainedModel(LipschitzModel):
    def __init__(self, classifier_layer, p=2, betas=1.0, num_iter=1):
        super(LipschitzConstrainedModel, self).__init__(p=p)
        print(betas)
        assert type(betas) is int or type(betas) is float or isinstance(betas, dict) or isinstance(betas, Scheduler), 'Type of beta incorrect {}'.format(str(type(betas)))
        if type(betas) is int or type(betas) is float or isinstance(betas, Scheduler):
            self.betas = util.DefaultFallbackDict(fallback=betas)
        else:
            self.betas = betas
        self.num_iter = num_iter
        self.classifier_layer = classifier_layer

    # def post_training_hook(self, **kwargs):
    #     # update lipschitz estimates
    #     with torch.no_grad():
    #         metrics = dict(lipschitz_estimate=OrderedDict())

    #         metrics['lipschitz_estimate']['global'] = self.lipschitz_estimate(num_iter=1000)

    #         for name, child in self.named_children():
    #             if isinstance(child, LipschitzLayerComputer):
    #                 metrics['lipschitz_estimate'][name] = child.estimate(num_iter=0)

    #     return metrics

    def __setattr__(self, name, value):
        if isinstance(value, nn.Module):
            if name == self.classifier_layer:
                # raise NotImplementedError
                # value = LinearLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
                # value = ClassifierLipschitzComputer(value, self.norm_p, use_softmax=self.use_softmax)
                value = ClassifierLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            elif isinstance(value, nn.Conv2d):
                value = Conv2dLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            elif isinstance(value, nn.Linear):
                value = LinearLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            elif isinstance(value, nn.modules.batchnorm._BatchNorm):
                value = BatchNormLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            elif isinstance(value, layers.BasicResNetBlock):
                value = ResNetBlockLipschitzConstraint(value, self.norm_p, self.betas[name], self.num_iter)
            else:
                value = wrap_into_lipschitz_layer(value, self.norm_p)

        super(LipschitzConstrainedModel, self).__setattr__(name, value)


class LipschitzLayerComputer(nn.Module):
    def __init__(self, parent_module, p, method='full'):
        super(LipschitzLayerComputer, self).__init__()
        self.parent = parent_module
        self.method = method
        self.norm_p = p
        self.calibrate_outputs = False
        if method == 'full':
            self.register_buffer('input_shape', torch.Tensor())
            # self.register_buffer('power_iterate', torch.Tensor())
            if isinstance(self, LipschitzLayerContainer):
                pass
            elif isinstance(self.parent, nn.Conv2d):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.weight.shape[1], 32, 32))
            elif isinstance(self.parent, nn.Linear):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.weight.shape[1]))
            elif isinstance(self, ScalingLipschitzComputer):
                pass
            elif isinstance(self, AdaptiveAvgPool2dLipschitzComputer) or isinstance(self, AvgPool2dLipschitzComputer):
                pass
            elif isinstance(self, BNConv2dLipschitzComputer):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.conv.weight.shape[1], 32, 32))
            elif isinstance(self, PreActBNConv2dLipschitzComputer):
                self.register_buffer('power_iterate', torch.randn(1,self.parent.weight.shape[1], 32, 32))
            elif isinstance(self, BatchNormLipschitzComputer):
                if self.parent.affine:
                    self.register_buffer('power_iterate', torch.randn(1,self.parent.weight.shape[0]))
                else:
                    self.register_buffer('power_iterate', torch.ones(1))
            elif isinstance(self, GradBatchNorm2dLipschitzComputer):
                self.register_buffer('power_iterate', torch.ones(1))
            elif isinstance(self, ResNetBlockLipschitzComputer):
                pass
            else:
                raise RuntimeError('Don\'t know how to initialize dimensioniality of power iterate.')
        elif method == 'flattened':
            M, N = self.parent.weight.view(self.parent.weight.shape[0], -1).shape
            u = torch.randn(M)
            v = torch.randn(N)
            self.register_buffer('u', u/torch.norm(u,p=p))
            self.register_buffer('v', v/torch.norm(v,p=p))
            self.power_iterate = lambda: (self.u, self.v)
        else:
            raise RuntimeError('Undefined method {}'.format(method))
        self.register_buffer('lip_estimate', torch.ones(1).cuda() * np.nan)
        self.convergence_iterations = np.inf

    def check(self):
        # if self.method == 'full':
            # assert self.input_shape.numel() > 0, 'LipschitzLayerComputer has not been initialized yet. Run a training sample through the network first.'        
        pass

    def power_iteration(self, num_iter, W, running_power_iterate):
        if self.method == 'full':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def power_iteration_converge(self, W, running_power_iterate, eps=1.e-2, max_iter=10000):
        running_power_iterate = self.power_iteration(1, W, running_power_iterate)
        with torch.no_grad():
            sigma_t = self.spectral_value(W, running_power_iterate)
        diff = np.inf

        it = 1
        while diff > eps and it < max_iter:
            running_power_iterate = self.power_iteration(1, W, running_power_iterate)
            with torch.no_grad():
                sigma_t1 = self.spectral_value(W, running_power_iterate)
                diff = (sigma_t1 - sigma_t).abs()
                sigma_t = sigma_t1
            it += 1
        self.convergence_iterations = it
        # print('{: 4d}: {} :: '.format(it, self, running_power_iterate.mean().detach().item()))
        return running_power_iterate

    def spectral_value(self, W, running_power_iterate):
        """
        Computes largest spectral value of weight matrix W, 
        using right-singular vector estimate: running_power_iterate
        """
        if self.method == 'full':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def estimate(self, num_iter):
        """
        Estimates upper Lipschitz constant of module
        """
        raise NotImplementedError

    def estimate_inverse(self, num_iter):
        """
        Estimates lower Lipschitz constant of module, by doing power iterate on inv(W'W)
        """
        raise NotImplementedError

    def compose(self, Kself, Kparent):
        """
        Produces composition of Lipschitz constant according to module graph.
        By default, take the product, which is valid for functional compositions as f(g(x)).
        """
        return Kself * Kparent

    def _setup(self, x):
        if self.method == 'full':
            if self.input_shape.numel() == 0:
                self.input_shape = torch.Tensor(list(x.shape)).int().to(x.device)
                self.input_shape[0] = 1
                print('{} :: New Input shape: {}'.format(str(self), self.input_shape.cpu().numpy().tolist()))
                # print('before', id(self.power_iterate), 'iterate_device', self.power_iterate.device)
                self.power_iterate = torch.randn(*self.input_shape.int().tolist()).to(x.device)
                # print('after', id(self.power_iterate), 'iterate_device', self.power_iterate.device)
            # if self.input_shape.tolist()[1:] != list(x.shape)[1:]:
            #     raise RuntimeError('Mismatch in input shape. Expected shape {}, but received input of shape {}'.format(self.input_shape, x.shape))

    def forward(self, x):
        self._setup(x)
        out = self.parent(x)
        if self.calibrate_outputs:
            with torch.no_grad():
                K = self.estimate(num_iter=1)
                if len(out.shape) == 4:
                    batch_mean = torch.mean(out, dim=[0,2,3], keepdim=True)
                else:
                    batch_mean = torch.mean(out, dim=[0], keepdim=True)
            out = (out - batch_mean) / (K * 0.1)
            # out = out / K
        return out

    def __repr__(self):
        return 'LC{'+str(self.parent)+'}'


class LipschitzLayerContainer(LipschitzLayerComputer):
    pass



# ------------------------------------------------------------------
# Lipschitz estimation classes


class Conv2dLipschitzComputer(LipschitzLayerComputer):
    # @property
    # def weight(self):
    #     return self.parent.weight

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.conv2d(x, weight=W, stride=self.parent.stride, padding=self.parent.padding, dilation=self.parent.dilation)
            x_ = F.conv_transpose2d(xp, weight=W, stride=self.parent.stride, padding=self.parent.padding, dilation=self.parent.dilation)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.conv2d(x, weight=W, stride=self.parent.stride, padding=self.parent.padding, dilation=self.parent.dilation)
        if self.norm_p == 2:
            sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        else:
            sigma = torch.norm(Wx, p=self.norm_p) / torch.norm(x, p=self.norm_p)
        return sigma

    def estimate(self, num_iter):
        if self.norm_p == 1:
            # operator norm equals absolute column sum
            sigma = torch.max(self.parent.weight.abs().sum(dim=[0,2,3]))
        elif self.norm_p == 2:
            self.check()

            if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
                return self.lip_estimate
            elif num_iter == 0:
                num_iter = 1

            # print('power iterate max', self.power_iterate.max().item())
            if num_iter > 0:
                x = self.power_iteration(num_iter, self.parent.weight, self.power_iterate.clone(memory_format=torch.contiguous_format))
            else:
                x = self.power_iteration_converge(self.parent.weight, self.power_iterate.clone(memory_format=torch.contiguous_format))
            sigma = self.spectral_value(self.parent.weight, x)#.clone(memory_format=torch.contiguous_format))

            if self.training:
                # self.power_iterate = x.detach()
                with torch.no_grad():
                    torch.add(x.detach(), 0.0, out=self.power_iterate)
        elif np.isinf(self.norm_p):
            # operator norm equals absolute row sum
            # if torch.is_grad_enabled() and self.training:
            #     abs_weights = self.parent.weight.abs().sum(dim=[1,2,3])
            #     alpha = F.softmax(abs_weights, dim=0).detach()
            #     sigma = torch.sum(abs_weights*alpha)
            # else:
            sigma = torch.max(self.parent.weight.abs().sum(dim=[1,2,3]))

        if isinstance(self.parent, layers.InputNormPreservingConv2d) and self.parent.denominator is not None:
            sigma = sigma * (1.0 / self.parent.denominator)

        self.lip_estimate = sigma.detach()

        return sigma


class LinearLipschitzComputer(LipschitzLayerComputer):
    # @property
    # def weight(self):
    #     return self.parent.weight

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.linear(x, weight=W)
            x_ = F.linear(xp, weight=W.transpose(1,0))
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x
    
    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.linear(x, weight=W)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter):
        if self.norm_p == 1:
            # operator norm equals absolute column sum
            sigma = torch.max(self.parent.weight.abs().sum(dim=0))
        elif self.norm_p == 2:
            self.check()

            if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
                return self.lip_estimate
            elif num_iter == 0:
                num_iter = 1

            if num_iter > 0:
                x = self.power_iteration(num_iter, self.parent.weight, self.power_iterate.clone(memory_format=torch.contiguous_format))
            else:
                x = self.power_iteration_converge(self.parent.weight, self.power_iterate.clone(memory_format=torch.contiguous_format))
            sigma = self.spectral_value(self.parent.weight, x) #.clone(memory_format=torch.contiguous_format))

            if self.training:
                # self.power_iterate = x.detach()
                with torch.no_grad():
                    torch.add(x.detach(), 0.0, out=self.power_iterate)
        elif np.isinf(self.norm_p):
            # operator norm equals absolute row sum
            # if torch.is_grad_enabled() and self.training:
            #     abs_weights = self.parent.weight.abs().sum(dim=1)
            #     alpha = F.softmax(abs_weights, dim=0).detach()
            #     sigma = torch.sum(abs_weights*alpha)
            # else:
            sigma = torch.max(self.parent.weight.abs().sum(dim=1))

        if isinstance(self.parent, layers.InputNormPreservingLinear) and self.parent.denominator is not None:
            sigma = sigma * (1.0 / self.parent.denominator)

        self.lip_estimate = sigma.detach()
        return sigma


class ClassifierLipschitzComputer(LinearLipschitzComputer):
    def __init__(self, *args, **kwargs):
        self.use_softmax = kwargs.pop('use_softmax', False)
        super(ClassifierLipschitzComputer, self).__init__(*args, **kwargs)

    def estimate(self, num_iter):
        # kW \in [10 x N]
        W = self.parent.weight
        K_ij = torch.cdist(W, W, p=self.norm_p)        
        # K_ij \in [10 x 10]

        if self.use_softmax and torch.is_grad_enabled() and self.training:
            inds = torch.triu_indices(*K_ij.shape, offset=1)
            K_ij = K_ij[inds[0,:], inds[1,:]]  # select triu entries
            sigma = torch.sum(K_ij * F.softmax(K_ij, dim=0).detach())
            self.lip_estimate = torch.max(K_ij).detach()
        else:
            sigma = torch.max(K_ij)
            self.lip_estimate = sigma.detach()
        
        return sigma

    def __repr__(self):
        return 'Classifier_'+super(ClassifierLipschitzComputer, self).__repr__()


class ResNetBlockLipschitzComputer(LipschitzLayerContainer):
    def __init__(self, *args, ignore_batchnorm=False, **kwargs):
        super(ResNetBlockLipschitzComputer, self).__init__(*args, **kwargs)
        for name, child in list(self.parent.named_children()):
            wrapped_child = wrap_into_lipschitz_layer(child, self.norm_p, ignore_batchnorm)
            self.parent.__setattr__(name, wrapped_child)

    def estimate(self, num_iter):
        Kmain = torch.ones_like(self.lip_estimate)
        Kresidual = torch.ones_like(self.lip_estimate)
        for name, child in self.parent.named_children():
            if not isinstance(child, LipschitzLayerComputer):
                continue

            if name == 'downsample':
                Kmain = child.estimate(num_iter)
            else:
                Kresidual = Kresidual * child.estimate(num_iter)

        return (Kmain, Kresidual)

    def compose(self, Kself, Kparent):
        assert type(Kself) is tuple and len(Kself) == 2
        Kmain, Kresidual = Kself

        if isinstance(self.parent, layers.PreActBasicResNetBlock):
            # preactivated blocks apply a normalization layer before downsampling
            if self.parent.has_norm_layers:
                Kbn1 = self.parent.bn1.estimate(num_iter=0)
                Kparent = self.parent.bn1.compose(Kbn1, Kparent)
        
        Kresidual = self.parent.conv1.compose(Kresidual, Kparent)
        if self.parent.downsample is not None:
            Kparent = self.parent.downsample.compose(Kmain, Kparent)
        return Kresidual + Kparent


class SequentialLipschitzComputer(LipschitzLayerContainer):
    def __init__(self, *args, ignore_batchnorm=False, **kwargs):
        super(SequentialLipschitzComputer, self).__init__(*args, **kwargs)
        for i in range(len(self.parent)):
            child = self.parent[i]
            wrapped_child = wrap_into_lipschitz_layer(child, self.norm_p, ignore_batchnorm)
            self.parent[i] = wrapped_child

    def estimate(self, num_iter):
        K = torch.ones_like(self.lip_estimate)
        for i in range(len(self.parent)):
            child = self.parent[i]
            if isinstance(child, LipschitzLayerComputer):
                Kchild = child.estimate(num_iter)
                K = child.compose(Kchild, K)
        return K


class ScalingLipschitzComputer(LipschitzLayerComputer):
    def _setup(self, x):
        if isinstance(self.parent, layers.MaxNormTo1):
            if self.parent.denominator is None:
                self.lip_estimate = torch.ones_like(self.lip_estimate)
            else:
                self.lip_estimate = 1./self.parent.denominator.detach()
        else:
            raise NotImplementedError
            # x_in, x_out = x

    def forward(self, x_in):
        x_out = self.parent(x_in)
        self._setup((x_in, x_out))
        return x_out

    def estimate(self, num_iter, **kwargs):
        return self.lip_estimate


class AvgPool2dLipschitzComputer(LipschitzLayerComputer):
    def __init__(self, *args, **kwargs):
        super(AvgPool2dLipschitzComputer, self).__init__(*args, **kwargs)
        self._is_setup = False        

    def _setup(self, x):
        if not self._is_setup:
            self._is_setup = True
            with torch.no_grad():
                Weight = torch.eye(x.shape[1])[:,:,None, None] * (
                    torch.ones(self.parent.kernel_size,self.parent.kernel_size)[None,None,:,:]) / (self.parent.kernel_size*self.parent.kernel_size)
                if np.isinf(self.norm_p):
                    if self.norm_p == 1:
                        # operator norm equals absolute column sum
                        factor = torch.max(Weight.abs().sum(dim=[0,2,3])).item()
                    else:
                        # operator norm equals absolute row sum
                        factor = torch.max(Weight.abs().sum(dim=[1,2,3])).item()
                else:
                    Weight = Weight.to(x.device)
                    x = torch.randn(1,x.shape[1], 32, 32).to(x.device)
                    x = self.power_iteration_converge(Weight, x)
                    # x = self.power_iteration(1000, Weight, x)
                    factor = self.spectral_value(Weight, x)
                self.lip_estimate = torch.Tensor([factor])
        self.lip_estimate = self.lip_estimate.to(x.device)

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.conv2d(x, weight=W, stride=self.parent.stride, padding=self.parent.padding)
            x_ = F.conv_transpose2d(xp, weight=W, stride=self.parent.stride, padding=self.parent.padding)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.conv2d(x, weight=W, stride=self.parent.stride, padding=self.parent.padding)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter, **kwargs):
        return self.lip_estimate

    def forward(self, x):
        self._setup(x)
        return self.parent(x)


class AdaptiveAvgPool2dLipschitzComputer(LipschitzLayerComputer):
    def __init__(self, *args, **kwargs):
        super(AdaptiveAvgPool2dLipschitzComputer, self).__init__(*args, **kwargs)
        # self.lip_estimate = torch.ones(1)
        self.lip_estimate_dict = dict()

    def forward(self, x_in):
        H, W = x_in.shape[-2:]

        # if self.training:
        if (H,W) not in self.lip_estimate_dict:
            with torch.no_grad():
                Weight = torch.eye(x_in.shape[1])[:,:,None, None] * (
                    torch.ones(H,W)[None,None,:,:]) / (H*W)
                if np.isinf(self.norm_p):
                    if self.norm_p == 1:
                        # operator norm equals absolute column sum
                        factor = torch.max(Weight.abs().sum(dim=[0,2,3])).item()
                    else:
                        # operator norm equals absolute row sum
                        factor = torch.max(Weight.abs().sum(dim=[1,2,3])).item()
                else:
                    Weight = Weight.to(x_in.device)
                    x = torch.randn(1,x_in.shape[1],H,W).to(x_in.device)
                    x = self.power_iteration_converge(Weight, x)
                    # x = self.power_iteration(1000, Weight, x)
                    factor = self.spectral_value(Weight, x)

            self.lip_estimate_dict[(H,W)] = torch.Tensor([factor]).to(x_in.device)

        self.lip_estimate = self.lip_estimate_dict[(H,W)]

        return self.parent(x_in)

    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.conv2d(x, weight=W, stride=1, padding=0)
            x_ = F.conv_transpose2d(xp, weight=W, stride=1, padding=0)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.conv2d(x, weight=W, stride=1, padding=0)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter, **kwargs):
        return self.lip_estimate


class PreActBNConv2dLipschitzComputer(Conv2dLipschitzComputer):
    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        bn = self.parent.bn
        conv = self.parent.conv

        for i in range(num_iter):
            # xp = x / torch.sqrt(bn.running_var[None, :, None, None] + bn.eps)
            # if bn.affine:
            #     xp = xp * bn.weight[None, :, None, None]
            xp = x
            xp = F.conv2d(xp, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
            x_ = F.conv_transpose2d(xp, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
            # if bn.affine:
            #     x_ = x_ * bn.weight[None, :, None, None]
            # x_ = x_ / torch.sqrt(bn.running_var[None, :, None, None] + bn.eps)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        bn = self.parent.bn
        conv = self.parent.conv

        # xp = x / torch.sqrt(bn.running_var[None, :, None, None] + bn.eps)
        # if bn.affine:
        #     xp = xp * bn.weight[None, :, None, None]
        Wx = F.conv2d(x, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))

        sigma = sigma * torch.max(bn.weight/torch.sqrt(bn.running_var+bn.eps))

        return sigma



class BNConv2dLipschitzComputer(Conv2dLipschitzComputer):
    def power_iteration(self, num_iter, W, running_power_iterate):
        x = running_power_iterate
        for i in range(num_iter):
            xp = F.conv2d(x, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
            xp = xp / torch.sqrt(self.parent.bn.running_var[None, :, None, None] + self.parent.bn.eps)
            if self.parent.bn.affine:
                xp = xp * self.parent.bn.weight[None, :, None, None]
            x_ = F.conv_transpose2d(xp, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
            x = x_ / torch.norm(x_, p=self.norm_p)
        return x

    def spectral_value(self, W, running_power_iterate):
        x = running_power_iterate
        Wx = F.conv2d(x, weight=W, stride=self.parent.conv.stride, padding=self.parent.conv.padding, dilation=self.parent.conv.dilation)
        Wx = Wx / torch.sqrt(self.parent.bn.running_var[None, :, None, None] + self.parent.bn.eps)
        if self.parent.bn.affine:
            Wx = Wx * self.parent.bn.weight[None, :, None, None]
        sigma = torch.sqrt(torch.sum(Wx**2.) / (torch.sum(x**2.) + 1.e-9))
        return sigma

    def estimate(self, num_iter):
        raise RuntimeError('Deprecated class. Use regular conv and BatchNorm')
        self.check()

        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        elif num_iter == 0:
            num_iter = 1

        # print('power iterate max', self.power_iterate.max().item())
        x = self.power_iteration(num_iter, self.parent.conv.weight, self.power_iterate.clone(memory_format=torch.contiguous_format))
        sigma = self.spectral_value(self.parent.conv.weight, x)#.clone(memory_format=torch.contiguous_format))

        if self.training:
            # self.power_iterate = x.detach()
            with torch.no_grad():
                torch.add(x.detach(), 0.0, out=self.power_iterate)

        # if isinstance(self.parent, layers.InputNormPreservingConv2d) and self.parent.denominator is not None:
        #     sigma = sigma * (1.0 / self.parent.denominator)

        self.lip_estimate = sigma.detach()

        return sigma


class BatchNormLipschitzComputer(LipschitzLayerComputer):
    def __init__(self, *args, ignore_batchnorm, **kwargs):
        super(BatchNormLipschitzComputer, self).__init__(*args, **kwargs)
        self.ignore_batchnorm = ignore_batchnorm
        # self.bn_blend = PolynomialScheduler(1.0, 0.0, 2.0, name='bn_blend')

    def check(self):
        pass

    def estimate(self, num_iter):
        # self.check()

        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        else:
            num_iter = 1


        if isinstance(self.parent, layers.ClippingBatchNorm2d) or isinstance(self.parent, layers.NoStdBatchNorm2d):
            eps = self.parent.eps
        else:    
            eps = self.parent.eps

        if self.parent.affine:            
            W = self.parent.weight / torch.sqrt(self.parent.running_var + eps)
        else:
            W = 1.0 / torch.sqrt(self.parent.running_var + eps)

        sigma = torch.max(W.abs())

        # print(type(self.parent), self.parent.running_var.mean().item(), sigma.detach().item())

        self.lip_estimate = sigma.detach()

        if self.ignore_batchnorm:
            return self.lip_estimate
        else:
            return sigma


class GroupNormLipschitzComputer(LipschitzLayerComputer):
    def __init__(self, *args, ignore_batchnorm, **kwargs):
        super(GroupNormLipschitzComputer, self).__init__(*args, **kwargs)
        self.ignore_batchnorm = ignore_batchnorm
        self.sigma = None

    def estimate(self, num_iter):
        if num_iter > 0:
            return self.sigma
        else:
            return self.lip_estimate

    def forward(self, x):
        assert x.shape[1]%self.parent.num_groups == 0
        G = self.parent.num_groups
        N, K, H, W = x.shape
        with torch.no_grad():
            group_var = x.view(N, G, -1).var(dim=2, unbiased=False)
            group_var = group_var[:, None].repeat(1, K//G).view(N, K)

        eps = self.parent.eps
        if self.parent.affine:
            W = self.parent.weight.unsqueeze(0) / torch.sqrt(group_var + eps)
        else:
            W = torch.sqrt(group_var + eps)

        self.sigma = torch.max(W.abs())
        self.lip_estimate = self.sigma.detach()

        x_out = self.parent(x)
        return x_out

            

class BlendBatchNormLipschitzComputer(BatchNormLipschitzComputer):
    def estimate(self, num_iter):
        sigma = super(BlendBatchNormLipschitzComputer, self).estimate(num_iter)
        sigma = self.parent.bn_blend * sigma + (1.0 - self.parent.bn_blend)
        self.lip_estimate = sigma.detach()
        return sigma


class GradBatchNorm2dLipschitzComputer(LipschitzLayerComputer):
    def _setup(self, x):
        pass

    def estimate(self, num_iter):
        return self.power_iterate.detach()


# ------------------------------------------------------------------
# Lipschitz constraining classes

class Conv2dLipschitzConstraint(Conv2dLipschitzComputer, ScheduledModule):
    def __init__(self, parent_module, norm_p, beta, num_iter):
        super(Conv2dLipschitzConstraint, self).__init__(parent_module, norm_p)

        # self.beta = beta
        self.beta = beta #nn.Parameter(torch.Tensor([beta]))
        self.num_iter = num_iter

        weight = self.parent.weight
        delattr(self.parent, 'weight')
        self.parent.register_parameter('weight_orig', weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(self.parent, 'weight', weight.data)

        self.register_buffer('normed_power_iterate', torch.Tensor())

        M, N = self.parent.weight.view(self.parent.weight.shape[0], -1).shape
        u = torch.randn(M)
        v = torch.randn(N)
        self.register_buffer('power_iterate_u', u/torch.norm(u,p=2))
        self.register_buffer('power_iterate_v', v/torch.norm(v,p=2))

    def _setup(self, x):
        need_setup = False
        if self.input_shape.numel() == 0:
            need_setup = True
        super(Conv2dLipschitzConstraint, self)._setup(x)
        if need_setup:
            self.normed_power_iterate = torch.randn(*self.input_shape.int().tolist()).to(x.device)

    def estimate(self, num_iter):
        self.check()

        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        elif num_iter == 0:
            num_iter = 1

        x = self.power_iteration(num_iter, self.parent.weight, self.normed_power_iterate)

        if self.training:
            torch.add(x.detach(), 0.0, out=self.normed_power_iterate)

        sigma = self.spectral_value(self.parent.weight, x.clone(memory_format=torch.contiguous_format))

        if isinstance(self.parent, layers.InputNormPreservingConv2d) and self.parent.denominator is not None:
            sigma = sigma * (1.0 / self.parent.denominator)
        # sigma = sigma * self.beta

        self.lip_estimate = sigma.detach()

        return sigma

    def forward(self, x):
        self._setup(x)

        with torch.no_grad():
            if self.training:
                iterate = self.power_iteration(self.num_iter, self.parent.weight_orig, self.power_iterate)
                torch.add(iterate, 0, out=self.power_iterate)
                iterate = iterate.clone(memory_format=torch.contiguous_format)
            else:
                iterate = self.power_iterate
        sigma = self.spectral_value(self.parent.weight_orig, iterate)

        # # SN-GAN Version, assuming flattened weight matrix
        # weight_mat = self.parent.weight_orig.view(self.parent.weight_orig.shape[0], -1)
        # u, v = self.power_iterate_u, self.power_iterate_v
        # if self.training:
        #     with torch.no_grad():
        #         for _ in range(self.num_iter):
        #             v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=1.e-9, out=v)
        #             u = normalize(torch.mv(weight_mat, v), dim=0, eps=1.e-9, out=u)
        #         u = u.clone(memory_format=torch.contiguous_format)
        #         v = v.clone(memory_format=torch.contiguous_format)
        #         # self.power_iterate_u = u
        #         # self.power_iterate_v = v
        # sigma = torch.dot(u, torch.mv(weight_mat, v))

        # if type(self.beta) is float:
        #     beta = self.beta
        # else:
        #     beta = self.beta.val
        W = self.beta * (self.parent.weight_orig / sigma)
        setattr(self.parent, 'weight', W)

        return self.parent(x)

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)


class LinearLipschitzConstraint(LinearLipschitzComputer, ScheduledModule):
    def __init__(self, parent_module, norm_p, beta, num_iter):
        super(LinearLipschitzConstraint, self).__init__(parent_module, norm_p)

        self.beta = beta #nn.Parameter(torch.Tensor([beta]))
        self.num_iter = num_iter
        self.norm_p = norm_p

        weight = self.parent.weight
        delattr(self.parent, 'weight')
        self.parent.register_parameter('weight_orig', weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(self.parent, 'weight', weight.data)

        self.register_buffer('normed_power_iterate', torch.Tensor())

    def _setup(self, x):
        need_setup = False
        if self.input_shape.numel() == 0:
            need_setup = True
        super(LinearLipschitzConstraint, self)._setup(x)
        if need_setup:
            self.normed_power_iterate = torch.randn(*self.input_shape.int().tolist()).to(x.device)

    def estimate(self, num_iter):
        self.check()

        if num_iter == 0 and (not torch.isnan(self.lip_estimate)):
            return self.lip_estimate
        elif num_iter == 0:
            num_iter = 1

        x = self.power_iteration(num_iter, self.parent.weight, self.normed_power_iterate)

        if self.training:
            torch.add(x.detach(), 0.0, out=self.normed_power_iterate)

        sigma = self.spectral_value(self.parent.weight, x.clone(memory_format=torch.contiguous_format))

        if isinstance(self.parent, layers.InputNormPreservingLinear) and self.parent.denominator is not None:
            sigma = sigma * (1.0 / self.parent.denominator)
        # sigma = sigma * self.beta

        self.lip_estimate = sigma.detach()

        return sigma

    def forward(self, x):
        self._setup(x)

        with torch.no_grad():
            if self.training:
                iterate = self.power_iteration(self.num_iter, self.parent.weight_orig, self.power_iterate)
                torch.add(iterate, 0, out=self.power_iterate)
                iterate = iterate.clone(memory_format=torch.contiguous_format)
            else:
                iterate = self.power_iterate
            sigma = self.spectral_value(self.parent.weight_orig, iterate)

        # if type(self.beta) is float:
        #     beta = self.beta
        # else:
        #     beta = self.beta.val

        W = self.beta * (self.parent.weight_orig / sigma)
        setattr(self.parent, 'weight', W)

        return self.parent(x)

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)


class ClassifierLipschitzConstraint(LinearLipschitzConstraint):

    def estimate(self, num_iter):
        sigma = torch.max(torch.norm(self.parent.weight, p=self.norm_p, dim=1))
        self.lip_estimate = sigma.detach()
        return sigma

    def forward(self, x):
        with torch.no_grad():
            sigma = torch.max(torch.norm(self.parent.weight_orig, p=self.norm_p, dim=1))

        W = self.beta * (self.parent.weight_orig / sigma)
        setattr(self.parent, 'weight', W)

        return self.parent(x)

    def __repr__(self):
        return 'Classifier_'+super(ClassifierLipschitzConstraint, self).__repr__()


class BatchNormLipschitzConstraint(BatchNormLipschitzComputer, ScheduledModule):
    def __init__(self, parent_module, norm_p, beta, num_iter=1, **kwargs):
        super(BatchNormLipschitzConstraint, self).__init__(parent_module, norm_p, ignore_batchnorm=True, **kwargs)
        self.num_iter = num_iter
        self.beta = beta #nn.Parameter(torch.Tensor([beta]))
        self.norm_p = norm_p

        weight = self.parent.weight
        delattr(self.parent, 'weight')
        self.parent.register_parameter('weight_orig', weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(self.parent, 'weight', weight.data)

    def forward(self, x):
        with torch.no_grad():
            if self.parent.affine:            
                W = self.parent.weight_orig / torch.sqrt(self.parent.running_var + self.parent.eps)
            else:
                W = 1.0 / torch.sqrt(self.parent.running_var + self.parent.eps)
            sigma = torch.max(W.abs())

        W = self.beta * (self.parent.weight_orig / sigma)
        setattr(self.parent, 'weight', W)

        return self.parent(x)

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)

class ResNetBlockLipschitzConstraint(ResNetBlockLipschitzComputer):
    def __init__(self, parent, norm_p, block_beta, num_iter, **kwargs):
        self.beta = block_beta
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d):
                wrapped_child = Conv2dLipschitzConstraint(child, norm_p, block_beta, num_iter)
            elif isinstance(child, nn.Linear):
                wrapped_child = LinearLipschitzConstraint(child, norm_p, block_beta, num_iter)
            elif isinstance(child, nn.modules.batchnorm._BatchNorm):
                wrapped_child = BatchNormLipschitzConstraint(child, norm_p, block_beta, num_iter)
            else:
                wrapped_child = wrap_into_lipschitz_layer(child, norm_p)
            
            parent.__setattr__(name, wrapped_child)

        super(ResNetBlockLipschitzConstraint, self).__init__(parent, p=norm_p, **kwargs)

    def __repr__(self):
        return 'LC={}{{{}}}'.format(self.beta, self.parent)
