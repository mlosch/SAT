import os
import sys

import numpy as np

import torch
import torch.nn as nn

from model import wide_resnet
import lib.scheduler as scheduler
import lib.util as util


def layerclass_from_string(name):
    if name.startswith('nn.'):
        layerclass = nn.__dict__[name.replace('nn.','')]
    else:
        layerclass = sys.modules[__name__].__dict__[name]
    return layerclass


class WideResNet(wide_resnet.WideResNet):
    def __init__(self, num_classes=10, depth=70, width=16):
        super(WideResNet, self).__init__(num_classes, depth, width, activation_fn=wide_resnet.Swish)


class AAResNet50(nn.Module):
    def __init__(self, **kwargs):
        super(AAResNet50, self).__init__()
        import antialiased_cnns
        self.model = antialiased_cnns.resnet50(**kwargs)

    def forward(self, x):
        return self.model(x)


class BasicResNetBlock(nn.Module):
    """
    Adaptation of torchvision.resnet.BasicBlock version
    """
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        nonlinearity = 'nn.ReLU',
        norm_layer = 'nn.BatchNorm2d'
    ):
        super(BasicResNetBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if norm_layer is not None:
            if isinstance(norm_layer, dict):
                norm_type = norm_layer.pop('type')
                norm_kwargs = norm_layer
                self.norm_layer_factory = lambda *args: layerclass_from_string(norm_type)(*args, **norm_kwargs)
            else:
                self.norm_layer_factory = layerclass_from_string(norm_layer)
            self.bn1 = self.norm_layer_factory(planes)
            self.bn2 = self.norm_layer_factory(planes)
            self.norm_is_affine = self.bn1.affine
            self.has_norm_layers = True
        else:
            self.has_norm_layers = False
            self.norm_is_affine = False

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=(self.norm_is_affine==False))

        if nonlinearity == 'nn.ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif nonlinearity == 'MaxMin':
            self.relu = MaxMin(planes//2)
        elif nonlinearity == 'nn.Sigmoid':
            self.relu = nn.Sigmoid()
        elif nonlinearity == 'nn.Tanh':
            self.relu = nn.Tanh()
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=(self.norm_is_affine==False))

        if inplanes != planes or stride != 1:
            downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=(self.norm_is_affine==False))
            if self.has_norm_layers:
                downsample = nn.Sequential(
                    downsample,
                    self.norm_layer_factory(planes)
                    )
            self.downsample = downsample
        else:
            self.downsample = None
        self.stride = stride


    def forward(self, x):
        identity = x

        residual = self.conv1(x)
        if self.has_norm_layers:
            residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        if self.has_norm_layers:
            residual = self.bn2(residual)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(residual + identity)

        return out


class ResNetBottleneck(nn.Module):
    """
    Adaptation of torchvision.resnet.BasicBlock version
    """
    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        base_width = 64,
        nonlinearity = 'nn.ReLU',
        norm_layer = 'nn.BatchNorm2d'
    ):
        super(ResNetBottleneck, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        expansion = 4
        width = max(base_width, planes // expansion)
        # width = int(planes * (base_width / 64.0)) # * groups
        if norm_layer is not None:
            if isinstance(norm_layer, dict):
                norm_type = norm_layer.pop('type')
                norm_kwargs = norm_layer
                self.norm_layer_factory = lambda *args: layerclass_from_string(norm_type)(*args, **norm_kwargs)
            else:
                self.norm_layer_factory = layerclass_from_string(norm_layer)
            self.bn1 = self.norm_layer_factory(width)
            self.bn2 = self.norm_layer_factory(width)
            self.bn3 = self.norm_layer_factory(planes)
            self.norm_is_affine = self.bn1.affine
            self.has_norm_layers = True
        else:
            self.has_norm_layers = False
            self.norm_is_affine = False

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, padding=0, bias=(self.norm_is_affine==False))
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=(self.norm_is_affine==False))
        self.conv3 = nn.Conv2d(width, planes, kernel_size=1, stride=1, padding=0, bias=(self.norm_is_affine==False))

        if nonlinearity == 'nn.ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif nonlinearity == 'MaxMin':
            self.relu = MaxMin(planes//2)
        elif nonlinearity == 'nn.Sigmoid':
            self.relu = nn.Sigmoid()
        elif nonlinearity == 'nn.Tanh':
            self.relu = nn.Tanh()
        else:
            raise NotImplementedError

        if inplanes != planes or stride != 1:
            downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=(self.norm_is_affine==False))
            if self.has_norm_layers:
                downsample = nn.Sequential(
                    downsample,
                    self.norm_layer_factory(planes)
                    )
            self.downsample = downsample
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PreActBasicResNetBlock(BasicResNetBlock):

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        nonlinearity = 'nn.ReLU',
        norm_layer = 'nn.BatchNorm2d'
    ):
        super(PreActBasicResNetBlock, self).__init__(inplanes, planes, stride, nonlinearity, norm_layer)
        if self.has_norm_layers:
            self.bn1 = self.norm_layer_factory(inplanes)
        if nonlinearity == 'MaxMin':
            self.relu1 = MaxMin(inplanes//2)
            self.relu2 = MaxMin(planes//2)
        else:
            self.relu1 = self.relu2 = self.relu
            self.relu = None
        self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=(self.norm_is_affine==False))

    def forward(self, x):
        identity = x

        if self.has_norm_layers:
            x = self.bn1(x)
        x = self.relu1(x)
        residual = self.conv1(x)

        if self.has_norm_layers:
            residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = residual + identity

        return out

class PreActBNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity='nn.ReLU', *args, **kwargs):
        super(PreActBNConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs)
        self.bn = nn.BatchNorm2d(in_channels)
        if nonlinearity == 'nn.ReLU':
            kwargs = {'inplace': True}
            nonlin_type = nonlinearity
        elif isinstance(nonlinearity, dict):
            nonlin_type = nonlinearity.pop('type')
            kwargs = nonlinearity
        nonlin_class = layerclass_from_string(nonlin_type)
        self.relu = nonlin_class(**kwargs)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        # x = self.conv(self.bn(x))
        return x

    @property
    def weight(self):
        """
        For Lipschitz estimates only!
        """
        return self.conv.weight
        # if self.bn.affine:
        #     return self.conv.weight * (self.bn.weight[None, :, None, None] / torch.sqrt(self.bn.running_var[None, :, None, None] + self.bn.eps))
        # else:
        #     return self.conv.weight * (1.0 / torch.sqrt(self.bn.running_var[None, :, None, None] + self.bn.eps))

    @property
    def stride(self):
        return self.conv.stride

    @property
    def padding(self):
        return self.conv.padding

    @property
    def dilation(self):
        return self.conv.dilation


class BlendBatchNorm2d(nn.BatchNorm2d, scheduler.ScheduledModule):
    def __init__(self, *args, bn_blend=1.0, **kwargs):
        super(BlendBatchNorm2d, self).__init__(*args, **kwargs)
        self.bn_blend = bn_blend

    def forward(self, x):
        if self.bn_blend > 0:
            x_bn = super(BlendBatchNorm2d, self).forward(x)
        else:
            x_bn = 0
        return self.bn_blend * x_bn + (1.0-self.bn_blend) * x


class RunningBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        kwargs['track_running_stats'] = True
        # kwargs['momentum'] = kwargs.get('momentum', 0.9)  # default is 0.1 in BatchNorm2d
        super(RunningBatchNorm2d, self).__init__(*args, **kwargs)


    def forward(self, input):
        """Copied from Pytorch v1.6.0
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            if self.num_batches_tracked == 0:
                exponential_average_factor = 1.0
                self.num_batches_tracked = self.num_batches_tracked + 1
            else:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

            # calculate mean and var and update stats
            with torch.no_grad():
                N, K, H, W = input.shape
                Nvalues = N*H*W
                batch_mean = torch.sum(input, dim=[0,2,3]) / Nvalues
                batch_var = torch.sum(((input - batch_mean[None, :, None, None])**2), dim=[0,2,3]) / Nvalues

                # update running stats:
                torch.add(self.running_mean * (1.0 - exponential_average_factor), batch_mean * exponential_average_factor, out=self.running_mean)
                torch.add(self.running_var * (1.0 - exponential_average_factor), batch_var * exponential_average_factor, out=self.running_var)

        """Always set to validation mode, such that buffers are used for normalization.
        The buffer update is performed within this forward pass
        """
        return nn.functional.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight, self.bias, False, exponential_average_factor, self.eps)


class NoStdBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        # kwargs['affine'] = False
        super(NoStdBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        """Copied from Pytorch v1.6.0
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

            # calculate mean and var and update stats
            N, K, H, W = input.shape
            Nvalues = N*H*W
            
            batch_mean = input.mean(dim=[0, 2, 3], keepdim=False)

            with torch.no_grad():
                # update running stats:
                torch.add(self.running_mean * (1.0 - exponential_average_factor), batch_mean * exponential_average_factor, out=self.running_mean)

            mean = batch_mean[None, :, None, None]
        else:
            mean = self.running_mean[None, :, None, None]

        if self.affine:
            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]
            return (input - mean) * weight + bias
        else:
            return (input - mean)


class ClippingBatchNorm2d(nn.BatchNorm2d, scheduler.ScheduledModule):
    def __init__(self, *args, **kwargs):
        super(ClippingBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        # return super(ClippingBatchNorm2d, self).forward(input)
        """Copied from Pytorch v1.6.0
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

            # calculate mean and var and update stats
            N, K, H, W = input.shape
            Nvalues = N*H*W

            batch_mean = input.mean(dim=[0, 2, 3], keepdim=False)
            batch_var = ((input - batch_mean[None, :, None, None]) ** 2).mean(dim=[0, 2, 3], keepdim=False)
            batch_var.clamp_(min=self.eps)

            with torch.no_grad():
                # update running stats:
                torch.add(self.running_mean * (1.0 - exponential_average_factor), batch_mean * exponential_average_factor, out=self.running_mean)
                torch.add(self.running_var * (1.0 - exponential_average_factor), batch_var * exponential_average_factor, out=self.running_var)

            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]
            mean = batch_mean[None, :, None, None]
            invstd = (1./torch.sqrt(batch_var))[None, :, None, None]

            return ((input - mean)*invstd) * weight + bias
        else:
            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]
            mean = self.running_mean[None, :, None, None]
            invstd = (1./torch.sqrt(self.running_var))[None, :, None, None]
            # print(invstd.mean().detach().item())

            return ((input - mean)*invstd) * weight + bias


class DroppingBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(DroppingBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        """Copied from Pytorch v1.6.0
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

            # calculate mean and var and update stats
            N, K, H, W = input.shape
            Nvalues = N*H*W
            batch_mean = input.mean(dim=[0, 2, 3], keepdim=False)
            batch_var = ((input - batch_mean[None, :, None, None]) ** 2).mean(dim=[0, 2, 3], keepdim=False)
            drop_mask = batch_var < self.eps

            with torch.no_grad():
                # update running stats:
                torch.add(self.running_mean * (1.0 - exponential_average_factor), batch_mean * exponential_average_factor, out=self.running_mean)
                torch.add(self.running_var * (1.0 - exponential_average_factor), batch_var * exponential_average_factor, out=self.running_var)

            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]
            mean = batch_mean[None, :, None, None]
            invstd = (1./torch.sqrt(batch_var))[None, :, None, None]

            result = ((input - mean)*invstd) * weight + bias
        else:
            weight = self.weight[None, :, None, None]
            bias = self.bias[None, :, None, None]
            mean = self.running_mean[None, :, None, None]
            drop_mask = self.running_var < self.eps
            invstd = (1./torch.sqrt(self.running_var))[None, :, None, None]
            # print(invstd.mean().detach().item())

            result = ((input - mean)*invstd) * weight + bias

        # replace dropouts with non-normalized values
        drop_mask = drop_mask[None, :, None, None].expand_as(input)
        result[drop_mask] = input[drop_mask]
        return result


class ScheduledBatchNorm2d(nn.BatchNorm2d, scheduler.ScheduledModule):
    pass


class GradBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1.e-5, momentum=0.1):
        super(GradBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = False
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def output_grad_hook(self, grad):
        if self.training:
            N, K, H, W = grad.shape
            return grad.view(N,K,-1).renorm(dim=2, p=2, maxnorm=1).view(N,K,H,W)
            # N, K, H, W = grad.shape
            # Nvalues = N*H*W
            # mean = grad.mean(dim=[0, 2, 3], keepdim=False)
            # var = ((grad - mean[None, :, None, None]) ** 2).mean(dim=[0, 2, 3], keepdim=False)

            # with torch.no_grad():
            #     # update running stats:
            #     torch.add(self.running_mean * (1.0 - self.momentum), mean * self.momentum, out=self.running_mean)
            #     torch.add(self.running_var * (1.0 - self.momentum), var * self.momentum, out=self.running_var)

            # mean = mean[None, :, None, None]
            # invstd = (1./torch.sqrt(var+self.eps))[None, :, None, None]

            # return (grad - mean)*invstd
        else:
            return grad
            # mean = self.running_mean[None, :, None, None]
            # invstd = (1./torch.sqrt(self.running_var+self.eps))[None, :, None, None]

            # return (grad - mean)*invstd

    def forward(self, x):
        if x.requires_grad:
            try:
                x.retain_grad()
                x.register_hook(self.output_grad_hook)
            except:
                pass
        return x



class AddGloroAbsentLogit(scheduler.ScheduledModule):
    def __init__(self, output_module, epsilon, num_iter, lipschitz_computer, detach_lipschitz_computer=False, lipschitz_multiplier=1.0):
        super(AddGloroAbsentLogit, self).__init__()
        assert hasattr(output_module, 'parent'), 'output module must have attribute parent (as is defined in LipschitzLayerComputer.'
        self.W = lambda : output_module.parent.weight
        self.lc = lipschitz_computer
        self.detach_lipschitz_computer = detach_lipschitz_computer
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.lipschitz_multiplier = lipschitz_multiplier

    def forward(self, x):
        eps = self.epsilon

        K_lip = self.lipschitz_multiplier * self.lc(num_iter=self.num_iter)
        W = self.W()
        if self.detach_lipschitz_computer:
            K_lip = K_lip.detach()
            W = W.detach()
        return gloro_absent_logit(x, W, K_lip, epsilon=eps)

    def __repr__(self):
        if isinstance(self.epsilon, scheduler.Scheduler):
            return 'AddGloroAbsentLogit(eps={}, num_iter={})'.format(self.epsilon, self.num_iter)
        else:
            return 'AddGloroAbsentLogit(eps={:.2f}, num_iter={})'.format(self.epsilon, self.num_iter)

# ToDo: Norm preserving layer
# Compare input with output to determine factor: factor=(output.norm()/input.norm()).max()

# class InputNormPreservingLayer(nn.Module):
#     def __init__(self, momentum=0.01):
#         super(InputNormPreservingLayer, self).__init__()
#         self.register_buffer('denominator', torch.ones(1))
#         self.momentum = momentum

#     def _forward(self, x):
#         raise NotImplementedError

#     def forward(self, x_in):
#         x_in_norm = x_in.view(x_in.shape[0], -1).norm(p=2, dim=-1)
#         x_out = self._forward(x_in)
#         x_out_norm = x_in.view(x_in.shape[0], -1).norm(p=2, dim=-1)

#         with torch.no_grad():
#             norm_div = (x_out_norm / x_in_norm).mean()
#             self.denominator = self.denominator * (1.0-self.momentum) + norm_div * self.momentum

#         return x_out / self.denominator


class PatchView(nn.Module):
    def forward(self, x):
        assert len(x.shape) == 5
        return x.view(x.shape[0]*x.shape[1], *x.shape[2:])


class InputNormPreservingConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        self.momentum = kwargs.pop('momentum', 0.01)
        self.dims = kwargs.pop('preserve_dim', [1, 2, 3])
        self.grad_norm_preserving = kwargs.pop('grad_norm_preserving', False)
        super(InputNormPreservingConv2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('denominator', torch.ones(1))
        self.register_buffer('grad_denominator', torch.ones(1))
        self.output_grad_norm = None

    def output_grad_hook(self, grad):
        with torch.no_grad():
            norms = torch.norm(grad, dim=self.dims, p=2)
            # norms = grad.view(grad.shape[0], -1).norm(p=2, dim=-1)
            self.output_grad_norm = norms.detach()

        return None

    def input_grad_hook(self, grad):
        if self.training:
            assert self.output_grad_norm is not None
            with torch.no_grad():
                norms = torch.norm(grad, dim=self.dims, p=2)
                
                if self.output_grad_norm.shape != norms.shape:
                    norm_div = (norms.view(norms.shape[0], -1).mean(-1)[0] / (self.output_grad_norm.view(norms.shape[0], -1).mean(-1)[0] + 1.e-9)).mean()
                else:
                    norm_div = (norms / (self.output_grad_norm + 1.e-9)).mean()
                # print('out_norm: {:.3f}, in_norm: {:.3f}, div: {:.3f}, denom={:.3f}'.format(self.output_grad_norm.mean().item(), norms.mean().item(), norm_div.item(), self.grad_denominator.item()))
                self.output_grad_norm = None

                self.grad_denominator = self.grad_denominator * (1.0-self.momentum) + norm_div * self.momentum
        else:
            self.output_grad_norm = None

        return grad / self.grad_denominator

    def _forward(self, x):
        return super(InputNormPreservingConv2d, self).forward(x)

    def forward(self, x_in):
        if self.grad_norm_preserving:
            try:
                x_in.retain_grad()
                x_in.register_hook(self.input_grad_hook)
            except:
                pass

        x_out = self._forward(x_in)

        if self.training:
            x_in_norm = torch.norm(x_in, dim=self.dims, p=2)
            x_out_norm = torch.norm(x_out, dim=self.dims, p=2)
            # x_in_norm = x_in.view(x_in.shape[0], -1).norm(p=2, dim=-1)
            # x_out_norm = x_out.view(x_out.shape[0], -1).norm(p=2, dim=-1)

            with torch.no_grad():
                if x_out_norm.shape != x_in_norm.shape:
                    norm_div = (x_out_norm.view(x_out_norm.shape[0], -1).mean(-1)[0] / (x_in_norm.view(x_in_norm.shape[0], -1).mean(-1)[0] + 1.e-9)).mean()
                else:
                    norm_div = (x_out_norm / (x_in_norm + 1.e-9)).mean()
                self.denominator = self.denominator * (1.0-self.momentum) + norm_div * self.momentum
                # print('weight-avg: {:.3f}, bias-avg: {:.3f} || in_norm: {:.3f}, out_norm: {:.3f}, div: {:.3f}, denom={:.3f}'.format(self.weight.abs().mean().item(), self.bias.abs().mean().item(), x_in_norm.mean().item(), x_out_norm.mean().item(), norm_div.item(), self.denominator.item()))
                # print('denom: {:.3f}'.format(self.denominator.item()))

        x = x_out / self.denominator.clamp(min=1.0)

        if self.grad_norm_preserving:
            try:
                x.retain_grad()
                x.register_hook(self.output_grad_hook)
            except:
                pass

        return x

class InputNormPreservingLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, **kwargs):
        self.momentum = kwargs.pop('momentum', 0.01)
        self.dims = kwargs.pop('preserve_dim', [1])
        self.grad_norm_preserving = kwargs.pop('grad_norm_preserving', False)
        super(InputNormPreservingLinear, self).__init__(in_channels, out_channels, **kwargs)
        self.register_buffer('denominator', torch.ones(1))
        self.register_buffer('grad_denominator', torch.ones(1))
        self.output_grad_norm = None

    def output_grad_hook(self, grad):
        with torch.no_grad():
            # norms = grad.view(grad.shape[0], -1).norm(p=2, dim=-1)
            norms = torch.norm(grad, dim=self.dims, p=2)
            self.output_grad_norm = norms.detach()

        return None

    def input_grad_hook(self, grad):
        if self.training:
            assert self.output_grad_norm is not None
            with torch.no_grad():
                norms = torch.norm(grad, dim=self.dims, p=2)
                
                if self.output_grad_norm.shape != norms.shape:
                    norm_div = (norms.view(norms.shape[0], -1).mean(-1)[0] / (self.output_grad_norm.view(norms.shape[0], -1).mean(-1)[0] + 1.e-9)).mean()
                else:
                    norm_div = (norms / (self.output_grad_norm + 1.e-9)).mean()
                self.output_grad_norm = None

                self.grad_denominator = self.grad_denominator * (1.0-self.momentum) + norm_div * self.momentum
        else:
            self.output_grad_norm = None

        return grad / self.grad_denominator

    def _forward(self, x):
        return super(InputNormPreservingLinear, self).forward(x)

    def forward(self, x_in):
        if self.grad_norm_preserving:
            try:
                x_in.retain_grad()
                x_in.register_hook(self.input_grad_hook)
            except:
                pass

        x_out = self._forward(x_in)

        if self.training:
            x_in_norm = torch.norm(x_in, dim=self.dims, p=2)
            x_out_norm = torch.norm(x_out, dim=self.dims, p=2)
            # x_in_norm = x_in.view(x_in.shape[0], -1).norm(p=2, dim=-1)
            # x_out_norm = x_out.view(x_out.shape[0], -1).norm(p=2, dim=-1)

            with torch.no_grad():
                if x_out_norm.shape != x_in_norm.shape:
                    norm_div = (x_out_norm.view(x_out_norm.shape[0], -1).mean(-1)[0] / (x_in_norm.view(x_in_norm.shape[0], -1).mean(-1)[0] + 1.e-9)).mean()
                else:
                    norm_div = (x_out_norm / (x_in_norm + 1.e-9)).mean()
                self.denominator = self.denominator * (1.0-self.momentum) + norm_div * self.momentum
                # print('denom: {:.3f}'.format(self.denominator.item()))

        x = x_out / self.denominator.clamp(min=1.0)

        if self.grad_norm_preserving:
            try:
                x.retain_grad()
                x.register_hook(self.output_grad_hook)
            except:
                pass

        return x


class BNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(BNConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, *args, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MaxNormTo1(nn.Module):
    def __init__(self, forward_norm=False, momentum=0.01, clamp_below=True, grad_norm=False, grad_momentum=0.3):
        super(MaxNormTo1, self).__init__()
        if forward_norm:
            self.register_buffer('denominator', torch.ones(1))
        else:
            self.denominator = None
            
        if grad_norm:
            self.register_buffer('grad_denominator', torch.ones(1))
        else:
            self.grad_denominator = None
        self.momentum = momentum
        self.grad_momentum = grad_momentum
        self.clamp_below = clamp_below
        
        self.stats = dict(num=0)
        
    def output_grad_hook(self, grad):
#         norms = torch.norm(grad,dim=1)
#         grad = grad / norms.max() #(self.stats['avg_max_grad_norm'] / self.stats['num'])
        if self.grad_denominator is not None:
            grad = self.norm_grad(grad)
        # with torch.no_grad():
        #     norms = torch.norm(grad,dim=1)
        #     if 'avg_grad_norm' not in self.stats:
        #         self.stats['avg_grad_norm'] = norms.mean()
        #         self.stats['avg_max_grad_norm'] = norms.max()
        #         self.stats['avg_min_grad_norm'] = norms.min()
        #     else:
        #         self.stats['avg_grad_norm'] += norms.mean()
        #         self.stats['avg_max_grad_norm'] += norms.max()
        #         self.stats['avg_min_grad_norm'] += norms.min()
        return grad
    
    def reset_stats(self):
        self.stats = dict(num=0)
        
    def forward(self, x):
        if self.denominator is not None:
            if self.training:
                with torch.no_grad():
                    norms = torch.norm(x,dim=1)
                    max_norm = norms.max()
                    self.denominator = self.denominator * (1.0-self.momentum) + max_norm * self.momentum
                    if self.clamp_below:
                        self.denominator.clamp_(min=1.0)

            x = x / self.denominator
        
        # with torch.no_grad():
        #     norms = torch.norm(x,dim=1)
        #     self.stats['num'] += 1
        #     if 'avg_norm' not in self.stats:
        #         self.stats['avg_norm'] = norms.mean()
        #         self.stats['avg_max_norm'] = norms.max()
        #         self.stats['avg_min_norm'] = norms.min()
        #         self.stats['avg_value'] = x.mean()
        #     else:
        #         self.stats['avg_norm'] += norms.mean()
        #         self.stats['avg_max_norm'] += norms.max()
        #         self.stats['avg_min_norm'] += norms.min()
        #         self.stats['avg_value'] += x.mean()
        
        if self.grad_denominator is not None and x.requires_grad:
            try:
                x.retain_grad()
                x.register_hook(self.output_grad_hook)
            except:
                pass
        
        return x
    
    def norm_grad(self, x):
        if self.training:
            with torch.no_grad():
                if len(x.shape) == 4:
                    norms = torch.norm(x, dim=[1,2,3], p=2, keepdim=True)
                else:
                    norms = torch.norm(x, dim=1, p=2, keepdim=True)
                max_norm = norms.max()
                self.grad_denominator = self.grad_denominator * (1.0-self.grad_momentum) + max_norm * self.grad_momentum
            result = x / norms.clamp(1.e-8)
            return result
        else:
            return x / self.grad_denominator
    
    def __repr__(self):
        if self.denominator is None and self.grad_denominator is None:
            return 'NormStats()'
        if self.grad_denominator is not None:
            return 'MaxNormTo1(momentum={}, grad_momentum={}, clamp_below={})'.format(self.momentum, self.grad_momentum, self.clamp_below)
        else:
            return 'MaxNormTo1(momentum={}, clamp_below={})'.format(self.momentum, self.clamp_below)


class FakeReLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeReLU(nn.Module):
    def forward(self, x):
        return FakeReLUFunc.apply(x)


class MaxMin(nn.Module):
    def __init__(self, num_units, axis=1):
        super(MaxMin, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        maxes = maxout(x, self.num_units, self.axis)
        mins = minout(x, self.num_units, self.axis)
        maxmin = torch.cat((maxes, mins), dim=self.axis)
        return maxmin

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units)


class Flatten(nn.Module):
	def __init__(self, dim):
		super(Flatten, self).__init__()
		self.dim = dim

	def forward(self, x):
		return torch.flatten(x, self.dim)

	def __repr__(self):
		return 'Flatten(dim={})'.format(self.dim)


class Max(nn.Module):
    def __init__(self, dim):
        super(Max, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1], -1).max(-1)[0]
        # return torch.max(x, dim=self.dim)

    def __repr__(self):
        return 'Max(dim={})'.format(self.dim)


class PrintOutputShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x


class EventLayer(nn.Module):
    pass


class GradientDumpingLayer(EventLayer):
    def __init__(self, name, save_path, phase='both', at_epochs=-1):
        super(GradientDumpingLayer, self).__init__()
        self.layer = name
        self.save_path = save_path
        self.at_epochs = at_epochs
        self.cur_epoch = -1
        self.phase = phase
        assert phase in ['train', 'val', 'both']

    def dump(self, tensor, fileprefix):
        filename = fileprefix
        if self.cur_epoch != -1:
            filename += '_epoch{}'.format(self.cur_epoch)
        filename += '.bin'

        if self.training:
            filep = os.path.join(self.save_path, 'train')
        else:
            filep = os.path.join(self.save_path, 'val')
        
        if not os.path.exists(filep):
            os.makedirs(filep)

        util.write_bin_data(os.path.join(filep, filename), tensor)
        util.write_bin_data(os.path.join(filep, filename.replace('.bin', '_targets.bin')), self.imids.cpu())

    def output_grad_hook(self, grad):
        tensor = grad.mean(1) # take average over color channels
        N, H, W = tensor.shape
        tensor = tensor.view(N,-1).cpu()
        # print('GradientDumpingLayer :: dumping')
        self.dump(tensor, self.layer)
        # print('GradientDumpingLayer :: done')

        return grad

    def forward(self, x, **kwargs):
        epoch = kwargs['epoch']
        self.imids = kwargs['imids']
        if self.phase != 'both':
            if self.training and self.phase == 'val':
                return x
            if not self.training and self.phase == 'train':
                return x

        if self.at_epochs == -1 or epoch in self.at_epochs:
            if epoch != self.cur_epoch:
                print('GradientDumpingLayer :: dumping at epoch {}'.format(epoch))
            self.cur_epoch = epoch

            x.requires_grad = True
            if x.requires_grad:
                try:
                    x.retain_grad()
                    x.register_hook(self.output_grad_hook)
                except:
                    pass
        return x




# -------------------------------------------------------------------------------------
# Helper functions

def gloro_absent_logit(predictions, last_linear_weight, lipschitz_estimate, epsilon):
    def get_Kij(pred, lc, W):
        kW = W*lc
        
        # with torch.no_grad():
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

    return all_logits

def process_maxmin_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis+1, num_channels // num_units)
    return size


def maxout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]