import torch
import torch.nn.functional as F
# import torch.nn as nn

class MaskedLinearFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        ctx.save_for_backward(input, weight, bias, mask)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            if mask is not None:
                grad_weight = grad_output[mask].t().mm(input[mask])
            else:
                grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            if mask is not None:
                grad_bias = grad_output[mask].sum(0)
            else:
                grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


class GradientMaskingLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super(GradientMaskingLinear, self).__init__(*args, **kwargs)
        self.mask_update = None
        
    @property
    def mask_update(self):
        return self.__mask_update
    
    @mask_update.setter
    def mask_update(self, mask):
        self.__mask_update = mask
        
    def forward(self, x):
        if self.training:
            mask = self.__mask_update
        else:
            mask = None
        o = MaskedLinearFunction.apply(x, self.weight, self.bias, mask)
        self.mask_update = None
        return o

    # def __repr__(self):
    #     return 'M{'+super(GradientMaskingLinear, self).__repr__()+'}'



# -------------------------------------------------------------------------------------

class MaskedConv2dFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, mask=None):
        ctx.save_for_backward(input, weight, bias, mask)
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups

        output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        grad_input = grad_weight = grad_bias = grad_stride = grad_padding = grad_dilation = grad_groups = None

        # print(grad_output.shape)

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            if mask is not None:
                grad_weight = torch.nn.grad.conv2d_weight(input[mask], weight.shape, grad_output[mask], stride, padding, dilation, groups)
            else:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[2]:
            if mask is not None:
                grad_bias = grad_output[mask].sum(0, 2, 3).squeeze(0) 
            else:
                grad_bias = grad_output.sum(0, 2, 3).squeeze(0) 
    
        return grad_input, grad_weight, grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups, None


class GradientMaskingConv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(GradientMaskingConv2d, self).__init__(*args, **kwargs)
        self.mask_update = None

    @property
    def mask_update(self):
        return self.__mask_update
    
    @mask_update.setter
    def mask_update(self, mask):
        self.__mask_update = mask
        
    def forward(self, x):
        if self.training:
            mask = self.__mask_update
        else:
            mask = None
        o = MaskedConv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, mask)
        self.mask_update = None
        return o

    # def __repr__(self):
    #     return 'M{'+super(GradientMaskingConv2d, self).__repr__()+'}'
