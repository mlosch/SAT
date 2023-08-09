import os
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import struct
import time


class DefaultFallbackDict(dict):
    def __init__(self, *args, **kwargs):
        self.fallback = kwargs.pop('fallback')
        super(DefaultFallbackDict, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        if key not in self:
            return self.fallback
        else:
            return super(DefaultFallbackDict, self).__getitem__(key)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Meter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0

    def update(self, val, n=None):
        self.val = val
        self.avg = val
        

class AverageMeter(Meter):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def merge_dicts(x, y):
    z = {}
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        if isinstance(x[key], dict) and isinstance(y[key], dict):
            z[key] = dict_of_dicts_merge(x[key], y[key])
        if isinstance(x[key], list) and isinstance(y[key], list):
            z[key] = x[key] + y[key]
    for key in x.keys() - overlapping_keys:
        z[key] = deepcopy(x[key])
    for key in y.keys() - overlapping_keys:
        z[key] = deepcopy(y[key])
    return z


def domain_mask(domain, target):
    with torch.no_grad():
        # mask = target == domain[0]
        # for class_idx in domain[1:]:
        #     mask |= (target == class_idx)

        # domain = torch.Tensor(domain).cuda()
        compareview = domain.expand(target.shape[0], domain.shape[0]).T
        mask = ((compareview == target).T.sum(1)).bool()
        # compareview = target.expand(domain.shape[0], target.shape[0]).T
        # mask2 = ((compareview == domain).T.sum(1)).bool()

        # print(mask.shape, mask2.shape)
        # print(mask2)
        # assert torch.all(mask==mask2)
    return mask


def load_bin_data(filep, fmt, verbose=False):
    assert fmt in ['f', 'I']
    with open(filep, 'rb') as f:
        # read header (e.g. dimensioanlity of tensor)
        header_fmt = '>I'
        header_sz = struct.calcsize(header_fmt)
        chunk_numel = struct.unpack(header_fmt, f.read(header_sz))[0]
        if verbose:
            print('elements per chunk', chunk_numel)
        
        # read content
        dt = time.time()
        buffer = f.read()
        if verbose:
            print('reading file', time.time()-dt)
        
        result = np.frombuffer(buffer, dtype='>'+fmt)
        if chunk_numel > 1:
            result = result.reshape(len(result)//chunk_numel, chunk_numel)
    
    return result


def write_bin_data(filep, tensor):
    """
    Appends tensor as binary data to file. Hence, allows to continuously stream data to file.
    Creates file if it does not exist.
    """
    dims = max(1, tensor.numel() // tensor.shape[0])
    if not os.path.exists(filep):
        # first entry in file is dimensionality of output in big endian format (>)
        with open(filep, 'wb') as f:
            # value is saved as unsigned int
            f.write(struct.pack('>I', dims))

    with open(filep, 'ab') as f:
        for entry in tensor:
            entry = entry.numpy()
            fmt = {'float32': 'f', 'int32': 'i', 'uint32': 'I', 'int64': 'l', 'uint64': 'L'}[str(entry.dtype)]
            # convert float32 array to bytes in big endian format (>)
            if dims == 1:
                s = struct.pack('>%s'%fmt, entry)
            else:
                s = struct.pack('>%d%s'%(dims, fmt), *entry)
            f.write(s)


def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """

    bn_init_parts = batchnorm.split('-')
    batchnorm = bn_init_parts[0]

    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif conv == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif conv == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif conv == 'normal1':
                nn.init.normal_(m.weight, 0.0, 1.0)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)) and m.weight is not None:
            if batchnorm == 'normal':
                nn.init.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                nn.init.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            nn.init.constant_(m.bias, 0.0)

            if len(bn_init_parts) >= 2:
                with torch.no_grad():
                    if bn_init_parts[1] == 'L2normed':
                        m.weight /= torch.norm(m.weight, p=2)
                    elif bn_init_parts[1] == 'L1normed':
                        m.weight /= torch.norm(m.weight, p=1)
                    elif bn_init_parts[1] == 'Linfnormed':
                        m.weight /= torch.max(m.weight)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif linear == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif linear == 'normal':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif linear == 'normal1':
                nn.init.normal_(m.weight, 0.0, 1.0)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        nn.init.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        nn.init.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    nn.init.constant_(param, 0)