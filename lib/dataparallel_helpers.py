import torch
from torch.nn.parallel._functions import Scatter, Gather


# copy from torch.nn.parallel.scatter_gather
def _is_namedtuple(obj):
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def gracious_gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device.
    Use 'cpu' for CPU to avoid a deprecation warning.
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            # if not all(len(out) == len(d) for d in outputs):
            #     raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs if k in d])) # gracious
                             for k in out)
        if _is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


class GraciousDataParallel(torch.nn.DataParallel):
    def gather(self, outputs, output_device):
        return gracious_gather(outputs, output_device, dim=self.dim)