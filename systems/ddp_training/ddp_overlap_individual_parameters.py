import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDPOverlap(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module):
        super().__init__()

        params = [param.data for param in model.parameters()]

        flattened_params = _flatten_dense_tensors(params)
        dist.broadcast(
            flattened_params,
            src=0
        )

        broadcast_params = _unflatten_dense_tensors(flattened_params, params)

        for original_param, broadcast_param in zip(params, broadcast_params):
            original_param.copy_(broadcast_param)

        self.hooks = []
        for param in model.parameters():
            if param.requires_grad:
                h = param.register_post_accumulate_grad_hook(lambda p: dist.all_reduce(p.grad,
                                                                               op=dist.ReduceOp.AVG,
                                                                               async_op=True))
                self.hooks.append(h)
        self.wrapped_model = model


    def forward(self, *inputs, **kwargs):
        return self.wrapped_model(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for h in self.hooks:
            h.wait()
        self.hooks.clear()