import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class DDPOverlap(torch.nn.Module):
    def __init__(self,
                 module: torch.nn.Module):
        super().__init__()

        params = [param.data for param in module.parameters()]

        flattened_params = _flatten_dense_tensors(params)
        dist.broadcast(
            flattened_params,
            src=0
        )

        broadcast_params = _unflatten_dense_tensors(flattened_params, params)

        for original_param, broadcast_param in zip(params, broadcast_params):
            original_param.copy_(broadcast_param)

        self.pending_ops = []

        def create_hook(param):
            def hook_fn(p):
                h = dist.all_reduce(p.grad,
                                    op=dist.ReduceOp.AVG,
                                    async_op=True)
                self.pending_ops.append(h)
            return hook_fn

        for param in module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(create_hook(param))
        self.module = module


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for h in self.pending_ops:
            h.wait()
        self.pending_ops.clear()