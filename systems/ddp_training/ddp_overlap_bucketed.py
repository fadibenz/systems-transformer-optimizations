import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

MB = 1024 * 1024

class DDPOverlap(torch.nn.Module):
    def __init__(self,
                 module: torch.nn.Module,
                 bucket_size_mb: float):
        super().__init__()

        self.pending_ops = []
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        self._broadcast()

        self._setup_hooks()

    def _broadcast(self):
        params = [param.data for param in self.module.parameters()]

        flattened_params = _flatten_dense_tensors(params)
        dist.broadcast(
            flattened_params,
            src=0
        )

        broadcast_params = _unflatten_dense_tensors(flattened_params, params)

        for original_param, broadcast_param in zip(params, broadcast_params):
            original_param.copy_(broadcast_param)

    def _setup_hooks(self):
        buckets = self._construct_buckets()
        for bucket_params in buckets:
            hook = self._create_bucket_hook(bucket_params)

            for param in bucket_params:
                param.register_post_accumulate_grad_hook(hook)


    def _create_bucket_hook(self, bucket_params):
        grad_count = 0
        def bucket_hook(p):
            nonlocal grad_count
            grad_count += 1
            if grad_count == len(bucket_params):
                grads = [t.grad for t in bucket_params]
                flattened_grads = _flatten_dense_tensors(grads)
                work = dist.all_reduce(flattened_grads,
                                    op=dist.ReduceOp.AVG,
                                    async_op=True)
                self.pending_ops.append((work, flattened_grads, grads))
                grad_count = 0
        return bucket_hook

    def _construct_buckets(self):
        params = list(reversed([p for p in self.module.parameters() if p.requires_grad]))

        buckets = []
        current_bucket = []
        current_bucket_size = 0

        for param in reversed(params):
            current_bucket.append(param)
            current_bucket_size += param.data.numel() * param.data.element_size()

            if current_bucket_size / MB >= self.bucket_size_mb:
                buckets.append(current_bucket)

                current_bucket = []
                current_bucket_size = 0

        if current_bucket_size != 0:
            buckets.append(current_bucket)

        return buckets

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for work, flattened_grads, grads in self.pending_ops:
            work.wait()
            unflattened_tensors = _unflatten_dense_tensors(flattened_grads, grads)
            for original_grad, average_grad in zip(grads, unflattened_tensors):
                original_grad.copy_(average_grad)

        self.pending_ops.clear()