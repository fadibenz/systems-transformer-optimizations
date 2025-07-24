from typing import Dict, Any, Optional, Callable
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class OptimizerStateSharding(torch.optim.Optimizer):

    def __init__(self, params,
                 optimizer_cls,
                 **kwargs):

        if not dist.is_initialized():
            print("You must initialize distributed environment to use sharded optimizer")
            raise RuntimeError

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.optimizer_cls = optimizer_cls
        self.local_optimizer = None
        self.optimizer_kwargs = kwargs
        super().__init__(params, kwargs)

    def step(self,
             closure: Optional[Callable[[], float]] = None,
             **kwargs
             ) -> Optional[float]:

        loss = self.local_optimizer.step(closure, **kwargs)

        if loss is not None:
            loss = torch.tensor([loss], dtype=torch.float32)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            loss = loss.item()

        for group in self.param_groups:
            params = group["params"]
            params_data = [p.data for p in params]
            nb_params = len(params_data)
            start_idx, end_idx = self._get_shard_range(nb_params, self.rank)
            local_shard = params_data[start_idx:end_idx]

            if not local_shard:
                flat_local_shard = torch.empty(0, dtype=params_data[0].dtype, device=params_data[0].device)
            else:
                flat_local_shard = _flatten_dense_tensors(local_shard)


            size = torch.tensor([flat_local_shard.numel()], dtype=torch.long, device=flat_local_shard.device)
            size_list = [torch.zeros(1, dtype=torch.long, device=flat_local_shard.device ) for _ in range(self.world_size)]
            dist.all_gather(size_list, size)

            size_list = [s.item() for s in size_list]
            max_size = max(size_list)

            padded_local_shard = torch.zeros(max_size, device=flat_local_shard.device, dtype=flat_local_shard.dtype)
            padded_local_shard[:size.item()] = flat_local_shard

            gathered_tensors = [torch.zeros_like(padded_local_shard) for _ in range(self.world_size)]
            dist.all_gather(gathered_tensors, padded_local_shard)

            for rank, shard_size  in zip(range(self.world_size), size_list):
                start_idx, end_idx = self._get_shard_range(nb_params, rank)
                shard = params_data[start_idx:end_idx]
                if not shard:
                    continue

                padded_shard_data = gathered_tensors[rank][:int(shard_size)]
                unflat = _unflatten_dense_tensors(padded_shard_data, shard)

                for param, data in zip(shard, unflat):
                    param.copy_(data)

        return loss

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        super().add_param_group(param_group)

        params = param_group["params"]
        nb_params = len(params)

        start_idx, end_idx = self._get_shard_range(nb_params, self.rank)
        local_params = params[start_idx:end_idx]

        if not local_params:
            return

        local_param_group = {key: value for key, value in param_group.items()}
        local_param_group['params'] = local_params

        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls(local_params, **self.optimizer_kwargs)
        else:
            self.local_optimizer.add_param_group(local_params)

    def _get_shard_range(self, nb_params, rank):
        local_size = (nb_params + self.world_size - 1) // self.world_size
        start_idx = rank * local_size
        end_idx = min(start_idx + local_size, nb_params)

        return start_idx, end_idx