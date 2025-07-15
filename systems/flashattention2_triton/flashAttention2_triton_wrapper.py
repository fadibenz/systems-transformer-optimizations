import math
from typing import Any
import torch
from systems.flashattention2_triton.flashAttention2_fwd_triton  import flash_fwd_kernel

class FlashAttention2Triton(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                Q: torch.Tensor,
                K:torch.Tensor,
                V:torch.Tensor,
                is_causal: bool = False)-> torch.Tensor:

        BATCH, N_QUERIES, D = Q.size()
        N_KEYS = K.size(-2)

        assert D == K.shape[-1], "Incompatible dimensions of keys and queries"
        assert N_KEYS == V.shape[-2], "Incompatible dimensions of keys and values"
        assert Q.is_cuda and K.is_cuda and V.is_cuda, "Expected CUDA tensors"
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous(), "Tensors must be contiguous"

        O = torch.empty_like(Q, requires_grad=True)
        L = torch.empty((BATCH, N_QUERIES), device=Q.device, requires_grad=True)
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32


        T_q = (N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE
        scale = 1 / math.sqrt(D)

        flash_fwd_kernel[(T_q, BATCH)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE
        )

        ctx.save_for_backward(L)
        return O

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any):
        raise NotImplemented
