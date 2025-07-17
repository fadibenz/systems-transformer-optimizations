import math
import torch
import triton

from systems.flashattention2_triton.flashAttention2_fwd_triton  import flash_fwd_kernel
from systems.flashattention2_triton.flashAttention2_bwd_triton import flash_bwd_kernel_dK_dV, flash_bwd_kernel_dQ


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

        O = torch.empty_like(Q)
        L = torch.empty((BATCH, N_QUERIES), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        T_q = triton.cdiv(N_QUERIES, Q_TILE_SIZE)
        scale = 1.0 / math.sqrt(float(D))

        flash_fwd_kernel[(T_q, BATCH)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        Q, K, V, O, L = ctx.saved_tensors
        dO = grad_outputs[0]

        BATCH, N_QUERIES, D = Q.size()
        N_KEYS = K.size(-2)

        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32
        scale = 1.0 / math.sqrt(float(D))

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        grid_dk_dv = (triton.cdiv(N_KEYS, K_TILE_SIZE), BATCH)
        grid_dq = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), BATCH)

        common_args = [
            Q, K, V, O, dO, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES, N_KEYS, scale,
            D, Q_TILE_SIZE, K_TILE_SIZE, ctx.is_causal
        ]

        flash_bwd_kernel_dK_dV[grid_dk_dv](*common_args[:6], dK, dV, *common_args[6:])

        flash_bwd_kernel_dQ([grid_dq])(*common_args[:6], dQ, *common_args[6:])

        return dQ, dK, dV, None