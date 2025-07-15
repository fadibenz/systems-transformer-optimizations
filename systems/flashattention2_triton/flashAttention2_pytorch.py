from typing import Any
import math
import torch
from einops import einsum


class FlashAttention2(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                Q: torch.Tensor,
                K:torch.Tensor,
                V:torch.Tensor,
                is_causal: bool = False)-> torch.Tensor:


        B_q = 16
        B_k = 16

        b, N_q, d = Q.size()
        N_k = K.size(-2)

        # Initialization
        m: torch.Tensor = torch.fill(torch.empty(b, N_q), -1e-10)
        l: torch.Tensor = torch.zeros(b, N_q)

        L: torch.Tensor = torch.zeros(b, N_q)
        O: torch.Tensor = torch.empty_like(Q, requires_grad=True)

        # Outer Loop
        for i in range(0, N_q, B_q):
            q_tile = Q[:, i:i + B_q, :] # load Q_i
            m_i = m[:, i: i+ B_q]
            l_i = l[:, i: i + B_q]
            O_i = O[:, i: i + B_q, :]

            # Inner Loop
            for j in range(0, N_k, B_k):
                k_tile = K[:, j: j + B_k, :] # load K_i
                v_tile = V[:, j: j + B_k, :] # load V_i

                # Tile of pre-softmax attention
                S = einsum(q_tile, k_tile, "... B_q d, ... B_k d -> ... B_q B_k") / math.sqrt(d)

                # Portion of unnormalized Similarity matrixx
                m_i_new = torch.maximum(m_i, torch.max(S, dim=-1).values)
                P_i = torch.exp(S - m_i_new.unsqueeze(-1))

                correction_term = torch.exp(m_i - m_i_new)
                l_i = correction_term * l_i + torch.sum(P_i, -1)
                O_i = correction_term.unsqueeze(-1) * O_i + P_i @ v_tile

                # Update max
                m_i = m_i_new

            # Normalize rows
            O[:, i: i + B_q, :] = O_i / l_i.unsqueeze(-1)
            L[:, i: i + B_q] = m_i + torch.log(l_i)

        ctx.save_for_backward(L)
        return O

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplemented