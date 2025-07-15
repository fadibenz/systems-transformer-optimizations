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
                is_causal: bool = False)-> tuple[torch.Tensor, torch.Tensor]:


        # Simple arithmetic
        B_q = 16
        B_k = 16

        N_q, d = Q.size()
        N_k = K.size(0)

        num_splits_q = math.ceil(N_q / B_q)
        num_splits_k = math.ceil(N_k / B_k)

        # Initialization
        m = torch.fill(torch.empty(B_q), 1e-6)
        l = torch.zeros(B_q)
        L: torch.Tensor = torch.zeros(B_q)
        O: torch.Tensor = torch.empty_like(Q, requires_grad=True)

        # Main algorithm
        # Outer Loop
        for i in range(0, num_splits_q, B_q):
            q_tile = Q[i:i + B_q, :] # load Q_i

            m_i = m[i, i+ B_q]
            l_i = l[i, i + B_q]
            O_i = O[i, i + B_q]

            # Inner Loop
            for j in range(0, num_splits_k, B_k):
                k_tile = K[i: i + B_k, :] # load K_i
                v_tile = V[i: i + B_k, :] # load V_i

                # Tile of pre-softmax attention
                S = einsum(q_tile, k_tile, "B_q d, B_k d -> B_q B_k") / math.sqrt(d)

                # Portion of un-normalized Similarity matrixx
                P_i = torch.exp(S - m_i)

                m_i_new = torch.maximum(m_i, torch.max(S, dim=-1))

                correction_term = torch.exp(m_i - m_i_new)
                l_i = correction_term * l_i + torch.sum(S, -1)
                O_i = correction_term * O_i + P_i @ v_tile

                # Update max
                m_i = m_i_new

            # Normalize rows
            O_i = O_i / l_i

            L[i: i + B_q, :] = m_i + torch.log(l_i)
        return O, L

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplemented