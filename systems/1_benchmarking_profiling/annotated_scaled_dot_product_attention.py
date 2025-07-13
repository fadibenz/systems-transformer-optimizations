import torch
from einops import  einsum
from torch import Tensor
from jaxtyping import Float, Bool
import math
from transformer_implementation.nn_utils import softmax
from torch.cuda import nvtx


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention annotated.

        This is an annotated version of the scaled-dot product function found in transformer_implementation
        it adds nvtx ranges to allow for clean progiling

    """

    d_k = K.shape[-1]

    nvtx.range_push("computing attention scores")
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    nvtx.range_pop()

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    nvtx.range_push("computing softmax")
    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
    nvtx.range_pop()

    nvtx.range_push("final matmul")
    final = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    nvtx.range_pop()

    return final
