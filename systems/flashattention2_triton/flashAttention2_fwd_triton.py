import triton
import triton.language as tl


@triton.jit
def _attention_step(Q, K, V, O_i, acc_denominator, _max, scale, mask=None):
    S = tl.dot(Q, tl.trans(K)) * scale
    if mask is not None:
        S = tl.where(mask, S, float("-inf"))
    _max_new = tl.maximum(_max, tl.max(S, axis=-1))
    P = tl.exp(S - _max_new[:, None])
    correction_term = tl.exp(_max - _max_new)
    acc_denominator_new = correction_term * acc_denominator + tl.sum(P, axis=-1)
    O_i_new = correction_term[:, None] * O_i + tl.dot(P, V)
    return O_i_new, acc_denominator_new, _max_new


@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr
):
    # Program indices
    seq_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Q block pointer
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(seq_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(seq_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(seq_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    acc_denominator = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    _max = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    if is_causal:
        # deal with fully unmasked portions

        for _ in range (tl.cdiv(seq_tile_index, K_TILE_SIZE)):

            K = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
            V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            O_i, acc_denominator, _max = _attention_step(Q, K, V, O_i, acc_denominator, _max, scale)

            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        # Load diagonal entries, already pointing at these

        K = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Apply mask

        query_indices = seq_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        key_indices = seq_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        mask = query_indices[:, None] >= key_indices[None, :]
        O_i, acc_denominator, _max = _attention_step(Q, K, V, O_i, acc_denominator, _max, scale, mask=mask)

    else:
        for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):

            K = tl.load(K_block_ptr, boundary_check=(1, 0), padding_option="zero")
            V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            O_i, acc_denominator, _max = _attention_step(Q, K, V, O_i, acc_denominator, _max, scale)
            K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
            V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = O_i / acc_denominator[:, None]
    tl.store(O_block_ptr, O_i, boundary_check=(0, 1))

    acc_L = _max + tl.log(acc_denominator)
    tl.store(L_block_ptr, acc_L, boundary_check=(0,))