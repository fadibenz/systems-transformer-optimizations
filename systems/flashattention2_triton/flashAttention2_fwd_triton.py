import triton
import triton.language as tl

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
        K_TILE_SIZE: tl.constexpr
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
        offsets=(seq_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(seq_tile_index * K_TILE_SIZE, 0),
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
        order=(1, 0)
    )

    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L = tl.load(L_block_ptr, boundary_check=(0, 1), padding_option="zero")

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    acc_denominator = tl.zeros(Q_TILE_SIZE, dtype=tl.float32)
    _max = tl.full(Q_TILE_SIZE, 1e-6, dtype=tl.float32)

    for _ in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        S = tl.dot(Q, K, transB=True) * scale
        _max_new = tl.maximum(_max, tl.max(S, dim=-1).values())

        P = tl.exp(S - _max_new)
        correction_term = tl.exp(_max - _max_new)

        acc_denominator = correction_term * acc_denominator + tl.sum(P, -1)
        O_i = correction_term * O_i + P @ V

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
        _max = _max_new

    O_i = O_i / acc_denominator
    tl.store(O_block_ptr, O_i.to(O.dtype), boundary_check=(0, 1))

    acc_L = _max + tl.torch(acc_denominator)
    tl.store(L_block_ptr, acc_L.to(L.dtype), boundary_check=(0, 1))