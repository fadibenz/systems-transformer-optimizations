import triton
import triton.language as tl

@triton.jit
def flash_bwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, dO_ptr,
        L_ptr,
        dQ_ptr, dK_ptr, dV_ptr,

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
):
    seq_len_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (seq_len_index * K_TILE_SIZE, 0),
        block_shape= (K_TILE_SIZE, D),
        order=(1, 0)
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(seq_len_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(seq_len_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(seq_len_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(Q_TILE_SIZE,),
        block_shape=(seq_len_index * Q_TILE_SIZE,),
        order=(0,)
    )

    K_j = tl.load(K_block_ptr, boundary_check= (0, 1), padding_option="zero")
    V_j = tl.load(V_block_ptr, boundary_check= (0, 1), padding_option="zero")

    dK_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_j = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr, boundary_check= (0, 1), padding_option="zero")
        O_i = tl.load(O_block_ptr, boundary_check= (0, 1), padding_option="zero")
        L_i = tl.load(L_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_i = tl.load(dO_block_ptr, boundary_check= (0, 1), padding_option="zero")

        D_i = tl.sum(dO_i * O_i, axis=-1)

        S_i = tl.dot( Q_i, tl.trans(K_j)) * scale
        P_i = tl.exp(S_i - L_i[:, None])
        dV_j += tl.dot(tl.trans(P_i), dO_i)

        dP_i = tl.dot(dO_i, tl.trans(V_j))

        dS_i = P_i * (dP_i - D_i[:, None]) * scale

        tl.atomic_add(dQ_block_ptr, tl.dot(dS_i, K_j))
        dK_j += tl.trans(dS_i, Q_i)

        Q_block_ptr = tl.advance(Q_block_ptr, (Q_TILE_SIZE, 0))
        O_block_ptr = tl.advance(O_block_ptr, (Q_TILE_SIZE, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Q_TILE_SIZE, 0))

        dQ_block_ptr = tl.advance(dQ_block_ptr, (Q_TILE_SIZE, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Q_TILE_SIZE, 0))

    tl.store(dK_block_ptr, dK_j, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_j, boundary_check=(0, 1))
