import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import sqrt

from utils.masking_ms import TriangularCausalMask, ProbMask
from utils.tools_ms import mask_fill, trans_shape

class FullAttention(nn.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(1 - attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)

        scores = paddle.matmul(queries.transpose([0, 2, 1, 3]), keys.transpose([0, 2, 3, 1]))
        if self.mask_flag:
            if attn_mask is None:
                mask_shape = (B, H, L, L)
                attn_mask = paddle.triu(paddle.ones(mask_shape, dtype='bool'), diagonal=1)
            scores = paddle.where(attn_mask, scores, paddle.full_like(scores, -float('inf')))

        A = self.dropout(F.softmax(scale * scores, axis=-1))
        V = paddle.matmul(A, values.transpose([0, 2, 1, 3])).transpose([0, 2, 1, 3])

        if self.output_attention:
            return V, A
        else:
            return V, None

class ProbAttention(nn.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(1 - attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c * ln(L_q)
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = paddle.randint(0, L_K, (L_Q, sample_k))
        K_sample = paddle.gather(K_expand, index_sample, axis=3)
        Q_K_sample = paddle.matmul(Q.unsqueeze(-2), K_sample.transpose([0, 1, 3, 2])).squeeze(-2)
        # find the Top_k query with sparsity measurement
        M = paddle.argmax(Q_K_sample, axis=-1) - paddle.sum(Q_K_sample, axis=-1) / L_K
        M_top = paddle.topk(M, n_top, axis=-1, largest=False).indices
        # use the reduced Q to calculate Q_K
        Q_reduce = paddle.gather(Q, M_top, axis=2)
        Q_K = paddle.matmul(Q_reduce, K.transpose([0, 1, 3, 2]))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = paddle.mean(V, axis=-2)
            context = paddle.unsqueeze(V_sum, axis=-2).expand([B, H, L_Q, V_sum.shape[-1]])
        else:
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            context = paddle.cumsum(V, axis=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores)
            scores = paddle.where(attn_mask.mask, scores, paddle.full_like(scores, -float('inf')))
        attn = F.softmax(scores, axis=-1)
        context_in[paddle.arange(B)[:, None, None],
                   paddle.arange(H)[None, :, None],
                   index, :] = paddle.matmul(attn, V).astype(context_in.dtype)
        if self.output_attention:
            attns = paddle.ones([B, H, L_V, L_V]) / L_V
            attns[paddle.arange(B)[:, None, None], paddle.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose([0, 2, 1, 3])
        keys = keys.transpose([0, 2, 1, 3])
        values = values.transpose([0, 2, 1, 3])

        U_part = self.factor * paddle.ceil(paddle.log(L_K)).astype('int').item()
        u = self.factor * paddle.ceil(paddle.log(L_Q)).astype('int').item()

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_tops, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = 1. / sqrt(D)
        scores_top = scores_tops / sqrt(D)
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose([0, 2, 1, 3]), attn

class AttentionLayer(nn.Layer):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).reshape([B, L, H, -1])
        keys = self.key_projection(keys).reshape([B, S, H, -1])
        values = self.value_projection(values).reshape([B, S, H, -1])

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
        )
        if self.mix:
            out = out.transpose([0, 2, 1, 3])
        out = out.reshape([B, L, -1])
        return self.out_projection(out), attn

