import paddle
import paddle.nn as nn
import math


class PositionalEmbedding(nn.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = paddle.zeros((max_len, d_model), dtype='float32')
        position = paddle.arange(0, max_len, dtype='float32').unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2, dtype='float32') * -(math.log(10000.0) / d_model))

        self.pe[:, 0::2] = paddle.sin(position * div_term)
        self.pe[:, 1::2] = paddle.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :, :x.shape[1]]


class TokenEmbedding(nn.Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1D(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding)
        for name, m in self.named_sublayers():
            if isinstance(m, nn.Conv1D):
                nn.initializer.KaimingNormal()(m.weight)


    def forward(self, x):
        x = self.tokenConv(x.transpose((0, 2, 1))).transpose((0, 2, 1))
        return x


class DataEmbedding(nn.Layer):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
