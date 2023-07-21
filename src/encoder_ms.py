import paddle
import paddle.nn as nn
import paddle.tensor as tensor
from paddle import fluid

def trans_shape(trans_index, tensor):
    s = list(range(len(tensor.shape)))
    s[trans_index[0]], s[trans_index[1]] = s[trans_index[1]], s[trans_index[0]]
    return tuple(s)

class ConvLayer(nn.Layer):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1
        self.downConv = nn.Conv1D(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='reflect')
        self.norm = nn.BatchNorm2D(num_features=c_in, momentum=0.9)
        self.activation = nn.ELU()
        self.pad = nn.Pad2D(((0, 0), (0, 0), (1, 1)), mode="constant")
        self.maxPool = nn.MaxPool1D(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.downConv(x.transpose(0, 2, 1))
        x = fluid.layers.squeeze(self.norm(fluid.layers.unsqueeze(x, -1)))
        x = self.activation(x)
        x = self.maxPool(self.pad(x))
        x = x.transpose(trans_shape((1, 2), x))
        return x

class EncoderLayer(nn.Layer):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1D(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1D(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(trans_shape((-1, 1), y)))))
        y = self.dropout(self.conv2(y).transpose(trans_shape((-1, 1), y)))

        return self.norm2(x+y), attn

class Encoder(nn.Layer):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.LayerList(attn_layers)
        self.conv_layers = nn.LayerList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
