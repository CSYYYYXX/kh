"""
cae-transformer model
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CaeEncoder(nn.Layer):
    """
    encoder net
    """
    def __init__(self, conv_kernel_size, maxpool_kernel_size, maxpool_stride, channels_encoder, channels_dense):
        super(CaeEncoder, self).__init__()
        self.conv1 = nn.Conv2D(channels_encoder[0], channels_encoder[1], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv2 = nn.Conv2D(channels_encoder[1], channels_encoder[2], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv3 = nn.Conv2D(channels_encoder[2], channels_encoder[3], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv4 = nn.Conv2D(channels_encoder[3], channels_encoder[4], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv5 = nn.Conv2D(channels_encoder[4], channels_encoder[5], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv6 = nn.Conv2D(channels_encoder[5], channels_encoder[6], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))

        self.max_pool2d = nn.MaxPool2D(kernel_size=maxpool_kernel_size, stride=maxpool_stride)

        self.relu = nn.ReLU()

        self.flatten = paddle.nn.Flatten()

        self.channels_decoder = channels_encoder



        self.dense1 = nn.Linear(channels_dense[0], channels_dense[1], weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.dense2 = nn.Linear(channels_dense[1], channels_dense[2], weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.dense3 = nn.Linear(channels_dense[2], channels_dense[3], weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))

    def forward(self, x):
        """
        encoder forward
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.conv6(x)
        x = self.relu(x)
        x = self.max_pool2d(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = F.relu(x)  # 设置激活函数
        x = self.dense2(x)
        x = F.relu(x)  # 设置激活函数
        x = self.dense3(x)
        return x


class CaeDecoder(nn.Layer):
    """
    decoder net
    """
    def __init__(self, data_dimension, conv_kernel_size, channels_decoder, channels_dense):
        super(CaeDecoder, self).__init__()
        self.dense1 = nn.Linear(channels_dense[3], channels_dense[2], weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.dense2 = nn.Linear(channels_dense[2], channels_dense[1], weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.dense3 = nn.Linear(channels_dense[1], channels_dense[0], weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv1 = nn.Conv2D(channels_decoder[0], channels_decoder[1], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv2 = nn.Conv2D(channels_decoder[1], channels_decoder[2], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv3 = nn.Conv2D(channels_decoder[2], channels_decoder[3], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv4 = nn.Conv2D(channels_decoder[3], channels_decoder[4], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv5 = nn.Conv2D(channels_decoder[4], channels_decoder[5], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv6 = nn.Conv2D(channels_decoder[5], channels_decoder[6], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))
        self.conv7 = nn.Conv2D(channels_decoder[6], channels_decoder[7], conv_kernel_size,
                               padding='same', weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()))

        self.relu = nn.ReLU()

        self.data_dimension = data_dimension

        self.channels_decoder = channels_decoder

    def forward(self, x):
        """
        decoder forward
        """
        x = self.dense1(x)
        x = F.relu(x)  # 设置激活函数
        x = self.dense2(x)
        x = F.relu(x)  # 设置激活函数
        x = self.dense3(x)

        x = paddle.reshape(x, (x.shape[0], self.channels_decoder[0],
                               round(pow(x.shape[-1]/self.channels_decoder[0], 0.5)), -1))

        x = self.conv1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(self.data_dimension[5], self.data_dimension[5]), mode='nearest')

        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(self.data_dimension[4], self.data_dimension[4]), mode='nearest')

        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(self.data_dimension[3], self.data_dimension[3]), mode='nearest')

        x = self.conv4(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(self.data_dimension[2], self.data_dimension[2]), mode='nearest')

        x = self.conv5(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(self.data_dimension[1], self.data_dimension[1]), mode='nearest')

        x = self.conv6(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(self.data_dimension[0], self.data_dimension[0]), mode='nearest')

        x = self.conv7(x)
        return x


class CaeNet(nn.Layer):
    """
    cae net
    """
    def __init__(self, data_dimension, conv_kernel, maxpool_kernel, maxpool_stride,
                 channels_encoder, channels_decoder, channels_dense):
        super(CaeNet, self).__init__()
        self.encoder = CaeEncoder(conv_kernel, maxpool_kernel, maxpool_stride, channels_encoder, channels_dense)
        self.decoder = CaeDecoder(data_dimension, conv_kernel, channels_decoder, channels_dense)

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x


class Lstm(nn.Layer):
    """
    lstm net
    """
    def __init__(self, latent_size, hidden_size, num_layers):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_size, hidden_size=hidden_size, num_layers=num_layers, direction="bidirectional")
        self.dense = nn.Linear(hidden_size, latent_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h0 = paddle.zeros((self.num_layers, x.shape[0], self.hidden_size), dtype='float32')
        c0 = paddle.zeros((self.num_layers, x.shape[0], self.hidden_size), dtype='float32')
        x, _ = self.lstm(x, (h0, c0))
        x = self.dense(x)
        return x
