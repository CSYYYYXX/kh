import paddle.tensor as tensor
import paddle.fluid.dygraph as dygraph
import paddle.fluid as fluid
import paddle.nn as nn

class TriangularCausalMask:
    def __init__(self, B, L):
        mask_shape = (B, 1, L, L)
        self._mask = dygraph.to_variable(tensor.triu(fluid.layers.ones(mask_shape, dtype=bool), diagonal=1).numpy())

    @property
    def mask(self):
        return self._mask

class ProbMask:
    def __init__(self, B, H, L, index, scores):
        mask = tensor.triu(fluid.layers.ones((L, scores.shape[-1]), dtype=bool), diagonal=1)
        mask_ex = tensor.broadcast_to(mask.unsqueeze(0).unsqueeze(0), (B, H, L, scores.shape[-1]))
        indicator = mask_ex[tensor.arange(B)[:, None, None],
                            tensor.arange(H)[None, :, None],
                            index, :]
        self._mask = indicator.view(scores.shape)

    @property
    def mask(self):
        return self._mask
