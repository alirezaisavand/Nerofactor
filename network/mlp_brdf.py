import torch
import torch.nn as nn
import torch.nn.functional as F

from network.seq_brdf import SequentialNetwork

class MLPNetwork(SequentialNetwork):
    def __init__(self, widths, act=None, skip_at=None):
        super(MLPNetwork, self).__init__()
        depth = len(widths)

        if act is None:
            act = [None] * depth

        assert len(act) == depth, "If not `None`, `act` must have the same length as `widths`"

        activation_map = {
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'softmax': nn.Softmax,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'selu': nn.SELU,
            None: nn.Identity  # Handle case where no activation is specified
        }

        # Define layers
        for w, a in zip(widths, act):
            activation = None
            activation = activation_map.get(a.lower() if a else None, None)
            if activation is None:
                raise ValueError(f"Unsupported activation function: {a}")
            layer = nn.Sequential(
                nn.Linear(w, w),  # Dense layer
                activation()  # Instantiate activation
            )
            self.layers.append(layer)

        self.skip_at = skip_at

    def forward(self, x):
        if self.skip_at is None:
            return super().forward(x)

        # Handle skip connections
        x_ = x.clone()  # Make a copy of the input tensor
        for i, layer in enumerate(self.layers):
            y = layer(x_)
            if self.skip_at and i in self.skip_at:
                y = torch.cat((y, x), dim=-1)  # Concatenate input with output
            x_ = y
        return y
