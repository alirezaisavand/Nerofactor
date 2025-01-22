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

        # Define layers
        for w, a in zip(widths, act):
            activation = None
            if isinstance(a, str):
                # Convert string activation to PyTorch equivalent
                activation = getattr(nn, a.capitalize(), None)
                if activation is None:
                    raise ValueError(f"Unsupported activation function: {a}")
                activation = activation()  # Instantiate activation
            layer = nn.Sequential(
                nn.Linear(w, w),  # Dense layer
                activation if activation is not None else nn.Identity()  # Activation or identity
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
