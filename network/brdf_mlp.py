import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SeqNetwork(Network):
    def __init__(self):
        super(SeqNetwork, self).__init__()
        self.sequential = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.sequential(x)

class MLPNetwork(SeqNetwork):
    def __init__(self, widths, act=None, skip_at=None):
        super(MLPNetwork, self).__init__()
        self.layers = []
        self.skip_at = skip_at

        if act is None:
            act = [None] * len(widths)

        assert len(act) == len(widths), "If not `None`, `act` must have the same length as `widths`"

        for w, a in zip(widths, act):
            activation = self._get_activation(a)
            layer = nn.Linear(w, w if activation is None else w + activation.in_features)
            self.layers.append(layer)

    def forward(self, x):
        if self.skip_at is None:
            return super(MLPNetwork, self).forward(x)

        x_ = x.clone()
        for i, layer in enumerate(self.layers):
            y = layer(x_)
            if i in self.skip_at:
                y = torch.cat((y, x), dim=-1)
            x_ = y
        return y

    @staticmethod
    def _get_activation(act):
        if isinstance(act, str):
            if act.lower() == "relu":
                return nn.ReLU()
            elif act.lower() == "sigmoid":
                return nn.Sigmoid()
            elif act.lower() == "tanh":
                return nn.Tanh()
        return None
