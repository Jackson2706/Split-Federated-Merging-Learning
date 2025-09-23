import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dims=[128, 64], num_classes=10):
        super(MLP, self).__init__()
        layers = []

        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        self.prototype = nn.Linear(prev_dim, 192)
        self.predict = nn.Linear(prev_dim, num_classes)
        self.network = nn.Sequential(*layers)


    def forward_prototype(self, x):
        x = x.view(x.size(0), -1)
        x = self.network(x)
        return self.prototype(x)

    def forward(self, x):
        # Flatten MNIST images: [batch, 1, 28, 28] → [batch, 784]

        x = x.view(x.size(0), -1)
        x = self.network(x)
        return self.predict(x)