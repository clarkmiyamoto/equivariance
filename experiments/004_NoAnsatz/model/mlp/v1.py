import torch.nn as nn

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

    activation = nn.LeakyReLU(0.1)
    # Model
    self.encoder = nn.Sequential(
        nn.Linear(512, 1024),
        activation,

        nn.Linear(1024, 1024),
        activation,
        
        nn.Linear(1024, 1024),
        activation,

        nn.Linear(1024, 512),
        activation,

        nn.Linear(512, 512),
        activation,

        nn.Linear(512, 512),
        activation,

        nn.Linear(512, 512),
    )

  def forward(self, x):
    x = self.encoder(x)
    return x