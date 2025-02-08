import torch.nn as nn

class Model(nn.Module):
  def __init__(self, dropout_rate):
    super(Model, self).__init__()

    # Model
    self.encoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),

        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        
        nn.Linear(512, 512),
    )

  def forward(self, x):
    x = self.encoder(x)
    return x