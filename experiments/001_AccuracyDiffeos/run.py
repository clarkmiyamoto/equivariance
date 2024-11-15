import argparse
import sys
sys.path.append('/vast/xj2173/diffeo/')
from utils.diffeo_container import sparse_diffeo_container, diffeo_container

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import v2

from models import ModelsWithIMAGENET1K_V1

def get_model(name):
  subset_of_models = ModelsWithIMAGENET1K_V1()
  model = subset_of_models.init_model(name).to(device)
  # ENV2 = tv.models.efficientnet_v2_s().to(device) # random initialization
  model.eval()
  for param in model.parameters():
      param.requires_grad = False
  return model