import torch

from torch import nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue
from helpers import *

class OurNet(nn.Module):

    def __init__(self):
        super().__init__()
        # define components

    def forward(self, x):
        # define architecture
        return x

if __name__ == '__main__':
  n_pairs = 1000
  train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(n_pairs)

  # model = OurNet()
