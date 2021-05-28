import torch

from torch import nn
from torch.nn import functional as F
# -------------------------------------------------------------------------------------
  # Each model has two parameters
  # use_bn ... whether the model uses batch normalisation
  # dropout ... drop rate
  # Batch normalisation layers are placed after each convolutional or linear layer (except the final classification layer)
  # Dropout layers are placed after linear layer, except the final classification layer
  # -------------------------------------------------------------------------------------

# LeNet-like architecture, kernel sizes modified to suit the input size
class BaseNet(nn.Module):
    def __init__(self, batch_normalization = False, dropout = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm1d(200)
        self.batch_normalization = batch_normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = F.max_pool2d(self.conv1(x), kernel_size = 2)
        if self.batch_normalization: y = self.bn1(y)
        y = F.relu(y)
        y = F.max_pool2d(self.conv2(y), kernel_size = 2)
        if self.batch_normalization: y = self.bn2(y)
        y = F.relu(y)
        y = self.fc1(y.view(-1, 256))
        if self.batch_normalization: y = self.bn3(y)
        y = self.fc2(self.dropout(F.relu(y)))
        return y


# Model with auxiliary loss
class BaseNetAux(nn.Module):
    def __init__(self, batch_normalization = False, dropout = 0):
        # Initially mirrors BaseNet but splits the input into two sub-inputs and processes each independently to get the two digit predictions
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size = 3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1_1 = nn.Linear(256, 200)
        self.fc1_2 = nn.Linear(256, 200)
        self.fc2_1 = nn.Linear(200, 10)
        self.fc2_2 = nn.Linear(200, 10)
        # Then combine the two outputs from the digit classification layers, add one hidden layer and a layer for the final the Boolean prediction
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 2)

        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm1d(200)
        self.bn3_2 = nn.BatchNorm1d(200)
        self.bn4_1 = nn.BatchNorm1d(10)
        self.bn4_2 = nn.BatchNorm1d(10)
        self.bn5 = nn.BatchNorm1d(200)
        self.batch_normalization = batch_normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y_1 = F.max_pool2d(self.conv1_1(x[:, 0].view(-1, 1, 14, 14)), kernel_size = 2)
        y_2 = F.max_pool2d(self.conv1_2(x[:, 1].view(-1, 1, 14, 14)), kernel_size = 2)
        if self.batch_normalization:
            y_1 = self.bn1_1(y_1)
            y_2 = self.bn1_2(y_2)

        y_1 = F.max_pool2d(self.conv2_1(F.relu(y_1)), kernel_size = 2)
        y_2 = F.max_pool2d(self.conv2_2(F.relu(y_2)), kernel_size = 2)
        if self.batch_normalization:
            y_1 = self.bn2_1(y_1)
            y_2 = self.bn2_2(y_2)

        y_1 = self.fc1_1(F.relu(y_1).view(-1, 256))
        y_2 = self.fc1_1(F.relu(y_2).view(-1, 256))
        if self.batch_normalization:
            y_1 = self.bn3_1(y_1)
            y_2 = self.bn3_2(y_2)
        # Introduces auxiliary loss
        y_1_loss = self.fc2_1(self.dropout(F.relu(y_1)))
        y_2_loss = self.fc2_2(self.dropout(F.relu(y_2)))
        y_1 = y_1_loss
        y_2 = y_2_loss
        if self.batch_normalization:
            y_1 = self.bn4_1(y_1_loss)
            y_2 = self.bn4_2(y_2_loss)

        y = self.fc3(torch.cat((self.dropout(F.relu(y_1)), self.dropout(F.relu(y_2))), 1))
        if self.batch_normalization: y = self.bn5(y)
        y = self.fc4(self.dropout(F.relu(y)))
        return (y, y_1_loss, y_2_loss)

# Model with weight sharing
# Mirrors model with auxiliary loss (BaseNetAux) but uses the same weights for both sub-inputs
class BaseNetWeightShare(nn.Module):
    def __init__(self, batch_normalization = False, dropout = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 2)

        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm1d(200)
        self.bn3_2 = nn.BatchNorm1d(200)
        self.bn4_1 = nn.BatchNorm1d(10)
        self.bn4_2 = nn.BatchNorm1d(10)
        self.bn5 = nn.BatchNorm1d(200)

        self.batch_normalization = batch_normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y_1 = F.max_pool2d(self.conv1(x[:, 0].view(-1, 1, 14, 14)), kernel_size = 2)
        y_2 = F.max_pool2d(self.conv1(x[:, 1].view(-1, 1, 14, 14)), kernel_size = 2)
        if self.batch_normalization:
            y_1 = self.bn1_1(y_1)
            y_2 = self.bn1_2(y_2)

        y_1 = F.max_pool2d(self.conv2(F.relu(y_1)), kernel_size = 2)
        y_2 = F.max_pool2d(self.conv2(F.relu(y_2)), kernel_size = 2)
        if self.batch_normalization:
            y_1 = self.bn2_1(y_1)
            y_2 = self.bn2_2(y_2)

        y_1 = self.fc1(F.relu(y_1).view(-1, 256))
        y_2 = self.fc1(F.relu(y_2).view(-1, 256))
        if self.batch_normalization:
            y_1 = self.bn3_1(y_1)
            y_2 = self.bn3_2(y_2)

        y_1 = self.fc2(self.dropout(F.relu(y_1)))
        y_2 = self.fc2(self.dropout(F.relu(y_2)))
        if self.batch_normalization:
            y_1 = self.bn4_1(y_1)
            y_2 = self.bn4_2(y_2)

        y = self.fc3(torch.cat((self.dropout(F.relu(y_1)), self.dropout(F.relu(y_2))), 1))
        if self.batch_normalization: y = self.bn5(y)
        y = self.fc4(self.dropout(F.relu(y)))
        return y

# Model with weight sharing and with auxiliary loss
class BaseNetWeightShareAux(nn.Module):
    def __init__(self, batch_normalization = False, dropout = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 2)

        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.bn3_1 = nn.BatchNorm1d(200)
        self.bn3_2 = nn.BatchNorm1d(200)
        self.bn4_1 = nn.BatchNorm1d(10)
        self.bn4_2 = nn.BatchNorm1d(10)
        self.bn5 = nn.BatchNorm1d(200)

        self.batch_normalization = batch_normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y_1 = F.max_pool2d(self.conv1(x[:, 0].view(-1, 1, 14, 14)), kernel_size = 2)
        y_2 = F.max_pool2d(self.conv1(x[:, 1].view(-1, 1, 14, 14)), kernel_size = 2)
        if self.batch_normalization:
            y_1 = self.bn1_1(y_1)
            y_2 = self.bn1_2(y_2)

        y_1 = F.max_pool2d(self.conv2(F.relu(y_1)), kernel_size = 2)
        y_2 = F.max_pool2d(self.conv2(F.relu(y_2)), kernel_size = 2)
        if self.batch_normalization:
            y_1 = self.bn2_1(y_1)
            y_2 = self.bn2_2(y_2)

        y_1 = self.fc1(F.relu(y_1).view(-1, 256))
        y_2 = self.fc1(F.relu(y_2).view(-1, 256))
        if self.batch_normalization:
            y_1 = self.bn3_1(y_1)
            y_2 = self.bn3_2(y_2)
        # Introduces auxiliary loss
        y_1_loss = self.fc2(self.dropout(F.relu(y_1)))
        y_2_loss = self.fc2(self.dropout(F.relu(y_2)))
        y_1 = y_1_loss
        y_2 = y_2_loss
        if self.batch_normalization:
            y_1 = self.bn4_1(y_1_loss)
            y_2 = self.bn4_2(y_2_loss)

        y = self.fc3(torch.cat((self.dropout(F.relu(y_1)), self.dropout(F.relu(y_2))), 1))
        if self.batch_normalization: y = self.bn5(y)
        y = self.fc4(self.dropout(F.relu(y)))
        return (y, y_1_loss, y_2_loss)
