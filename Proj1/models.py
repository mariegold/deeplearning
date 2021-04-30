import torch

from torch import nn
from torch.nn import functional as F

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, 200) 
        self.fc2 = nn.Linear(200, 2)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x


# Model with auxiliary loss
class BaseNetAux(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv1_2 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size = 3)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1_1 = nn.Linear(256, 200) 
        self.fc1_2 = nn.Linear(256, 200)  
        self.fc2_1 = nn.Linear(200, 10)
        self.fc2_2 = nn.Linear(200, 10)
        # Part above mirrors BaseNet but splits input and processes each independently to get the two digit predictions
        # Then combine the two, add one hidden layer and final layer to do the Boolean prediction
        self.fc3 = nn.Linear(20, 200)  
        self.fc4 = nn.Linear(200, 2)   
    
    def forward(self, x):
        x_1 = F.relu(F.max_pool2d(self.conv1_1(x[:, 0].view(-1, 1, 14, 14)), kernel_size = 2))
        x_2 = F.relu(F.max_pool2d(self.conv1_2(x[:, 1].view(-1, 1, 14, 14)), kernel_size = 2))
        x_1 = F.relu(F.max_pool2d(self.conv2_1(x_1), kernel_size = 2))
        x_2 = F.relu(F.max_pool2d(self.conv2_2(x_2), kernel_size = 2))
        x_1 = F.relu(self.fc1_1(x_1.view(-1, 256)))
        x_2 = F.relu(self.fc1_2(x_2.view(-1, 256)))
        x_1_loss = self.fc2_1(x_1)
        x_2_loss = self.fc2_2(x_2)
        x_1 = F.relu(x_1_loss)
        x_2 = F.relu(x_2_loss)
        x = F.relu(self.fc3(torch.cat((x_1, x_2), 1)))
        x = self.fc4(x)
        return (x, x_1_loss, x_2_loss)

# Model with weight sharing (siamese architecture)
# Mirrors model with auxiliary loss but use the same weights for both sub-inputs
class BaseNetWeightShare(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, 200)  
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 200)  
        self.fc4 = nn.Linear(200, 2)   
    
    def forward(self, x):
        x_1 = F.relu(F.max_pool2d(self.conv1(x[:, 0].view(-1, 1, 14, 14)), kernel_size = 2))
        x_2 = F.relu(F.max_pool2d(self.conv1(x[:, 1].view(-1, 1, 14, 14)), kernel_size = 2))
        x_1 = F.relu(F.max_pool2d(self.conv2(x_1), kernel_size = 2))
        x_2 = F.relu(F.max_pool2d(self.conv2(x_2), kernel_size = 2))
        x_1 = F.relu(self.fc1(x_1.view(-1, 256)))
        x_2 = F.relu(self.fc1(x_2.view(-1, 256)))
        x_1 = F.relu(self.fc2(x_1))
        x_2 = F.relu(self.fc2(x_2))
        x = F.relu(self.fc3(torch.cat((x_1, x_2), 1)))
        x = self.fc4(x)
        return x

# Model with weight sharing (siamese architecture) and with auxiliary loss
class BaseNetWeightShareAux(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, 200)  
        self.fc2 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(20, 200)  
        self.fc4 = nn.Linear(200, 2)   
    
    def forward(self, x):
        x_1 = F.relu(F.max_pool2d(self.conv1(x[:, 0].view(-1, 1, 14, 14)), kernel_size = 2))
        x_2 = F.relu(F.max_pool2d(self.conv1(x[:, 1].view(-1, 1, 14, 14)), kernel_size = 2))
        x_1 = F.relu(F.max_pool2d(self.conv2(x_1), kernel_size = 2))
        x_2 = F.relu(F.max_pool2d(self.conv2(x_2), kernel_size = 2))
        x_1 = F.relu(self.fc1(x_1.view(-1, 256)))
        x_2 = F.relu(self.fc1(x_2.view(-1, 256)))
        x_1_loss = self.fc2(x_1)
        x_2_loss = self.fc2(x_2)
        x_1 = F.relu(x_1_loss)
        x_2 = F.relu(x_2_loss)
        x = F.relu(self.fc3(torch.cat((x_1, x_2), 1)))
        x = self.fc4(x)
        return (x, x_1_loss, x_2_loss)        