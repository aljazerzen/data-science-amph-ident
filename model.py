import torch
import torch.nn as nn
import torch.nn.functional as F
import os

PATH = './amphi_order_net.pth'

class AmphiOrderNet(nn.Module):
    def __init__(self):
        super(AmphiOrderNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 97 * 47, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 97 * 47)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AmphiNameNet(nn.Module):
    def __init__(self):
        super(AmphiNameNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 97 * 47, 250)
        self.fc2 = nn.Linear(250, 140)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 97 * 47)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def filepath(self):
        return './amphi_name_net.pht'

    def save(self):
        torch.save(self.state_dict(), self.filepath())

    def load(self):
        if os.path.isfile(self.filepath()):
            self.load_state_dict(torch.load(self.filepath()))
