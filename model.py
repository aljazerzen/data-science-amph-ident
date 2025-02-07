import torch
import torch.nn as nn
import torch.nn.functional as F
import os

PATH = './amphi_order_net.pth'

class ScaleLayer(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.alpha = nn.Parameter(torch.Tensor(torch.full((in_size,), 0.0)))
        self.beta = nn.Parameter(torch.Tensor(torch.full((in_size,), 0.0)))

    def forward(self, input):
        return self.alpha + input * self.beta

class BiasLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.randn([size]))

    def forward(self, input):
        return input + self.bias

class AmphiNameNet(nn.Module):
    
    def __init__(self):
        super(AmphiNameNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 66, kernel_size=3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(66)
        # self.scale1 = ScaleLayer(128)

        
        self.conv2 = nn.Conv2d(66, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        # self.scale2 = ScaleLayer(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        # self.scale3 = ScaleLayer(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        # self.scale4 = ScaleLayer(128)

        
        self.conv5 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(192)
        # self.scale5 = ScaleLayer(128)

        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv6 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(192)
        # self.scale6 = ScaleLayer(64)
        
        self.conv7 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(192)
        # self.scale7 = ScaleLayer(64)
        
        self.conv8 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn8 = torch.nn.BatchNorm2d(192)
        # self.scale8 = ScaleLayer(64)
        
        self.conv9 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn9 = torch.nn.BatchNorm2d(192)
        # self.scale9 = ScaleLayer(64)
        
        
        self.conv10 = nn.Conv2d(192, 288, kernel_size=3, padding=1)
        self.bn10 = torch.nn.BatchNorm2d(288)
        # self.scale10 = ScaleLayer(64)

        
        self.conv11 = nn.Conv2d(288, 288, kernel_size=3, padding=1)
        self.bn11 = torch.nn.BatchNorm2d(288)
        # self.scale11 = ScaleLayer(32)
        
        self.conv12 = nn.Conv2d(288, 355, kernel_size=3, padding=1)
        self.bn12 = torch.nn.BatchNorm2d(355)
        # self.scale12 = ScaleLayer(32)
        
        self.conv13 = nn.Conv2d(355, 432, kernel_size=3, padding=1)
        self.bn13 = torch.nn.BatchNorm2d(432)
        # self.scale13 = ScaleLayer(32)

        self.fc1 = nn.Linear(108 * 32 * 32, 185)
        self.enable_dropout()

    def enable_dropout(self):
        self.dropout = nn.Dropout(0)
    
    def disable_dropout(self):
        self.dropout = nn.Dropout(0)

    def modify(self):
        for param in self.parameters():
            param.requires_grad = False
        
        self.fc1 = nn.Linear(108 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, 185)

    def forward(self, x):
        
        x = F.relu(self.dropout(self.bn1(self.conv1(x))))

        x = F.relu(self.dropout(self.bn2(self.conv2(x))))
        x = F.relu(self.dropout(self.bn3(self.conv3(x))))
        x = F.relu(self.dropout(self.bn4(self.conv4(x))))
        x = F.relu(self.dropout(self.bn5(self.conv5(x))))
        x = self.dropout(self.pool(x))
        
        x = F.relu(self.dropout(self.bn6(self.conv6(x))))
        x = F.relu(self.dropout(self.bn7(self.conv7(x))))
        x = F.relu(self.dropout(self.bn8(self.conv8(x))))
        x = F.relu(self.dropout(self.bn9(self.conv9(x))))

        x = F.relu(self.dropout(self.bn10(self.conv10(x))))
        x = self.dropout(self.pool(x))

        x = F.relu(self.dropout(self.bn11(self.conv11(x))))
        x = F.relu(self.dropout(self.bn12(self.conv12(x))))
        x = F.relu(self.dropout(self.bn13(self.conv13(x))))
        x = self.dropout(self.pool(x))
        
        x = x.view(-1, 108 * 32 * 32)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc1(x)
        return x

    def filepath(self):
        return './amphi_name_net.pth'

    def save(self):
        torch.save(self.state_dict(), self.filepath())

    def load(self, device):
        if os.path.isfile(self.filepath()):
            self.load_state_dict(torch.load(self.filepath(), map_location=torch.device(device)))
        else:
            print("no saved net found, starting from scratch")
        self.to(device)



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
    def filepath(self):
        return './amphi_order_net.pth'

    def save(self):
        torch.save(self.state_dict(), self.filepath())

    def load(self):
        if os.path.isfile(self.filepath()):
            self.load_state_dict(torch.load(self.filepath()))
        else:
            print("no saved net found, starting from scratch")