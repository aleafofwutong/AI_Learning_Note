import torch.nn as nn
import torch
import torchvision

class Inception(nn.Module):
    def __init__(self,modules:list,concatenation_dim:int):
        super(Inception, self).__init__()
        self.modules = modules
        self.concatenation_dim = concatenation_dim
    def forward(self, x):
        output = []
        for module in self.modules:
            output.append(module(x))
        output = torch.cat(output, dim=self.concatenation_dim)
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostConv(nn.Module):
    """
    GhostConv
    param in_channels: 输入通道数
    param out_channels: 输出通道数
    param kernel_size: 卷积核大小
    param stride: 步长
    param padding: 填充
    param ratio: 比例
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=2):
        super(GhostConv, self).__init__()
        self.mid_channels = out_channels // ratio  
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.mid_channels,
            kernel_size=1, 
            stride=stride,
            padding=0, 
            bias=False 
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=self.mid_channels,
            out_channels=self.mid_channels,
            kernel_size=kernel_size,
            stride=1,  
            padding=padding,
            groups=self.mid_channels, 
            bias=False
        )

    def forward(self, x):
        # 生成固有特征图（shape: [B, mid_channels, H', W']）
        intrinsic = self.conv1(x)
        # 生成幽灵特征图（shape: [B, mid_channels, H', W']）
        ghost = self.conv2(intrinsic)
        # 通道维度拼接：固有特征图 + 幽灵特征图 → 恢复目标通道数（shape: [B, out_channels, H', W']）
        out = torch.cat([intrinsic, ghost], dim=1)
        return out


class GhostConvBN(nn.Module):
    """
    input_size: ()
    output_size: 

    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, ratio=2):
        super(GhostConvBN, self).__init__()
        self.mid_channels = out_channels // ratio
        # out_channels = mid_channels + mid_channels
        # so out_channel is a 2k number and k>0
        
        # 固有特征图分支：Conv1x1 + BN + ReLU
        self.intrinsic_branch = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)  # inplace=True节省内存
        )
        
        # 幽灵特征图分支：DepthConv3x3 + BN + ReLU
        self.ghost_branch = nn.Sequential(
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size, 1, padding, groups=self.mid_channels, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        intrinsic = self.intrinsic_branch(x)
        ghost = self.ghost_branch(intrinsic)
        return torch.cat([intrinsic, ghost], dim=1)


class GhostBottleneck(nn.Module):
    """
    幽灵瓶颈结构（G-bneck）：基于GhostConv构建，适配轻量网络（如GhostNet）
    借鉴ResNet残差块设计，包含shortcut分支（恒等映射或1×1卷积下采样）
    """
    def __init__(self, in_channels, out_channels, stride=1, ratio=2):
        super(GhostBottleneck, self).__init__()
        # 瓶颈结构：拓展通道 → 幽灵卷积特征提取 → 扩张通道
        self.ghost1 = GhostConvBN(in_channels, in_channels*2, ratio=ratio)  
        # make sure the main path downsamples when stride != 1 to match the shortcut
        self.ghost1 = GhostConvBN(in_channels, in_channels*2, stride=stride, ratio=ratio)
        self.ghost2 = GhostConvBN(in_channels*2, out_channels, stride=1, ratio=ratio)  
        
        # shortcut分支：当输入输出通道/尺寸不一致时，用1×1卷积对齐
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)  # 残差连接
        out = self.ghost1(x)
        out = self.ghost2(out)
        out += residual  # 残差相加
        out = F.relu(out)  # 最终激活
        # input: (batch_size, in_channels, H, W) -> output: (batch_size, out_channels, H, W)
        return out

class GoogleNet(nn.Module):
    def __init__(self):
        """
        GoogleNet architecture
        Description:
        Input: 28x28x1
        Output: 10
        Convolution Layer 1: 28x28x32
        Convolution Layer 2: 28x28x64
        Fully Connected Layer 1: 128
        Fully Connected Layer 2: 10
        Inspection1: 32
        Inspection2: 64
        """
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) 
        # convolution layer: 
        # input size:(input channel=1, output channel=32, kernel size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1) 
        # convolution layer:
        # input size:(input channel=32, output channel=64, kernel size=3, stride=1, padding=0)
        self.dropout1 = nn.Dropout2d(0.25) 
        # random dropout:
        # dropout rate=0.25 
        # inplace = False
        # use Dropout (not Dropout2d) for 2D tensors after Linear
        self.dropout2 = nn.Dropout(0.5) 
        # random dropout
        # dropout rate=0.5
        self.fc1 = nn.Linear(9216, 128) 
        # full connection layer for branch1: input size=9216 , output size=128
        # branch2 produces a much smaller flattened feature (e.g. 256), create separate fc
        self.fc1_b2 = nn.Linear(256, 128)
        # full connection layer for branch2: input size=256, output size=128
        self.fc2 = nn.Linear(128, 10)
        # full connection layer:
        # input size=128, output size=10
        self.inspection = nn.InstanceNorm2d(32)
        # instance normalization layer:
        # input size=32
        self.inspec2 = nn.InstanceNorm2d(64)
        # instance normalization layer:
        # input size=64
        self.Sequential1 = nn.Sequential(
             self.conv1, self.inspection,nn.ReLU(),
             self.conv2, self.inspec2,nn.ReLU(),
             nn.MaxPool2d(kernel_size=2, stride=2), 
             self.dropout1, 
               nn.Flatten(),
             self.fc1,
             nn.ReLU(),
             self.dropout2, 
             self.fc2
             )
        self.Sequential2 = nn.Sequential(# input size:channel=1, output size:channel=32, kernel size=3, stride=1, padding=0
            GhostBottleneck(1, 32, stride=2, ratio=2),
            GhostBottleneck(32, 64, stride=2, ratio=2),
            # input size:channel=32, output size:channel=64, kernel size=3, stride=1, padding=0
            GhostBottleneck(64, 128, stride=2, ratio=2),
            GhostBottleneck(128, 64, stride=2, ratio=2),
            # input size:channel=128, output size:channel=64, kernel size=3, stride=1, padding=0
            # 将两个通道合为一个通道
            self.dropout1,
            nn.Flatten(),
            self.fc1_b2,
            nn.ReLU(),
            self.dropout2,
            self.fc2
        )
        self.inception = Inception([self.Sequential1,self.Sequential2], concatenation_dim=1)

    def forward(self, x):
        """
        :param x: (batch_size, 1, 28, 28)
        :return: (batch_size, 10)
        """
        output = self.inception(x)
        return output
    
model = GoogleNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

model.train()
epochs = 100
for epoch in range(1, epochs + 1):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

torch.save(model.state_dict(), 'mnist_google_net.pt')
model2 = GoogleNet()
model2.to(device)
state_dict = torch.load("mnist_google_net.pt", map_location=device)
model2.load_state_dict(state_dict)
model2.eval()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model2(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
accuracy = 100. * correct / total
print('\nTest Accuracy: {}/{} ({:.2f}%)'.format(correct, total, accuracy))