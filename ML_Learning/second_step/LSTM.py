import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

class LSTMModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=2, output_size=10, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # 动态初始化隐藏状态
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # 动态初始化细胞状态
        out, _ = self.lstm(x, (h_0, c_0))  # LSTM前向传播
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

model = LSTMModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch学习率减半

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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

model.train()
epochs = 10
for epoch in range(1, epochs + 1):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.squeeze(1)  # 去掉通道维度，变成(batch_size, 28, 28)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    scheduler.step()  # 更新学习率
    print(f'Epoch {epoch}, Average Loss: {running_loss / len(train_loader):.6f}')

# 测试循环
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.squeeze(1)  # 去掉通道维度
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

accuracy = 100. * correct / total
print(f'\nTest Accuracy: {correct}/{total} ({accuracy:.2f}%)')

# 保存模型
torch.save(model.state_dict(), 'mnist_lstm.pt')