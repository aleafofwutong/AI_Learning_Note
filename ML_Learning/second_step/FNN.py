import torch.nn as nn
import torch
import torchvision

class FNN(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], output_size=10):
        super(FNN, self).__init__()
        layers = []
        current_input_size = input_size

        # Add multiple hidden layers with BatchNorm and Dropout
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Dropout with 50% rate
            current_input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_input_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        out = self.model(x)
        return out
    
model = FNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define transformations for the dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # Normalize to mean=0.1307, std=0.3081
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Compute loss
        loss = criterion(output, target)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print(f'Epoch {epoch}, Average Loss: {running_loss / len(train_loader):.6f}')

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

accuracy = 100. * correct / total
print(f'\nTest Accuracy: {correct}/{total} ({accuracy:.2f}%)')


