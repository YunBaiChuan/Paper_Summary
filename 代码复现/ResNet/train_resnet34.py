import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from models.resnet34 import ResNet34

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010])
])

train_dataset = datasets.CIFAR10("data", train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10("data", train=False, download=True, transform=test_transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet34().to(device)

print(f"Using {device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

train_losses, train_accs = [], []
val_losses, val_accs = [], []

n_epochs = 20

for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0 
    total = correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total 
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    running_loss = 0.0
    total = correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = correct / total 
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch + 1} / {n_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.4f}")
    print(f"Epoch {epoch + 1} / {n_epochs} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_acc:.4f}")
    print()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, n_epochs + 1)
ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2, marker='s', markersize=4)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, train_accs, label='Train Accuracy', linewidth=2, marker='o', markersize=4)
ax2.plot(epochs, val_accs, label='Val Accuracy', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('res/ResNet34_training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ 结果已保存为 res/ResNet34_training_results.png")