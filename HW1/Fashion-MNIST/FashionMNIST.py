import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# hyperparameters
NUM_CLASSES = 10
INPUT_SIZE = 28
HIDDEN_SIZE = 500
LEARNING_RATE = 0.01
BATCH_SIZE = 500
EPOCHS = 8
TRAIN_DATA_SIZE = 58000
VAL_DATA_SIZE = 2000


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Linear(9 * 4 * 4, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def get_data_loaders():
    # input and normalize data
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    train_data, val_data = random_split(train_data, [TRAIN_DATA_SIZE, VAL_DATA_SIZE])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    epoch_losses = []
    epoch_val_losses = []
    epoch_accuracies = []
    val_accuracies = []

    # Train model
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0
        total_val_loss = 0
        correct = 0
        total = 0

        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_correct, val_total = 0, 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                total_val_loss += val_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        # Record the average losses and accuracy for plotting
        epoch_losses.append(total_loss / len(train_loader))
        epoch_accuracies.append(100 * correct / total)
        epoch_val_losses.append(total_val_loss / len(val_loader))
        val_accuracies.append(100 * val_correct / val_total)

        print(
            f'Epoch [{epoch + 1}/{EPOCHS}], '
            f'Train Loss: {total_loss / len(train_loader):.4f}, '
            f'Accuracy: {100 * correct / total:.2f}%, '
            f'Val Loss: {total_val_loss / len(val_loader):.4f}, '
            f'Val Accuracy: {100 * val_correct / val_total:.2f}%')
        scheduler.step()

    return epoch_losses, epoch_val_losses, epoch_accuracies, val_accuracies


def evaluate_model(model, loader, confusion=False):
    model.eval()
    all_predicted = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    if confusion:
        cm = confusion_matrix(all_targets, all_predicted)
        return cm
    else:
        correct = sum(p == t for p, t in zip(all_predicted, all_targets))
        return 100 * correct / len(all_targets)


if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the ConvNet
    model = ConvNet().to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # update learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    # Load data
    train_loader, val_loader, test_loader = get_data_loaders()

    # Train the model and collect the loss and accuracy metrics
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler)

    # Plot the training and validation loss
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot the training history
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Evaluate the model on test data
    test_accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Plot the confusion matrix
    conf_matrix = evaluate_model(model, test_loader, confusion=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
