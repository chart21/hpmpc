import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from models import LeNet, AlexNet, VGG16, ResNet50, ResNet152
# from utils import load_dataset, load_standard_mnist, transform_cifar10, transform_mnist_standard
import torch.optim as optim

def train_and_evaluate(model, train_loader, test_loader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=0.001, params=model.parameters())

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test set: {accuracy:.2f}%")
    return accuracy


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            print("label: ", labels, "predicted: ", predicted)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test set: {accuracy:.2f}%")

def train_test(model, train_loader, test_loader, num_epochs, model_name, dataset_name, transform="standard"):
    print(f"Training {model_name} on {dataset_name} with {transform} transform ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=0.001, params=model.parameters())
    best_accuracy = 0.0
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the model on the test set: {accuracy:.2f}%")
        if accuracy > best_accuracy:
            print(f"Accuracy improved, saving model")
            best_accuracy = accuracy
            torch.save(model.state_dict(), "./models/" + model_name + "_" + dataset_name + "_" + transform + "_best.pth")
            with open("./models/" + model_name + "_" + dataset_name + "_" + transform + "_best.txt", "w") as f:
                f.write(f"Epoch: {epoch + 1}/{num_epochs}\n")
                f.write(f"Best accuracy: {best_accuracy:.2f}%")
                
        
