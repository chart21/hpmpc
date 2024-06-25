import torch
import torch.nn as nn
import torchvision.models as models

# Define the models

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(800, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=9),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if num_classes == 10:
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 10),
            )
        elif num_classes == 200:
            self.fc_layers = nn.Sequential(
                nn.AvgPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 200),
            )
        elif num_classes == 1000:
            self.fc_layers = nn.Sequential(
                nn.AvgPool2d(kernel_size=4),
                nn.Flatten(),
                nn.Linear(9216, 4096),
                nn.ReLU(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 1000),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.fc_layers(x)
        return x

class AlexNet_32(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_32, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=56, stride=1, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')

        self.conv2 = nn.Conv2d(in_channels=56, out_channels=128, kernel_size=5, stride=1, padding='same')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding='same')
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding='same')
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.relu5 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding='same')
        
        #Flatten
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(128*4*4, 2048)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(2048, 2048)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x

        

            

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.vgg = models.vgg16(pretrained=False)
        
        self.vgg.features[4] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.vgg.features[9] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,  ceil_mode=False)
        self.vgg.features[16] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,  ceil_mode=False)
        self.vgg.features[23] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,  ceil_mode=False)
        self.vgg.features[30] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,  ceil_mode=False)

        if num_classes == 10:
            self.vgg.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
        elif num_classes == 200:
            self.vgg.classifier = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 200),
            )
        elif num_classes == 1000:
            self.vgg.classifier = nn.Sequential(
                nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(4608, 4096),
                nn.Linear(4096, 4096),
                nn.Linear(4096, 1000)
            )

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.classifier(x)
        return x

class block(nn.Module):
    def _print_layer_params(self, layer, layer_name):
        total_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"{layer_name}: {total_params} parameters")
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x



class ResNet(nn.Module):

    def _print_layer_params(self, layer, layer_name):
        total_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"{layer_name}: {total_params} parameters")

    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)  # Changed to AvgPool2d with no padding

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool2 = nn.AvgPool2d(kernel_size=1, stride=1)  # Changed to regular AvgPool2d
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool1(x)  # Changed to AvgPool2d

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final layers
        x = self.avgpool2(x)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.fc(x)

        return x



# class ResNet(nn.Module):

#     def _print_layer_params(self, layer, layer_name):
#         total_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
#         print(f"{layer_name}: {total_params} parameters")

#     def __init__(self, block, layers, image_channels, num_classes):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(
#             image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         # Essentially the entire ResNet architecture are in these 4 lines below
#         self.layer1 = self._make_layer(
#             block, layers[0], intermediate_channels=64, stride=1
#         )
#         self.layer2 = self._make_layer(
#             block, layers[1], intermediate_channels=128, stride=2
#         )
#         self.layer3 = self._make_layer(
#             block, layers[2], intermediate_channels=256, stride=2
#         )
#         self.layer4 = self._make_layer(
#             block, layers[3], intermediate_channels=512, stride=2
#         )

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * 4, num_classes)

#     def forward(self, x):
#         self._print_layer_params(self.conv1, 'conv1')
#         x = self.conv1(x)
#         self._print_layer_params(self.bn1, 'bn1')
#         x = self.bn1(x)
#         self._print_layer_params(self.relu, 'relu')
#         x = self.relu(x)
#         self._print_layer_params(self.maxpool, 'maxpool')
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         self._print_layer_params(self.avgpool, 'avgpool')
#         x = self.avgpool(x)
#         x = x.reshape(x.shape[0], -1)
#         self._print_layer_params(self.fc, 'fc')
#         x = self.fc(x)

#         return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

    def print_layers_and_params(self):
        for name, module in self.named_modules():
            total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if total_params > 0:
                print(f"{name}: {total_params} parameters")
            else:
                print(f"{name}: No parameters")


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Define transformations for the datasets
transform_mnist = transforms.Compose([
    transforms.Resize((32, 32)), # Resize to fit LeNet architecture
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Define transformations for the standard MNIST dataset (28x28)
transform_mnist_standard = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the standard MNIST dataset
def load_standard_mnist():
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform_mnist_standard, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform_mnist_standard)
    num_classes = 10
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, num_classes


transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_imagenet_tiny = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to fit AlexNet and VGG16 architectures
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a function to load the datasets
def load_dataset(dataset_name):
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform_mnist, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform_mnist)
        num_classes = 10
    elif dataset_name == "CIFAR-10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_cifar10, download=True)
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_cifar10)
        num_classes = 10
    elif dataset_name == "IMAGE-NET Tiny":
        # NOTE: The real ImageNet dataset is huge, so I'll assume you're referring to a smaller version called Tiny ImageNet.
        # However, I don't have direct access to download Tiny ImageNet from here, so I'll provide the setup but can't run it directly.
        train_dataset = datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform_imagenet_tiny)
        test_dataset = datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform_imagenet_tiny)
        num_classes = 200
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, num_classes

import torch.optim as optim

def train_and_evaluate(model, train_loader, test_loader, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

# Load CIFAR-10 dataset
train_loader, test_loader, num_classes = load_dataset("CIFAR-10")

# Train and evaluate VGG-16
#model = VGG16(num_classes=10)
#print(model.vgg.features)
#print(model.vgg.classifier)
#load model from pth file, map location is for cpu

def load_and_evaluate(path, model, test_loader):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy of the model on the test set: {accuracy:.2f}%")



# accuracy_vgg = train_and_evaluate(vgg, train_loader, test_loader, num_epochs=200)
# accuracy_vgg

# # Train and evaluate AlexNet
# alex = AlexNet()
# accuracy_alex = train_and_evaluate(alex, train_loader, test_loader, num_epochs=20)
# accuracy_alex



# # Load standard MNIST dataset
#train_loader_mnist, test_loader_mnist, _ = load_standard_mnist()

# # Train and evaluate the original LeNet model on standard MNIST dataset
# lenet_original = LeNet()
# accuracy_lenet_original = train_and_evaluate(lenet_original, train_loader_mnist, test_loader_mnist, num_epochs=20)

# accuracy_lenet_original
import numpy as np
import idx2numpy

def save_dataset2():
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Convert data to numpy arrays
    train_images = np.array([img[0].numpy() for img in train_dataset])
    train_labels = np.array([img[1] for img in train_dataset])
    test_images = np.array([img[0].numpy() for img in test_dataset])
    test_labels = np.array([img[1] for img in test_dataset])

    # Step 2: Save in IDX format
    # idx2numpy.convert_to_file('./data/train-images-idx3-ubyte', train_images)
    # idx2numpy.convert_to_file('./data/train-labels-idx1-ubyte', train_labels)
    idx2numpy.convert_to_file('./data/MNIST-test-images-idx3-ubyte', test_images)
    idx2numpy.convert_to_file('./data/MNIST-test-labels-idx1-ubyte', test_labels)



def save_dataset():
    # Define transformations: Convert PIL images to numpy arrays of type float32, and then normalize
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10 mean and std
    #     transforms.Lambda(lambda x: x.numpy())
    # ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
    #     transforms.Lambda(lambda x: x.numpy())

    # ])

    # Download datasets
    # train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    # test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform_mnist_standard, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform_mnist_standard, download=True)

    # Extract numpy arrays
    # train_images = np.array([image for image, _ in train_dataset], dtype=np.float32)
    # train_labels = np.array([label for _, label in train_dataset], dtype=np.uint8)
    test_images = np.array([image.numpy().reshape(-1) for image, _ in test_dataset], dtype=np.float32)
    test_images = np.array([image for image, _ in test_dataset], dtype=np.float32)
    test_labels = np.array([label for _, label in test_dataset], dtype=np.uint8)

    # Ensure data is in correct shape and type
    # Images: (n_samples, channels, height, width) --> (n_samples, channels * height * width)
    # train_images = train_images.reshape(len(train_dataset), -1)
    test_images = test_images.reshape(len(test_dataset), -1)

    # Save in IDX format
    #idx2numpy.convert_to_file('./data/train-images.idx3-ubyte', train_images)
    #idx2numpy.convert_to_file('./data/train-labels.idx1-ubyte', train_labels)
    idx2numpy.convert_to_file('./data/MNIST_t10k-images.idx3-ubyte', test_images)
    idx2numpy.convert_to_file('./data/MNIST_t10k-labels.idx1-ubyte', test_labels)

    print("Datasets saved in IDX format!")

import struct
# Function to export dataset
def export_dataset(dataset, images_filename, labels_filename):
    with open(images_filename, 'wb') as img_file, open(labels_filename, 'wb') as lbl_file:
        for image, label in dataset:
            # Convert image to numpy array, ensure it's float32, and flatten
            img_data = image.numpy().astype(np.float32).flatten()

            # Write the flattened image data to the binary file
            img_file.write(struct.pack('f' * len(img_data), *img_data))

            # Ensure label is an int and write to the binary file
            lbl_data = np.array(label).astype(np.uint32)
            lbl_file.write(struct.pack('i', lbl_data))


# def save_weights_compatible_with_cpp(model, filepath):
#     with open(filepath, 'wb') as f:
#         # First, write the total number of parameters
#         total_params = sum(p.numel() for p in model.parameters())
#         f.write(np.array([total_params], dtype=np.int32).tobytes())

#         # Now, write each parameter as a flat array of floats
#         for param_tensor in model.parameters():
#             # Ensure it's a CPU tensor and then convert to numpy
#             param_numpy = param_tensor.data.cpu().numpy().ravel().astype(np.float32)
#             f.write(param_numpy.tobytes())

# def save_weights_compatible_with_cpp(model, filepath):
#     with open(filepath, 'wb') as f:
#         # Collect all parameters and necessary buffers
#         all_params = []
#         for name, param in model.named_parameters():
#             all_params.append(param.data.cpu().numpy().ravel())
#         for name, buffer in model.named_buffers():
#             if 'running_mean' in name or 'running_var' in name:
#                 all_params.append(buffer.cpu().numpy().ravel())

#         # Flatten all parameters and convert to numpy array
#         all_params_flat = np.concatenate(all_params).astype(np.float32)

#         # Write total number of parameters and then write the parameters
#         f.write(np.array([all_params_flat.size], dtype=np.int32).tobytes())
#         f.write(all_params_flat.tobytes())


def save_weights_compatible_with_cpp(model, filepath):
    with open(filepath, 'wb') as f:
        all_params = []

        # Process each module according to its type
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())
            elif isinstance(module, torch.nn.Conv2d):
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                all_params.append(module.running_mean.cpu().numpy().ravel())
                all_params.append(module.running_var.cpu().numpy().ravel())
                all_params.append(module.weight.data.cpu().numpy().ravel())
                all_params.append(module.bias.data.cpu().numpy().ravel())

        # Flatten all parameters and convert to numpy array
        all_params_flat = np.concatenate(all_params).astype(np.float32)

        # Write total number of parameters and then write the parameters
        f.write(np.array([all_params_flat.size], dtype=np.int32).tobytes())
        f.write(all_params_flat.tobytes())

def print_model_layers(model):
    layer_counter = 0

    for module in model.modules():
        layer_type = None

        # Check the type of each module
        if isinstance(module, nn.Conv2d):
            layer_type = "CONV2D"
        elif isinstance(module, nn.BatchNorm2d):
            layer_type = "BATCHNORM2D"
        # elif isinstance(module, nn.ReLU) or isinstance(module, nn.LeakyReLU) or isinstance(module, nn.PReLU):
        #     layer_type = "ACTIVATION"
        elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
            layer_type = "MAXPOOL2D"
        # elif isinstance(module, nn.AdaptiveAvgPool2d):
        #     layer_type = "ADAPTIVEAVGPOOL2D"
        elif isinstance(module, nn.Linear):
            layer_type = "LINEAR"
        # elif isinstance(module, nn.Flatten):
        #     layer_type = "FLATTEN"
        # Add more layer types as needed

        # Print the layer information
        if layer_type:
            print(f"Layer: {layer_counter}, Layer Type: {layer_type}")
            layer_counter += 1


def save_and_print_model_layers(model, filepath):
    with open(filepath, 'wb') as f:
        all_params = []
        layer_counter = 0

        for module in model.modules():
            layer_type = None

            if isinstance(module, torch.nn.Linear):
                layer_type = "LINEAR"
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())
            elif isinstance(module, torch.nn.Conv2d):
                layer_type = "CONV2D"
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                if isinstance(module, torch.nn.BatchNorm1d):
                    layer_type = "BATCHNORM1D"
                elif isinstance(module, torch.nn.BatchNorm2d):
                    layer_type = "BATCHNORM2D"
                all_params.append(module.running_mean.cpu().numpy().ravel())
                all_params.append(module.running_var.cpu().numpy().ravel())
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())

            if layer_type:
                print(f"Layer: {layer_counter}, Layer Type: {layer_type}")
                layer_counter += 1

        # Flatten all parameters and convert to numpy array
        all_params_flat = np.concatenate(all_params).astype(np.float32)

        # Write total number of parameters and then write the parameters
        f.write(np.array([all_params_flat.size], dtype=np.int32).tobytes())
        f.write(all_params_flat.tobytes())

# Example Usage
# model = YourPyTorchModel()
# save_and_print_model_layers(model, 'path_to_save_weights')

# Save the weights of the LeNet model in a format compatible with the C++ code
model = ResNet50(num_classes=10)
# model.load_state_dict(torch.load('./ResNet50Custom.pth', map_location=torch.device('cpu')))
# model.eval()
load_and_evaluate('./Restnet50_custom2.pth', model, test_loader)
# save_weights_compatible_with_cpp(model, "./data/ResNet50_custom.bin")
# model.eval()
#dummy inference
# dummy_input = torch.randn(1, 3, 32, 32)  # Example input tensor
# output = model(dummy_input)
# model = VGG16(num_classes=10)
# model.load_state_dict(torch.load('./vgg_cifar.pth', map_location=torch.device('cpu')))

# print_model_layers(model)
# save_and_print_model_layers(model, './data/ResNet50_cifar.bin')

# print(model)
# save_weights_compatible_with_cpp(model, "./data/VGG16_cifar2.bin")
# save_weights_compatible_with_cpp(model, "./data/ResNet50_cifar.bin")
# model.eval()
# dummy_input = torch.randn(1, 3, 32, 32)  # Example input tensor
# output = model(dummy_input)
# print(model)
# model.print_layers_and_params()
# save_weights_compatible_with_cpp(model, "./data/ResNet50_cifar.bin")
# model = ResNet152(num_classes=10)
# print(model)
# model.load_state_dict(torch.load('./Resnet152_cifar.pth', map_location=torch.device('cpu')))
# save_weights_compatible_with_cpp(model, "./data/ResNet152_cifar.bin")
# save_dataset()
# save_dataset2()
# test_dataset = datasets.MNIST(root="./data", train=False, transform=transform_mnist_standard, download=True)
# export_dataset(test_dataset, './data/MNIST_t10k-images.bin', './data/MNIST_t10k-labels.bin')
# test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_cifar10, download=True)
# esexport_dataset(test_dataset, './data/cifar10-test-images.bin', './data/cifar10-test-labels.bin')
