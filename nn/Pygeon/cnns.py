import torch
import torch.nn as nn
import torchvision.models as models

# Define the models

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
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
            nn.Linear(500, num_classes)
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


import torch.nn as nn

class AlexNet_32(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_32, self).__init__()
        self.conv1 = nn.Conv2d(3, 56, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(56, 128, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)  # Adjust the size according to your input image size
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(2048, 2048)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool3(self.relu5(self.conv5(x)))
        x = self.flatten(x)
        x = self.dropout1(self.relu6(self.fc1(x)))
        x = self.dropout2(self.relu7(self.fc2(x)))
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
            block(self.in_channels, intermediate_channels, identity_downsample, stride))
        
            # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)
        

    def __init__(self, block, layers, image_channels, num_classes, pool_option='max'):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        if pool_option == 'max':
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pool_option == 'avg':
            self.maxpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Changed to MaxPool2d
        # self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=0)  # Changed to AvgPool2d with no padding

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        # self.avgpool2 = nn.AvgPool2d(kernel_size=1, stride=1)  # Changed to regular AvgPool2d
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.avgpool1(x)  # Changed to AvgPool2d
        x = self.maxpool1(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Final layers
        x = self.avgpool2(x)
        # x = x.reshape(x.shape[0], -1)
        x = torch.flatten(x, 1)  # Flatten the output
        x = self.fc(x)

        return x

def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)

def ResNet18(img_channel=3, num_classes=1000):
    return ResNet(block, [2, 2, 2, 2], img_channel, num_classes)

def ResNet50_avg(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes, 'avg')

def ResNet101_avg(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes, 'avg')

def ResNet152_avg(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes, 'avg')

def ResNet18_avg(img_channel=3, num_classes=1000):
    return ResNet(block, [2, 2, 2, 2], img_channel, num_classes, 'avg')

