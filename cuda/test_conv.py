import torch
import torch.nn as nn

# Define the convolutional layer with 1 input channel, 1 output channel, a 2x2 kernel, stride of 1, and no padding
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, bias=True)

# Define the input tensor and add batch and channel dimensions
input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Define the kernel tensor and adjust dimensions to match [out_channels, in_channels, height, width]
kernel_tensor = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Set the convolutional layer's weights
conv.weight.data = kernel_tensor

# Initialize the bias tensor. Since out_channels = 1, the bias should have one element.
conv.bias.data = torch.tensor([0.0])  # You can adjust this value as needed.

# Apply the convolutional layer
output_tensor = conv(input_tensor)

# Print the output tensor
print("Output Tensor:")
print(output_tensor)

