import torch
import torch.nn as nn
import numpy as np


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
    

def print_layers_and_params(model):
    for name, module in model.named_modules():
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if total_params > 0:
            # if isinstance(module, nn.Conv2d or nn.Linear or nn.BatchNorm2d or nn.BatchNorm1d):
            print(f"{name}: {total_params} parameters")
        else:
            print(f"{name}: No parameters")

