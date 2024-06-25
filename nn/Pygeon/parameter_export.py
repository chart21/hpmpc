import torch
import numpy as np


def save_quantized_weights_compatible_with_cpp(model, filepath):
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
        all_params_flat = np.concatenate(all_params).astype(np.int32)

        # Write total number of parameters and then write the parameters
        f.write(np.array([all_params_flat.size], dtype=np.int32).tobytes())
        f.write(all_params_flat.tobytes())


def save_quantization_params(model, filepath):
    with open(filepath, 'wb') as f:
        all_params = []

        # Process each module according to its type
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                if module.scale is not None:
                    all_params.append(module.scale.data.cpu().numpy().ravel())
                if module.zero_point is not None:
                    all_params.append(module.zero_point.data.cpu().numpy().ravel())
            elif isinstance(module, torch.nn.Conv2d):
                if model.scale is not None:
                    all_params.append(module.scale.data.cpu().numpy().ravel())
                if model.zero_point is not None:
                    all_params.append(module.zero_point.data.cpu().numpy().ravel())

        # Flatten all parameters and convert to numpy array
        all_params_flat = np.concatenate(all_params).astype(np.float32)

        # Write total number of parameters and then write the parameters
        f.write(np.array([all_params_flat.size], dtype=np.int32).tobytes())
        f.write(all_params_flat.tobytes())

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


#multiline comment
"""
{
    for (Layer<T>* l : net) {
        vector<float> tempMatrix1, tempMatrix2, tempMatrix3, tempMatrix4; // Temporary vectors for parameter storage

        if (l->type == LayerType::LINEAR) {
            Linear<T>* lc = dynamic_cast<Linear<T>*>(l);
            int s1 = lc->W.rows() * lc->W.cols();
            int s2 = lc->b.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);

            if (mode == "write") {
                for (int i = 0; i < s1; i++) 
                {
                    tempMatrix1[i] = lc->W(i / lc->W.cols(), i % lc->W.cols()).reveal();
                }
                for (int i = 0; i < s2; i++)
                {
                    tempMatrix2[i] = lc->b[i].reveal();
                }
                    fs.write((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.write((char*)tempMatrix2.data(), sizeof(float) * s2);
            }
            else {
                fs.read((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(float) * s2);
                for (int i = 0; i < s1; i++) 
                {
                    lc->W(i / lc->W.cols(), i % lc->W.cols()) = T(tempMatrix1[i]);
                }
                for (int i = 0; i < s2; i++)
                {
                    lc->b[i] = T(tempMatrix2[i]);
                }
            }
        }
        else if (l->type == LayerType::CONV2D) {
            Conv2d<T>* lc = dynamic_cast<Conv2d<T>*>(l);
            int s1 = lc->kernel.rows() * lc->kernel.cols();
            int s2 = lc->bias.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);

            if (mode == "write") {
                for (int i = 0; i < s1; i++) 
                {
                    tempMatrix1[i] = lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()).reveal();
                }
                for (int i = 0; i < s2; i++) 
                {
                    tempMatrix2[i] = lc->bias[i].reveal();
                }
                fs.write((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.write((char*)tempMatrix2.data(), sizeof(float) * s2);
            }
            else {
                fs.read((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(float) * s2);
                for (int i = 0; i < s1; i++)
                {
                    lc->kernel(i / lc->kernel.cols(), i % lc->kernel.cols()) = T(tempMatrix1[i]);
                } 
                for (int i = 0; i < s2; i++)
                {
                    lc->bias[i] = T(tempMatrix2[i]);
                }
            }
        }
        else if (l->type == LayerType::BATCHNORM1D) {
            BatchNorm1d<T>* lc = dynamic_cast<BatchNorm1d<T>*>(l);
            int s1 = (int)lc->move_mu.size();
            int s2 = (int)lc->move_var.size();
            int s3 = (int)lc->gamma.size();
            int s4 = (int)lc->beta.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);
            tempMatrix3.resize(s3);
            tempMatrix4.resize(s4);
            if (mode == "write") {
                for (int i = 0; i < s1; i++) 
                {
                    tempMatrix1[i] = lc->move_mu[i].reveal();
                }
                for (int i = 0; i < s2; i++) 
                {
                    tempMatrix2[i] = lc->move_var[i].reveal();
                }
                for (int i = 0; i < s3; i++) 
                {
                    tempMatrix3[i] = lc->gamma[i].reveal();
                }
                for (int i = 0; i < s4; i++) 
                {
                    tempMatrix4[i] = lc->beta[i].reveal();
                }
                fs.write((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.write((char*)tempMatrix2.data(), sizeof(float) * s2);
                fs.write((char*)tempMatrix3.data(), sizeof(float) * s3);
                fs.write((char*)tempMatrix4.data(), sizeof(float) * s4);
            }
            else {
                fs.read((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(float) * s2);
                fs.read((char*)tempMatrix3.data(), sizeof(float) * s3);
                fs.read((char*)tempMatrix4.data(), sizeof(float) * s4);
                for (int i = 0; i < s1; i++)
                {
                    lc->move_mu[i] = T(tempMatrix1[i]);
                } 
                for (int i = 0; i < s2; i++)
                {
                    float var = 1 / std::sqrt(tempMatrix2[i] + 0.00001f);
                    lc->move_var[i] = T(var);
                }
                for (int i = 0; i < s3; i++)
                {
                    lc->gamma[i] = T(tempMatrix3[i]);
                }
                for (int i = 0; i < s4; i++)
                {
                    lc->beta[i] = T(tempMatrix4[i]);
                }
            }
        }
        else if (l->type == LayerType::BATCHNORM2D) {
            BatchNorm2d<T>* lc = dynamic_cast<BatchNorm2d<T>*>(l);
            int s1 = (int)lc->move_mu.size();
            int s2 = (int)lc->move_var.size();
            int s3 = (int)lc->gamma.size();
            int s4 = (int)lc->beta.size();
            tempMatrix1.resize(s1);
            tempMatrix2.resize(s2);
            tempMatrix3.resize(s3);
            tempMatrix4.resize(s4);
            if (mode == "write") {
                for (int i = 0; i < s1; i++) 
                {
                    tempMatrix1[i] = lc->move_mu[i].reveal();
                }
                for (int i = 0; i < s2; i++) 
                {
                    tempMatrix2[i] = lc->move_var[i].reveal();
                }
                for (int i = 0; i < s3; i++) 
                {
                    tempMatrix3[i] = lc->gamma[i].reveal();
                }
                for (int i = 0; i < s4; i++) 
                {
                    tempMatrix4[i] = lc->beta[i].reveal();
                }
                fs.write((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.write((char*)tempMatrix2.data(), sizeof(float) * s2);
                fs.write((char*)tempMatrix3.data(), sizeof(float) * s3);
                fs.write((char*)tempMatrix4.data(), sizeof(float) * s4);
            }
            else {
                fs.read((char*)tempMatrix1.data(), sizeof(float) * s1);
                fs.read((char*)tempMatrix2.data(), sizeof(float) * s2);
                fs.read((char*)tempMatrix3.data(), sizeof(float) * s3);
                fs.read((char*)tempMatrix4.data(), sizeof(float) * s4);
                for (int i = 0; i < s1; i++)
                {
                    lc->move_mu[i] = T(tempMatrix1[i]);
                } 
                for (int i = 0; i < s2; i++)
                {
                    float var = 1 / std::sqrt(tempMatrix2[i] + 0.00001f);
                    lc->move_var[i] = T(var);
                }
                for (int i = 0; i < s3; i++)
                {
                    lc->gamma[i] = T(tempMatrix3[i]);
                }
                for (int i = 0; i < s4; i++)
                {
                    lc->beta[i] = T(tempMatrix4[i]);
                }
            }
        }

    }
}
"""


import torch
import numpy as np

# def write_params(model, filename):
#     with open(filename, 'wb') as f:
#         for layer in model.children():
#             if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
#                 # Flatten and save weights
#                 np_weights = layer.weight.data.numpy().flatten()
#                 np_weights.tofile(f)

#                 # Save biases
#                 if layer.bias is not None:
#                     np_bias = layer.bias.data.numpy()
#                     np_bias.tofile(f)

#             elif isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
#                 # Save BatchNorm parameters
#                 np_mu = layer.running_mean.numpy()
#                 np_var = layer.running_var.numpy()
#                 np_gamma = layer.weight.data.numpy()
#                 np_beta = layer.bias.data.numpy()

#                 np_mu.tofile(f)
#                 np_var.tofile(f)
#                 np_gamma.tofile(f)
#                 np_beta.tofile(f)

# def read_params(model, filename):
#     with open(filename, 'rb') as f:
#         for layer in model.children():
#             if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
#                 # Load weights
#                 num_elements = np.prod(np.array(layer.weight.size()))
#                 np_weights = np.fromfile(f, dtype=np.float32, count=num_elements)
#                 layer.weight.data = torch.from_numpy(np_weights.reshape(layer.weight.size()))

#                 # Load biases
#                 if layer.bias is not None:
#                     np_bias = np.fromfile(f, dtype=np.float32, count=layer.bias.numel())
#                     layer.bias.data = torch.from_numpy(np_bias)

#             elif isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
#                 # Load BatchNorm parameters
#                 np_mu = np.fromfile(f, dtype=np.float32, count=layer.running_mean.numel())
#                 layer.running_mean = torch.from_numpy(np_mu)

#                 np_var = np.fromfile(f, dtype=np.float32, count=layer.running_var.numel())
#                 layer.running_var = torch.from_numpy(np_var)

#                 np_gamma = np.fromfile(f, dtype=np.float32, count=layer.weight.numel())
#                 layer.weight.data = torch.from_numpy(np_gamma)

#                 np_beta = np.fromfile(f, dtype=np.float32, count=layer.bias.numel())
#                 layer.bias.data = torch.from_numpy(np_beta)


def count_exported_params(model):
    total_params = 0
    for module in model.modules():  # `modules()` will iterate over all modules in the network
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Count weights and biases
            total_params += module.weight.data.nelement()
            if module.bias is not None:
                total_params += module.bias.data.nelement()

        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # Count all BatchNorm parameters
            total_params += module.running_mean.nelement()
            total_params += module.running_var.nelement()
            if module.weight is not None:
                total_params += module.weight.data.nelement()
            if module.bias is not None:
                total_params += module.bias.data.nelement()

    print("Total number of parameters: {}".format(total_params))
    return total_params


def write_params(model, filename):
    total_params = count_exported_params(model)
    with open(filename, 'wb') as f:
        # Write the total number of parameters first
        np.array([total_params], dtype=np.int32).tofile(f)

        # Then write the parameters for each module
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # Flatten and save weights
                np_weights = module.weight.data.numpy().flatten()
                np_weights.tofile(f)

                # Save biases
                if module.bias is not None:
                    np_bias = module.bias.data.numpy()
                    np_bias.tofile(f)

            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Save BatchNorm parameters
                np_mu = module.running_mean.numpy()
                np_var = module.running_var.numpy()
                np_gamma = module.weight.data.numpy()
                np_beta = module.bias.data.numpy()

                np_mu.tofile(f)
                np_var.tofile(f)
                np_gamma.tofile(f)
                np_beta.tofile(f)
def write_params(model, filename):
    total_params = count_exported_params(model)
    with open(filename, 'wb') as f:
        # Write the total number of parameters first
        np.array([total_params], dtype=np.int32).tofile(f)
        print(f"Total parameters: {total_params}")

        for module in model.modules():
            print(module)
            if isinstance(module, torch.nn.Linear):
                # Print and write weights, then biases for Linear layers
                print_layer_info("Linear", module.weight.data.nelement(), module.bias.data.nelement() if module.bias is not None else 0)
                module.weight.data.numpy().flatten().tofile(f)
                if module.bias is not None:
                    module.bias.data.numpy().tofile(f)

            elif isinstance(module, torch.nn.Conv2d):
                # Print and write kernel weights, then biases for Conv2d layers
                print_layer_info("Conv2d", module.weight.data.nelement(), module.bias.data.nelement() if module.bias is not None else 0)
                module.weight.data.numpy().flatten().tofile(f)
                if module.bias is not None:
                    module.bias.data.numpy().tofile(f)

            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Print and write BatchNorm parameters
                gamma_beta_count = module.weight.data.nelement() + module.bias.data.nelement() if module.weight is not None and module.bias is not None else 0
                print_layer_info("BatchNorm", module.running_mean.nelement(), module.running_var.nelement(), gamma_beta_count)
                module.running_mean.numpy().tofile(f)
                module.running_var.numpy().tofile(f)
                if module.weight is not None:
                    module.weight.data.numpy().tofile(f)
                if module.bias is not None:
                    module.bias.data.numpy().tofile(f)

def print_layer_info(layer_type, weight_count, bias_count, additional_count=0):
    total = weight_count + bias_count + additional_count
    print(f"{layer_type} Layer - Weights: {weight_count}, Biases: {bias_count}, Additional: {additional_count}, Total: {total}")

def read_params(model, filename):
    total_params = count_exported_params(model)
    with open(filename, 'rb') as f:
        # Read the total number of parameters first
        num_params = np.fromfile(f, dtype=np.int32, count=1)[0]
        if num_params != total_params:
            raise ValueError("Expected {} parameters, but found {}.".format(total_params, num_params))
        for module in model.modules():
            
            if isinstance(module, torch.nn.Linear):
                # Read and reshape weights, then biases
                read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)

            elif isinstance(module, torch.nn.Conv2d):
                # Read and reshape kernel weights, then biases
                read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)

            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Read BatchNorm parameters in order: moving mean, variance, gamma, beta
                read_to_tensor(f, module.running_mean)
                read_to_tensor(f, module.running_var)
                if module.weight is not None:
                    read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)

def read_to_tensor(file, tensor):
    num_elements = tensor.numel()
    tensor_data = np.fromfile(file, dtype=np.float32, count=num_elements)
    tensor.data.copy_(torch.from_numpy(tensor_data).view_as(tensor))

