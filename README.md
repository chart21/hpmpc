# HPMPC: High-Performance Implementation of Secure Multiparty Computation (MPC) Protocols


HPMPC implements multiple MPC protocols and provides a high-level C++ interface to define functions and use cases.
Out of the box, the framework supports computation in the boolean and arithmetic domain, mixed circuits, and fixed point arithmetic. 
Neural network models can be imported from PyTorch as part of [PIGEON (Private Inference of Neural Networks)](#pigeon-private-inference-of-neural-networks).

## Documentation

More extensive documentation can be found [here](https://c.harth-kitzerow.com/mkdocs-hpmpc/).



## Getting Started

TLDR instructions can be found [here](#tldr).

You can use the provided Dockerfile or set up the project manually. The only dependency is OpenSSL. Neural networks and other functions with matrix operations also require the Eigen library. 
```bash
#Install Dependencies:
sudo apt install libssl-dev libeigen3-dev
```

### Local Setting
You can run the following commands to compile and execute a program with an MPC protocol locally.
```bash
# Compile executables for protocol Trio (5) for all parties and unit tests for basic primitives (function 54)
make -j PARTY=all FUNCTION_IDENTIFIER=54 PROTOCOL=5 
# Run the MPC protocol locally
scripts/run.sh -p all -n 3 # Run three parties locally
```

### Distributed Setting
After setting up the framework on each node of a distributed setup, you can run the following commands to run the MPC protocol on a distributed setup. Replace ``<party_id>`` with e.g. 0 to compile an executable for party 0. 
```bash
make -j PARTY=<party_id>
# Run the MPC protocol on a distributed setup. For 2PC and 3PC protocols, the -c  or -d flags are not required.
scripts/run.sh -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> 
```

### GPU Acceleration

GPU acceleration for matrix multiplication and convolutions requires an NVIDIA GPU, the NVCC compiler, and a copy of the [CUTLASS](https://github.com/NVIDIA/cutlass) library.  To obtain the GPU architecture (sm_xx), refer to [this oerview](https://developer.nvidia.com/cuda-gpus).
#### Set up GPU Support
```bash
# Dependencies for GPU acceleration
git clone https://github.com/NVIDIA/cutlass.git

# Compile standalone executable for GPU acceleration
cd core/cuda
# Replace with your GPU architecture, nvcc path, and CUTLASS path:
make -j arch=sm_89 CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/home/user/cutlass 
cd ../..
```
#### Compile Executables with GPU support

```bash
# Compile executables for protocol Quad (12) for all parties and unit tests for matrix multiplication (function 54) with GPU acceleration (USE_CUDA_GEMM=2)
make -j PARTY=all FUNCTION_IDENTIFIER=57 PROTOCOL=12 USE_CUDA_GEMM=2 
```

### SplitRoles

SplitRoles compiles multiple executables per player to perform load balancing. Running a protocol with SplitRoles can be done by running the following commands. More information on Split-Roles can be found in the section [Scaling MPC to Billions of Gates per Second](#scaling-mpc-to-billions-of-gates-per-second).
#### Compile and Run executables with Split-Roles
```bash
make -j PARTY=<party_id> SPLITROLES=1 # Compile multiple executables for a 3PC protocol with Split-Roles
scripts/run.sh -s 1 -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> 
``` 



#### Compile and Run executables with Multi-GPU Support
SplitRoles supports multi-GPU setups. To run a protocol with multiple GPUs, you can run the following commands.
```bash 
make -j USE_CUDA_GEMM=2 # USE_CUDA_GEMM 1/2/4 work as well
scripts/run.sh -p <party_id> -s 1 -g 6 # Utilize 6 GPUs for the computation
```

## Project Structure
The framework uses a modular architecture with the following components.

| Software Component | Description |
| --- | --- |
| Core | Implements communication between parties, cryptographic primitives, and techniques for hardware acceleration. Uses Bitslicing, Vectorization, GPU acceleration, and hardware instruction for cryptographic primitives to accelerate local computation required by the MPC protocols. |
| Protocols | Implements MPC protocols and protocol-specific primitives. Each protocol utilizes high-level operations provided by `Core` for commonly used operations such as sampling shared random numbers or exchanging messages. |
| Datatypes | Implements different datatypes that serve as a high-level interface to compute on MPC shares generically with overloaded operators. |
| Programs | Implements high-level functions, routines, and use cases using the custom `datatypes`. Implements several MPC-generic functions such as matrix multiplication and comparisons. |
| NN | Implements a templated neural network inference engine that performs the forward pass of a CNN by relying on high-level MPC-generic functions provided by `Programs`. Models and datasets can be exported from PyTorch. |

### Extending the Framework
- New functions can be added to `programs/` by using the operations supported by `Datatypes`.
- New MPC protocols can be added to `protocols/` by using the networking and cryptographic utilities provided by `Core`.
- New Neural Network Model architectures can be added to `nn/PIGEON/architectures/` by using our PyTorch-like interface to define model architectures.
- Model parameters and datasets can be exported from PyTorch using `nn/Pygeon/`.


## Scaling MPC to Billions of Gates per Second

The framework offers multiple tweaks to accelerate MPC computation. The following are the most important settings that can be adjusted in by setting the respective flags when compiling with `make` or by permanently changing the entries in `config.h`. 
| Configuration Type | Options | Description |
| --- | --- | --- |
| Concurrency | `DATTYPE`, `PROCESS_NUM` | `DATTYPE` defines the register length to vectorize all integers and boolean variables to fully utilize the register. `PROCESS_NUM` sets the number of processes to use for parallel computation. 
| Hardware Acceleration | `RANDOM_ALGORITHM`, `USE_SSL_AES`, `ARM`, `USE_CUDA_GEMM` | Different approaches for efficiently implementing cryptographic primitives on various hardware architectures. Matrix operations can be accelerated using CUDA. |
| Tweaks | `SEND_BUFFER`, `RECV_BUFFER`, `VERIFY_BUFFER` | Setting buffer sizes for communication and sha hashing to verify messages can accelerate workloads. The default settings should provide a good starting point for most settings. |
| Preprocessing | `PRE` | Some protocols support a preprocessing phase that can be enabled to accelerate the online phase. |
| SplitRoles | `SPLITROLES` | By using the SPLITROLES flag when compiling, the framework compiles n! executables for a n-PC protocol where each executable has a different player assignment. This allows load balance the communication and computation between the nodes. `SPLITROLES=1` compiles all(6) executables for a 3PC protocol, `SPLITROLES=2` compiles all (24) executables for a 3PC protocol in a setting with four nodes, and `SPLITROLES=3` compiles all (24) executables for a 4PC protocol. |

For nodes equipped with a 32-core AVX-512 CPU, and a CUDA-enabled GPU, the following example may compile an optimized executable in a distributed setup. Note that this example inherently vectorizes the computation `PROCESS_NUM x DATTYPE/BITLENGTH x SPLITROLES_FACTOR` times. 
```bash
make -j PARTY=<party_id> FUNCTION_IDENTIFIER=<function_id> PROTOCOL=12 DATTYPE=512 PROCESS_NUM=32 RANDOM_ALGORITHM=2 USE_SSL_AES=0 ARM=0 USE_CUDA_GEMM=2 SEND_BUFFER=10000 RECV_BUFFER=10000 VERIFY_BUFFER=1 PRE=1 SPLITROLES=3
```

## Protocols

Out of the box, the framework supports multiple MPC protocols. For some protocols, only basic primitives such as secret sharing, addition, and multiplication are currently implemented. Other protocols support additional primitives to fully support mixed circuits and fixed point arithmetic. 
A protocol can be selected with the `PROTOCOL` flag when compiling.

| Protocol | Adversary Model | Preprocessing | Supported Primitives |
| --- | --- | --- | --- |
| `1` Sharemind (3PC) | Semi-Honest | ✘ | Basic
| `2` Replicated (3PC) | Semi-Honest |✘  | Basic
| `3` ASTRA (3PC) | Semi-Honest | ✔ | Basic
| `4` ABY2 Dummy (2PC) | Semi-Honest | ✔ | Basic
| `5` Trio (3PC) | Semi-Honest | ✔ | All
| `6` Trusted Third Party (3PC) | Semi-Honest | ✘ | All
| `7` Trusted Third Party (4PC) | Semi-Honest | ✘ | All
| `8` Tetrad (4PC) | Malicious | ✔ | Basic
| `9` Fantastic Four (4PC) | Malicious | ✘ | Basic
| `10` Quad (4PC) | Malicious | ✘ | All
| `11` Quad: Het (4PC) | Malicious | ✔ | All
| `12` Quad (4PC) | Malicious | ✔ | All

Trio, ASTRA, Quad, ABY2, and Tetrad support a Preprocessing phase.
The preprocessing phase can be enabled in `config.h` or by setting `PRE=1` when compiling.
Setting `PRE=0` interleaves the preprocessing and online phase.
New protocols can be added to `protocols/`and adding a protocol ID to `protocols/Protocols.h`. 

## Functions

Out of the box, the framework provides multiple high-level functions that operate on Additive and Boolean shares. 
`programs/functions/` contains unit tests and benchmarks for these functions. 
An overview of which id corresponds to which function can be found in `protocol_executer.hpp`. 
In the following, we provide a brief overview of the functions that are currently implemented.
| Category | Functions |
| --- | --- |
| Basic Primitives | Secret Sharing, Reconstruction, Addition, Multiplication, Division, etc. |
| Fixed Point Arithmetic | Fixed Point Addition, Multiplication, Truncation, Division, etc. |
| Matrix Operations | Matrix Multiplication, Dot Product, etc. |
| Multi-input Operations | Multi-input Multiplication, Multi-input Scalar Products, etc. |
| Comparisons | EQZ, LTZ, MAX, MIN, Argmax, etc. |
| Use Cases (Benchmarking) | Set Intersection, Auction, AES, Logistic Regression, etc. |
| Neural Networks | Forward Pass of CNN/ResNet, ReLU, Softmax, Pooling, Batchnorm, etc. |

To implement a custom programs, these functions can be used as building blocks. 
`programs/tutorials/` contains tutorials on how to use different functions. 
New functions can be added by first implementing the function in `programs/functions/` and then adding a `FUNCTION_IDENTIFIER` to `protocol_executer.hpp`. The tutorial `programs/tutorials/YourFirstProgram.hpp` should get you started after following the other tutorials.


## The Vectorized Programming Model
Scaling MPC requires a high degree of parallelism to overcome network latency bottlenecks.
HPMPC's architecture is designed to utilize hardware resources proportionally to the degree of parallelism required by the MPC workload. 
By increasing load balancing, register sizes, or number of processes, the framework executes multiple instances of the same function in parallel.
For instance, by setting `DATTYPE=512` and `PROCESS_NUM=32`, each arithmetic operation on 32-bit integers is executed 512 times in parallel by using 32 processes on 16 packed integers per register.
Similarly, a boolean operation is executed 512x32=16384 times in parallel with 32 processes and 512-bit registers due to Bitslicing.
For mixed circuits, HPMPC automatically groups blocks of arithmetic shares before share conversion to handle these different degrees of parallelism.
The degree of parallelism for operations can be calculated as follows (Boolean operations have a BITLENGTH of 1):

> `PROCESS_NUM x DATTYPE/BITLENGTH x SPLITROLES_Factor` 

The following examples illustrate the concept of parallelism in HPMPC.

- Setting `SPLITROLES=1`, `PROCESS_NUM=4`, and `DATTYPE=256` to compile a program computing 10 AES blocks (boolean circuit) will actually compute `6x4x256x10=61440` AES blocks in parallel by fully utilizing the available hardware resources, 
- Setting `DATTYPE=1`, `SplitRoles=0`, and `PROCESS_NUM=1` will compute 10 AES blocks on a single core without vectorization. 
- Setting `SPLITROLES=1`, `PROCESS_NUM=4`, and `DATTYPE=256`, `NUM_INPUTS=1` to compile a program computing a single neural network inference (mixed circuit) will evaluate `6x4x256/32=192` samples in parallel, thus effectively using a batch size of 192.

## Executing MP-SPDZ Bytecode (Experimental)

HPMPC can execute bytecode generated by the MP-SPDZ compiler.


## PIGEON: Private Inference of Neural Networks

PIGEON adds support for private inference of neural networks. PIGEON adds the following submodules to the framework.

* [FlexNN](https://github.com/chart21/flexNN/tree/hpmpc): A templated neural network inference engine to perform the forward pass of a CNN.
* [Pygeon](https://github.com/chart21/Pygeon): Python scripts for exporting models and datsets from PyTorch to the inference engine. 

All protocols that are fully supported by HPMPC can be used with PIGEON. To get started with PIGEON, initialize the submodules to set up FlexNN and Pygeon. 
```bash
git submodule update --init --recursive
```

### End-to-End Training and Inference Pipeline

A full end-to-end example can be executed as follows. To only benchmark the inference without real data, set `MODELOWNER` and `DATAOWNER` to `-1` and skip steps 1 and 5.

1. Use Pygeon to train a model in PyTorch and export its test labels, test images, and model parameters to `.bin` files using the provided scripts. Alternatively, download the provided pre-trained models.
   
    ```bash
    cd nn/Pygeon
    # Option 1: Train a model and export it to PyGEON 
    python main.py --action train --export_model --export_dataset --transform standard --model VGG16 --num_classes 10 --dataset_name CIFAR-10 --modelpath ./models/alexnet_cifar --num_epochs 30 --lr 0.01 --criterion CrossEntropyLoss --optimizer Adam
    # Option 2: Download a pretrained VGG16 model and CIFAR10 dataset
    python download_pretrained.py single_model datasets
    cd ../..
    ```

2. If it does not exist yet, add your model architecture to `nn/PIGEON/architectures/`.

3. If it does not exist yet, add a `FUNCTION_IDENTIFIER` for your model architecture and dataset dimensions in `Programs/functions/NN.hpp`.

4. Specify the `MODELOWNER` and `DATAOWNER` config options when compiling.

    ```bash
    # Example for MODELOWNER=P_0 and DATAOWNER=P_1
    make -j PARTY=<party_id> FUNCTION_IDENTIFIER=<function_id> DATAOWNER=P_0 MODELOWNER=P_1 
    ```

5. Specify the path of your model, images, and labels by exporting the environment variables `MODEL_DIR`, `DATA_DIR`, `MODEL_FILE`, `SAMPLES_FILE`, and `LABELS_FILE`.

    ```bash
    # Set environment variables for the party holding the model parameters (adjust paths if needed)
    export MODEL_DIR=nn/Pygeon/models/pretrained
    export MODEL_FILE=vgg16_cifar_standard.bin

    # Set environment variables for the party holding the dataset (adjust paths if needed)
    export DATA_DIR=nn/Pygeon/data/datasets 
    export SAMPLES_FILE=CIFAR-10_standard_test_images.bin 
    export LABELS_FILE=CIFAR-10_standard_test_labels.bin 
    ```

6. Run the program

    ```bash
    scripts/run.sh -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> 
    ```

### Inference Configurations


PIGEON provides several options to modify the inference. The following are the most important settings that can be adjusted by setting the respective flags when compiling.
| Configuration Type | Options | Description |
| --- | --- | --- |
| Bits | `BITLENGTH`, `FRACTIONAL` | The number of bits used for the total bitlength and the fractional part respectively. |
| Truncation | `TRUNC_APPROACH`, `TRUNC_THEN_MULT`, `TRUNC_DELAYED` | There are multiple approaches to truncation. The default approach is to truncate probabilistically after each multiplication. The different approaches allow switching between several truncation strategies. |
| ReLU | `REDUCED_BITLENGTH_m`, `REDUCED_BITLENGTH_k` | ReLU can be evaluated probabilistically by reducing its bitwidth to save communication and computation. The default setting is to evaluate ReLU with the same bitwidth as the rest of the computation. |
| Secrecy | `PUBLIC_WEIGHTS`, `COMPUTE_ARGMAX` | The weights can be public or private. The final argmax computation may not be required if parties should learn the probabilities of each class. |
| Optimizations | `ONLINE_OPTIMIZED`, `BANDWIDTH_OPTIMIZED` | All layers requiring sign bit extraction such as ReLU, Maxpooling, and Argmax can be evaluated with different types of adders. These have different trade-offs in terms of online/preprocessing communication as well as total round complexity and communication complexity. |
| Other Optimizations | `SPLITROLES`, `BUFFER_SIZE`, `VECTORIZE` | All default optimizations of HPMPC such as `SPLITROLES`, different buffers, and vectorization can be used with PIGEON. The parties automatically utilize the concurrency to perform inference on multiple independent samples from the dataset in parallel. To benchmark the inference without real data, `MODELOWNER` and `DATAOWNER` can be set to `-1`. |




## Measurements

To automate benchmarks and tests of various functions and protocols, users can define `.conf` files in the `measurements/configs` directory. The following is an example of a configuration file that runs a function with different number of inputs and different protocols.

```
PROTOCOL=8,9,12
NUM_INPUTS=10000,100000,1000000
FUNCTION_IDENTIFIER=1
DATTYPE=32
BITLENGTH=32
```

### Running Measurements
The `run_config.py` script runs compiles and executes all combinations in `.conf`. Outputs are stored as `.log` files in the `measurements/logs/` directory. 
```bash
python3 measurements/run_config.py -p <party_id> measurements/configs/<config_file>.conf
```



### Parsing Measurement Results
Results in `.log` files can be parsed with the `measurements/parse_logs.py` script. The parsed result contains information such as communication, runtime, throughput, and if applicable the number of unit tests passed or accuracy achieved.
```bash
python3 measurements/parse_logs.py measurements/logs/<log_file>.log
```

## Troubleshooting
The framework utilizes different hardware acceleration techniques for a range of hardware architectures. 
In case of timeouts, change the `BASE_PORT` or make sure that all previous executions have been terminated by executing `pkill -9 -f run-P` on all nodes.
In case of compile errors, please note the following requirements and supported bitlengths for different `DATTYPE` values.

### Register Size and Hardware Requirements

| Register Size | Requirements            | Supported BITLENGTH             | Config Option                  |
|------|-------------------------|---------------------------------|--------------------------------------|
| 512  | AVX512         | 16, 32, 64                      | `DATTYPE=512`                |
| 256  | AVX2           | 16, 32, (64 with AVX512)         | `DATTYPE=256`                |
| 128  | SSE                     | 16, 32, (64 with AVX512)         | `DATTYPE=128`                |
| 64   | None                    | 64                              | `DATTYPE=64`                 |
| 32   | None                    | 32                              | `DATTYPE=32`                 |
| 16   | None                    | 16                              | `DATTYPE=16`                 |
| 8    | None                    | 8 (Does not support all arithmetic instructions) | `DATTYPE=8`                  |
| 1    | None                    | 16,32,64 (Use only for boolean circuits)                               | `DATTYPE=1`                  |

### Hardware Acceleration Requirements

To benefit from Hardware Acceleration, the following config options are important.

| Config Option | Requirements            | Description |
|------|-------------------------|---------------------------------|
| `RANDOM_ALGORITHM=2` | AES-NI or VAES | Use the AES-NI or VAES instruction set for AES. If not available, set `USE_SSL_AES=1` or `RANDOM_ALGORITHM=1` |
| `USE_CUDA_GEMM>0` | CUDA, CUTLASS | Use CUDA for matrix multiplications and convolution. In case your CUDA-enabled GPU does not support datatypes such as UINT8, you can comment out the respective forward declaration in `core/cuda/conv_cutlass_int.cu` and `core/cuda/gemm_cutlass_int.cu`.|
| `ARM=1` | ARM CPU | For ARM CPUs, setting `ARM=1` may improve performance of SHA hashing. |

### Increase Accuracy of Neural Network Inference

If you encounter issues regarding the accuracy of neural network inference, the following options may increase accuracy.
- Increase the `BITLENGTH`.
- Increase or reduce the number of `FRACTIONAL` bits.
- Adjust the truncation strategy to `TRUNC_APPROACH=1` (REDUCED Slack) or `TRUNC_APPROACH=2` (Exact Truncation), along with `TRUNC_THEN_MULT=1` and `TRUNC_DELAYED=1`. Note that truncation approaches 1 and 2 require setting `TRUNC_DELAYED=1`.
- Inspect the terminal output for any errors regarding reading the model or dataset. PIGEON uses dummy data or model parameters if the files are not found. Make sure that `MODELOWNER` and `DATAOWNER` are set during compilation and that the respective environment variables point to existing files.


## TLDR

### Setup (CPU only)
```bash
sudo apt install libssl-dev libeigen3-dev
git submodule update --init --recursive
pip install torch torchvision gdown # if not already installed
```

### Unit Tests

#### Run all unit tests locally
```bash
python3 measurements/run_config.py measurements/configs/unit_tests/  
```
#### Run all unit tests on a distributed setup
```bash
python3 measurements/run_config.py measurements/configs/unit_tests/ -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3>
```
#### Parse the results
```bash
python3 measurements/parse_logs.py measurements/logs/ # results are stored as `.csv` in measurements/logs/
```

### End to End neural network training and secure inference

#### Prepare neural network inference with a pre-trained model
```bash
cd nn/Pygeon
python download_pretrained.py single_model datasets
export MODEL_DIR=nn/Pygeon/models/pretrained
export MODEL_FILE=vgg16_cifar_standard.bin
export DATA_DIR=nn/Pygeon/data/datasets
export SAMPLES_FILE=CIFAR-10_standard_test_images.bin
export LABELS_FILE=CIFAR-10_standard_test_labels.bin
cd ../..
```

#### Compile and run the neural network inference locally

```bash
make -j PARTY=all FUNCTION_IDENTIFIER=74 PROTOCOL=5 MODELOWNER=P_0 DATAOWNER=P_1 NUM_INPUTS=40 BITLENGTH=32 DATTYPE=32
scripts/run.sh -p all -n 3
```

#### Compile and run the neural network inference on a distributed setup

```bash
make -j PARTY=<party_id> FUNCTION_IDENTIFIER=74 PROTOCOL=5 MODELOWNER=P_0 DATAOWNER=P_1
scripts/run.sh -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3>
```

### Benchmarks

#### Run AND gate benchmark with different protocols and number of processes on a local/distributed setup
```bash
# use DATTYPE=256 or DATTYPE=128 or DATTYPE=64 for CPUs without AVX/SSE support.

#Local Setup
python3 measurements/run_config.py -p all measurements/configs/benchmarks/AND.conf

#Distributed Setup
python3 measurements/run_config.py -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> measurements/configs/benchmarks/Multiprocesssing.conf --override NUM_INPUTS=1000000 DATTYPE=512
```

#### Run LeNet5 on MNIST locally with batch size 24 using SPLITROLES
```bash
# use DATTYPE=256 or DATTYPE=128 or DATTYPE=64 for CPUs without AVX/SSE support.

# 3PC
python3 measurements/run_config.py -s 1 -p all measurements/configs/benchmarks/lenet5.conf --override PROTOCOL=5 PROCESS_NUM=4

# 4PC
python3 measurements/run_config.py -s 3 -p all measurements/configs/benchmarks/lenet5.conf --override PROTOCOL=12 PROCESS_NUM=1
```

#### Run various neural network models in a distributed stting on ImageNet with 3 iterations per run and SPLITROLES (Requires server-grade hardware)
```bash
# use DATTYPE=256 or DATTYPE=128 or DATTYPE=64 for CPUs without AVX/SSE support.

# 3PC
python3 measurements/run_config.py -s 1 -i 3 -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> measurements/configs/benchmarks/imagenetmodels.conf --override PROTOCOL=5 PROCESS_NUM=4 # 4PC

# 4PC
python3 measurements/run_config.py -s 3 -i 3 -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> measurements/configs/benchmarks/imagenetmodels.conf --override PROTOCOL=12 PROCESS_NUM=12 
```

#### Parse the results
```bash
python3 measurements/parse_logs.py measurements/logs/ # results are stored as `.csv` in measurements/logs/
```


## References

Our framework utilizes the following third-party implementations.
- Architecture-specific headers for vectorization and Bitslicing adapted from [USUBA](https://github.com/usubalang/usuba), [MIT LICENSE](https://raw.githubusercontent.com/usubalang/usuba/main/LICENSE).
- AES-NI implementation adapted from [AES-Brute-Force](https://github.com/sebastien-riou/aes-brute-force), [Apache 2.0 LICENSE](https://raw.githubusercontent.com/sebastien-riou/aes-brute-force/master/LICENSE)
- SHA-256 implementation adapted from [SHA-Intrinsics](https://github.com/noloader/SHA-Intrinsics/tree/master), No License.
- CUDA GEMM and Convolution implementation adapted from [Cutlass](https://github.com/NVIDIA/cutlass), [LICENSE](https://raw.githubusercontent.com/NVIDIA/cutlass/main/LICENSE.txt) and [Piranha](https://github.com/ucbrise/piranha/tree/main), [MIT LICENSE](https://raw.githubusercontent.com/ucbrise/piranha/main/LICENSE).
- Neural Network Inference engine adapted from [SimpleNN](https://github.com/stnamjef/SimpleNN), [MIT LICENSE](https://raw.githubusercontent.com/stnamjef/SimpleNN/master/LICENSE).

