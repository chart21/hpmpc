# HPMPC: High-Performance Implementation of Secure Multiparty Computation (MPC) Protocols


HPMPC implements multiple MPC protocols and provides a high-level C++ interface to define functions and use cases.
Out of the box, the framework supports computation in the boolean and arithmetic domain, mixed circuits, and fixed point arithmetic.
Neural network models can be imported from PyTorch as part of [PIGEON (Private Inference of Neural Networks)](#pigeon-private-inference-of-neural-networks).

## Documentation

More extensive documentation can be found [here](https://c.harth-kitzerow.com/mkdocs-hpmpc/).

## Publications and Artifacts

HPMPC won the Best Artifact award at PoPETs 2025 and can be used to reproduce the results of the following publications:

| Publication | Artifact Evaluation | Citation |
| --- | --- | --- |
| [High-Throughput Secure Multiparty Computation with an Honest Majority in Various Network Settings](https://eprint.iacr.org/2024/386.pdf) | [Link](measurements/configs/artifacts/hpmpc/ARTIFACT_EVALUATION.md) | [BibTex](#high-throughput-secure-multiparty-computation-in-various-network-settings) |
| [PIGEON: A High Throughput Framework for Private Inference of Neural Networks using Secure Multiparty Computation](https://eprint.iacr.org/2024/1371.pdf) | [Link](measurements/configs/artifacts/pigeon/ARTIFACT_EVALUATION.md) | [BibTex](pigeon-a-high-throughput-framework-for-private-inference-of-neural-networks-using-secure-multiparty-computation) |
| [Truncation Untangled: Scaling Fixed-Point Arithmetic for Privacy-Preserving Machine Learning to Large Models and Datasets](https://eprint.iacr.org/2024/1953.pdf) | [Link](measurements/configs/artifacts/truncation/ARTIFACT_EVALUATION.md) | [BibTex](#truncation-untangled-scaling-fixed-point-arithmetic-for-privacy-preserving-machine-learning-to-large-models-and-datasets) |



## Getting Started

TLDR instructions can be found [here](#tldr).

You can use the provided Dockerfile or set up the project manually. The only dependency is OpenSSL. Neural networks and other functions with matrix operations also require the Eigen library.
```bash
#Install Dependencies:
sudo apt install libssl-dev libeigen3-dev
```

```bash
#Run with Docker
docker build -t hpmpc .
#Run each command in different terminals or different machines
 docker run -it --network host --cap-add=NET_ADMIN --name p0 hpmpc
 docker run -it --network host --cap-add=NET_ADMIN --name p1 hpmpc
 docker run -it --network host --cap-add=NET_ADMIN --name p2 hpmpc
 docker run -it --network host --cap-add=NET_ADMIN --name p3 hpmpc
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
    # Option 3: Follow steps from PyGEON README to use pretrained PyTorch models on ImageNet
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

### Network Shaping

To simulate real world network settings you can specify a json file with network configuarations. Examples based on real-world measurements are found in `measurements/network_shaping`.
```json
{
  "name": "CMAN",
  "latencies": [
    [2.318, 1.244, 1.432],
    [2.394, 1.088, 2.020],
    [1.232, 1.091, 1.883],
    [1.418, 2.054, 1.892]
  ],
  "bandwidth": [
    [137, 1532, 417],
    [139, 1144, 312],
    [1550, 1023, 602],
    [444, 389, 609]
  ]
}
```
Each row in `latencies` and `bandwidth` corresponds to a party. The values are in milliseconds and Mbps respectively. The third row would for instance be parsed as party 2 having a latency of 1.232ms to party 0, 1.091ms to party 1, and 1.883ms to party 3. The bandwidth is parsed in the same way.
To apply the bandwidths from a config file, run the following script.
```bash
./measurements/network_shaping/shape_network.sh -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> -l 2 -f measurements/network_shaping/<config_file>.json
```
The `-l 2` flag divides the applied latencies by 2 to avoid that both round trip times between two parties are added up.
This option should be used for all provided json files and if the latencies are measured with the ping utility.

The resulting network shaping can be verified by running the following script on all nodes simultaneously.
The script sends and receives data between all parties in parallel and thus may deviate from pair-wise measurements but therefore might be more accurate to represent MPC communication.
Note that some deviations in network shaping and verification are expected.
```bash
./scripts/measure_connection.sh -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3>
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

### Other Compile Errors

Internal g++ or clang errors might be fixed by updating the compiler to a newer version.

If reading input files fails, adding `-lstdc++fs` to the Makefile compile flags may resolve the issue.

### Increase Accuracy of Neural Network Inference

If you encounter issues regarding the accuracy of neural network inference, the following options may increase accuracy.
- Increase the `BITLENGTH`.
- Increase or reduce the number of `FRACTIONAL` bits.
- Adjust the truncation strategy to `TRUNC_APPROACH=1` (Reduced Slack), `TRUNC_APPROACH=2` (Exact Truncation), `TRUNC_APPROACH=3` (Exact Truncation with no slack), `TRUNC_APPROACH=4` (Mixed Truncation)
, along with `TRUNC_THEN_MULT=0` and `TRUNC_DELAYED=1`. Note that truncation approaches 1 and 2 require setting `TRUNC_DELAYED=1`.
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
python3 measurements/run_config.py -p all measurements/configs/benchmarks/Multiprocessing.conf --override NUM_INPUTS=1000000 DATTYPE=512

#Distributed Setup, 3PC
python3 measurements/run_config.py -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> measurements/configs/benchmarks/Multiprocesssing.conf --override NUM_INPUTS=1000000 DATTYPE=512 PROTOCOL=1,2,3,5,6

#Distributed Setup, 4PC
python3 measurements/run_config.py -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> measurements/configs/benchmarks/Multiprocesssing.conf --override NUM_INPUTS=1000000 DATTYPE=512 PROTOCOL=7,8,9,10,11,12
```

#### Run LeNet5 on MNIST locally with batch size 24 using SPLITROLES
```bash
# use DATTYPE=256 or DATTYPE=128 or DATTYPE=64 for CPUs without AVX/SSE support.

# 3PC
python3 measurements/run_config.py -p all measurements/configs/benchmarks/lenet.conf --override PROTOCOL=5 PROCESS_NUM=4 SPLITROLES=1

# 4PC
python3 measurements/run_config.py -p all measurements/configs/benchmarks/lenet.conf --override PROTOCOL=12 PROCESS_NUM=1 SPLITROLES=3
```

#### Run various neural network models in a distributed setting on ImageNet with 3 iterations per run and SPLITROLES (Requires server-grade hardware)
```bash
# use DATTYPE=256 or DATTYPE=128 or DATTYPE=64 for CPUs without AVX/SSE support.

# 3PC
python3 measurements/run_config.py -i 3 -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> measurements/configs/benchmarks/imagenetmodels.conf --override PROTOCOL=5 PROCESS_NUM=4 SPLITROLES=1

# 4PC
python3 measurements/run_config.py -i 3 -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> measurements/configs/benchmarks/imagenetmodels.conf --override PROTOCOL=12 PROCESS_NUM=1 SPLITROLES=3
```

#### Parse the results
```bash
python3 measurements/parse_logs.py measurements/logs/ # results are stored as `.csv` in measurements/logs/
```


## Executing MP-SPDZ Bytecode 

HPMPC can execute bytecode generated by the MP-SPDZ compiler.
It is possible to run computation with bytecode compiled by [MP-SPDZ](https://github.com/data61/MP-SPDZ). Most instructions of [MP-SPDZ](https://github.com/data61/MP-SPDZ/releases?page=1) 0.3.8 are supported. 
Note that some MP-SPDZ instructions may show significant performance improvements when using the HPMPC framework, while others may show a performance decrease when workarounds are used to support MP-SPDZ bytecode with HPMPC functions.


### Setup and successfully run HP-MPC with MP-SPDZ

1. [Install MP-SPDZ](#install-the-mp-spdz-compiler)
2. [Required setup to run HP-MPC with MP-SPDZ as frontend](#setup)
3. [Define the input used for computation](#input)
4. [Add/Run your own functions (.mpc) files using HP-MPC](#run-your-own-functions)

### For developers:

1. [Add support for MP-SPDZ Instructions that are not yet implemented](#add-support-for-mp-spdz-instructions-not-yet-implemented)
2. [Formatting for source files](#formatting)


### Install the MP-SPDZ compiler

You need to install [MP-SPDZ](https://github.com/data61/MP-SPDZ/releases?page=1) 0.3.8 to compile your `<filename>.mpc`
```sh
wget https://github.com/data61/MP-SPDZ/releases/download/v0.3.8/mp-spdz-0.3.8.tar.xz
tar xvf mp-spdz-0.3.8.tar.xz
```

### Setup

#### Dependencies

For some MP-SPDZ programs [PyTorch](https://pytorch.org/) or [numpy](https://numpy.org/) are required. To install them you can use [requirements.txt](/MP-SPDZ/requirements.txt)

```sh
pip install -r ./MP-SPDZ/requirements.txt
```

#### 1. Create required Directories

In the `HPMPC` main directory, create two directories in `MP-SPDZ/`: `Schedules` for the schedule file and `Bytecodes` for the respective bytecode file.

```sh
mkdir -p "./MP-SPDZ/Schedules" "./MP-SPDZ/Bytecodes"
```

#### 2. Copy .mpc files and Compile them

In order to compile the `.mpc` files in `MP-SPDZ/Functions/` you have to:

Assuming [MP-SPDZ](https://github.com/data61/MP-SPDZ) is installed at `$MPSPDZ`, copy the desired `<file>.mpc` into `"$MPSPDZ"/Programs/Source` and compile them using their compiler with the bit length you intent to use.

```sh 
cp "./MP-SPDZ/Functions/<file.mpc>" "$MPSPDZ"/Programs/Source/ 
```

- For arithmetic programs using [Additive_Shares](/datatypes/Additive_Share.hpp) use:

```sh
cd "$MPSDZ" && ./compile.py -K LTZ,EQZ -R "<BITLENGTH>" "<file>"
```

where `BITLENGTH` is the integer bit-length you want to use for the computation.

- For boolean programs using [XOR_Shares](/protocols/XOR_Share.hpp)

```sh
cd "$MPSDZ" && ./compile.py -K LTZ,EQZ -B "<bit-length>" "<file>"
```

where `<bit-length>` can be anything **EXCEPT** when operating on int-types (cint, int) $\to$ `<bit-length>` <= `64`

**NOTE**
Adding:
- `-D/--dead-code-elimination` might decrease the size of the bytecode
- `-O/--optimize-hard` might even slow down execution as LTZ/EQZ are replaced by a bit-decomposition approach using random secret bits that are not yet properly supported
- `--budget=<num> -l/--flow-optimization` will prevent the compiler from completely unrolling every loop $\implies$ faster compilation and smaller bytecode but might slow down execution

#### 3. Move the bytecode/schedule file into the respective directory

To execute the compiled MP-SPDZ programs with HPMPC, move them to the respective directories in `HPMPC`. For your own functions, you can use the filename `custom.sch` for easier setup.

```sh 
mv "$MPSDZ/Programs/Schedules/*" "./MP-SPDZ/Schedules/"
mv "$MPSDZ/Programs/Bytecode/*" "./MP-SPDZ/Bytecodes/"
```

#### 4. Run computation

Make sure to use the correct `FUNCTION_IDENTIFIER` and `BITLENGTH`. The following example executes the `tutorial.mpc` file locally with `BITLENGTH=32.
```sh
make -j PARTY=all PROTOCOL=5 FUNCTION_IDENTIFIER=500 BITLENGTH=32
./run.sh -p all -n 3
```

### Run the example functions

We provide multiple example functions in [MP-SPDZ/Functions/](/MP-SPDZ/Functions/).
Mappings of `.mpc` files to `FUNCTION_IDENTIFIER` can be found in [programs/functions/mpspdz.hpp](programs/functions/mpspdz.hpp).
Note that many functions require specifying a number of operations when compiling the bytecode with the MP-SPDZ compiler or need input files to be present in [MP-SPDZ/Input/](/MP-SPDZ/Input/) when executing the program.

`FUNCTION_IDENTIFIER` | `.mpc`
----------------------|-------
`500` | [tutorial.mpc](/MP-SPDZ/Functions/tutorial.mpc)
`501` | `custom.mpc` (can be used for your own functions)
`502` | [add.mpc](/MP-SPDZ/Functions/add.mpc)
`503` | [mul.mpc](/MP-SPDZ/Functions/mul.mpc)
`504` | [mul_fix.mpc](/MP-SPDZ/Functions/mul_fix.mpc) (make sure that the precision is set correctly)
`505` | [int_test.mpc](/MP-SPDZ/Functions/int_test.mpc)/[int_test_32.mpc](/MP-SPDZ/Functions/int_test_32.mpc) (depending on `BITLENGTH` (`64` or `32`)) can be used to test public integer operations
`506-534` | Various functions used for benchmarks (see [here](/MP-SPDZ/Functions/bench)).


### Input

Input will be read from the files in [MP-SPDZ/Input/](/MP-SPDZ/Input/)

- public input will be read from [PUB-INPUT](/MP-SPDZ/Input/PUB-INPUT)
- private input will be read from `INPUT-P<player_number>-0-<vec>`
    - `<player_number>`: is the number associate with a specific player.
    - `<vec>`: is always `0`
        - except for SIMD circuits:
            - it is between [`0` - `DATTYPE/BITLENGTH`]
            - for all numbers between [`0` - `DATTYPE/BITLENGTH`], there must
            exist an input-file (otherwise there are not enough numbers to store
            in a SIMD register)

An example for formatting can be seen in [Input-P0-0-0](/MP-SPDZ/Input/Input-P0-0-0) which is used for:
- private input from party `0`
- from main thread (thread `0`)
- for the first number of the vectorization (`0`)


### Run your own functions

As with other `.mpc` files, copy the bytecode file and schedule file into the correct Directory (`./MP-SPDZ/Schedules/`, `./MP-SPDZ/Bytecodes/` respectively).
Make sure that for both MP-SPDZ and HPMPC you are using the same bitlength for compilation.

#### Using function `501`/`custom.mpc`

Rename the schedule file to `custom.sch` and compile with `FUNCTION_IDENTIFIER = 501`
```sh
mv "./MP-SPDZ/Schedules/<file>.sch" "./MP-SPDZ/Schedules/custom.sch"
make -j PARTY=<party_id> PROTOCOL=<protocol_id> FUNCTION_IDENTIFIER=501 BITLENGTH=<bit-length>
```

With `FUNCTION_IDENTIFIER` set to `501` the virtual machine will search for a file `custom.sch` in `./MP-SPDZ/Schedules/`

- **NOTE**: bytecode file(-s) do not have to be renamed as their name is referenced in the respective schedule-file

#### Adding a new function using mpspdz.hpp

In [programs/functions/mpspdz.hpp](/programs/functions/mpspdz.hpp) are all currently supported functions you'll notice the only thing that changes is the path of the `<schedule-file>`

To add a new `FUNCTION_IDENTIFIER`

1. Create a new header file in [programs](/programs/) you may use [programs/mp-spdz_interpreter_template.hpp](/programs/mp-spdz_interpreter_template.hpp)
2. Choose a number `<your-num>` for (`FUNCTION_IDENTFIER`)
    - make sure it does not exist yet (see [protocol_executer.hpp](/protocol_executer.hpp))
    - make sure that in [protocol_executer.hpp](/protocol_executer.hpp) the correct header file is included

You can do so by adding the following lines to [protocol_executre.hpp](/protocol_executer.hpp)
```cpp
#elif FUNCTION_IDENTIFIER == `<your-identifier>`
#include "programs/<your header file>.hpp"
```

3. Define the function for a given `FUNCTION_IDENTIFIER`:
    - when using the template make sure to replace the `FUNCTION_IDENTIFIER`, the function name and path to the `<schedule-file>`


### Add support for MP-SPDZ instructions not yet implemented

1. Add the instruction and its opcode in [MP-SPDZ/lib/Constants.hpp](/MP-SPDZ/lib/Constants.hpp) to the `IR::Opcode` enum class but also to `IR::valid_opcodes`

2. To read the parameters from the bytecode-file add a case to the switch statement in the `IR::Program::load_program([...]);` function in [MP-SPDZ/lib/Program.hpp](/MP-SPDZ/lib/Program.hpp). You may use:
    - `read_int(fd)` to read a 32-bit Integer
    - `read_long(fd)` to read a 64-bit Integer
    - `fd` (std::ifstream) if more/less bytes are required (keep in mind the bytcode uses big-endian)

To add the parameters to the parameter list of the current instruction you may use `inst.add_reg(<num>)`, where:
- `inst` is the current instruction (see the [`Instruction`](/MP-SPDZ/lib/Program.hpp) class)
- `<num>` is of type `int`

**OR** use `inst.add_immediate(<num>)` for a constant 64-bit integer some instructions may require.

This program also expects this function to update the greatest compile-time address that the compiler tries to access. Since the size of the registers is only set once and only a few instructions check if the registers have enough memory. Use:

- `update_max_reg(<type>, <address>, <opcode>)`: to update the maximum register address
    - `<type>`: is the type of the register this instruction tries to access
    - `<address>`: the maximum address the instruction tries to access
    - `<opcode>`: can be used for debugging

- `m.update_max_mem(<type>, <address>)`: to update the maximum memory address
    - `<type>`: is the type of the memory cell this instruction tries to access
    - `<address>`: the maximum memory address the instruction tries to access

3. To add functionality add the Opcode to the switch statment in `IR::Instruction::execute()` ([MP-SPDZ/lib/Program.hpp](/MP-SPDZ/lib/Program.hpp))

- for more complex instructions consider adding a new function to `IR::Program`
- registers can be accessed via `p.<type>_register[<address>]`, where `<type>` is:
    - `s` for secret `Additive_Share`s
    - `c` for clear integeres of length `BITLENGTH`
    - `i` for 64-bit integers
    - `sb` for boolean registers (one cell holds 64-`XOR_Share`s)
    - `cb` clear bit registers, represented by 64-bit integers (one cell can hold 64-bits) (may be vectorized with SIMD but is not guaranteed depending on the `BITLENGTH`)
- memory can be accessed via `m.<type>_mem[<address>]` where `<type>` is the same as for registers except 64-bit integers use `ci` instead of `i` (I do not know why I did this)

You may also look at [this commit](https://github.com/aSlunk/hpmpc/commit/d7fd4ec47c58fac9344682701e9052bcf52ef95b) which adds `INPUTPERSONAL` (`0xf5`) and `FIXINPUT` (`0xe8`)

### Formatting

You can use/change the clang-format file in [MP-SPDZ/](/MP-SPDZ/.clang-format)

```sh
clang-format --style=file:MP-SPDZ/.clang-format -i MP-SPDZ/lib/**/*.hpp MP-SPDZ/lib/**/*.cpp
```


## References

Our framework utilizes the following third-party implementations.
- Architecture-specific headers for vectorization and Bitslicing adapted from [USUBA](https://github.com/usubalang/usuba), [MIT LICENSE](https://raw.githubusercontent.com/usubalang/usuba/main/LICENSE).
- AES-NI implementation adapted from [AES-Brute-Force](https://github.com/sebastien-riou/aes-brute-force), [Apache 2.0 LICENSE](https://raw.githubusercontent.com/sebastien-riou/aes-brute-force/master/LICENSE)
- SHA-256 implementation adapted from [SHA-Intrinsics](https://github.com/noloader/SHA-Intrinsics/tree/master), No License.
- CUDA GEMM and Convolution implementation adapted from [Cutlass](https://github.com/NVIDIA/cutlass), [LICENSE](https://raw.githubusercontent.com/NVIDIA/cutlass/main/LICENSE.txt) and [Piranha](https://github.com/ucbrise/piranha/tree/main), [MIT LICENSE](https://raw.githubusercontent.com/ucbrise/piranha/main/LICENSE).
- Neural Network Inference engine adapted from [SimpleNN](https://github.com/stnamjef/SimpleNN), [MIT LICENSE](https://raw.githubusercontent.com/stnamjef/SimpleNN/master/LICENSE).

## BibTeX

<!-- | [High-Throughput Secure Multiparty Computation with an Honest Majority in Various Network Settings](https://eprint.iacr.org/2024/386.pdf) | [Link](#measurements/configs/artifacts/hpmpc/ARTIFACT_EVALUATION.md) | [BibTeX](#highthroughputmpc) | -->
<!-- | [PIGEON: A High Throughput Framework for Private Inference of Neural Networks using Secure Multiparty Computation](https://eprint.iacr.org/2024/1371.pdf) | [Link](#measurements/configs/artifacts/pigeon/ARTIFACT_EVALUATION.md) | [BibTeX](#pigeon) | -->
<!-- | [SoK: Truncation Untangled: Scaling Fixed-Point Arithmetic for Privacy-Preserving Machine Learning to Large Models and Datasets](https://eprint.iacr.org/2024/1953) | [Link](#measurements/configs/artifacts/truncation/ARTIFACT_EVALUATION.md) | [BibTeX](#truncationuntangled) | -->
### High Throughput Secure Multiparty Computation in Various Network Settings
```bibtex
@article{HighThroughputMPC,
  author  = {Christopher Harth-Kitzerow and Ajith Suresh and Yongqin Wang and Hossein Yalame and Georg Carle and Murali Annavaram},
  title   = {High-Throughput Secure Multiparty Computation with an Honest Majority in Various Network Settings},
  journal = {Proceedings on Privacy Enhancing Technologies},
  year    = {2025},
  volume  = {2025},
  number  = {1},
  pages   = {250--272},
  url     = {https://doi.org/10.56553/popets-2025-0015}
}
```
### PIGEON: A High Throughput Framework for Private Inference of Neural Networks using Secure Multiparty Computation
```bibtex
@article{PIGEON,
  author  = {Christopher Harth-Kitzerow and Yongqin Wang and Rachit Rajat and Georg Carle and Murali Annavaram},
  title   = {PIGEON: A High Throughput Framework for Private Inference of Neural Networks using Secure Multiparty Computation},
  journal = {Proceedings on Privacy Enhancing Technologies},
  year    = {2025},
  volume  = {2025},
  number  = {3},
  pages   = {88--105},
  url     = {https://doi.org/10.56553/popets-2025-0090}
}
```
### Truncation Untangled: Scaling Fixed-Point Arithmetic for Privacy-Preserving Machine Learning to Large Models and Datasets
```bibtex
@article{TruncationUntangled,
  author  = {Christopher Harth-Kitzerow and Ajith Suresh and Georg Carle},
  title   = {SoK: Truncation Untangled: Scaling Fixed-Point Arithmetic for Privacy-Preserving Machine Learning to Large Models and Datasets},
  journal = {Proceedings on Privacy Enhancing Technologies},
  year    = {2025},
  volume  = {2025},
  number  = {4},
  pages   = {369--391},
  url     = {https://doi.org/10.56553/popets-2025-0135}
}
```


