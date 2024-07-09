# HPMPC: High-Performance Implementation of Secure Multiparty Computation (MPC) Protocols

HPMPC implements multiple MPC protocols and provides a high-level C++ interface to define functions and use cases.
Out of the Box, the framework supports computation in the boolean and arithmetic domain, mixed circuits, and fixed point arithmetic. 
Neural networks models can be imported from PyTorch as part of PIGEON (Private Inference of Neural Networks).


## Getting Started

You can use the provided Dockerfile or set up the project manually. The only dependencies is OpenSSL. Neural Networks and other functions with matrix operations also require the Eigen library. Install on your target system, for instance via ```apt install libssl-dev libeigen3-dev```. 

To use GPU acceleration for matrix multiplication and convolutions, you need an NVIDIA GPU and the nvcc compiler. You also need a copy of the [CUTLASS](https://github.com/NVIDIA/cutlass) library. You can then set up the project as follows.
```bash
git clone https://github.com/chart21/hpmpc.git
sudo apt install libssl-dev libeigen3-dev # Install dependencies 
cd hpmpc
make -j PARTY=all FUNCTION_IDENTIFIER=54 PROTOCOL=5 # Compile executables for protocol Trio (5) for all parties and a unit tests for basic primitives (function 54)
scripts/run.sh -p all -n 3 # Run three parties locally
```

To use GPU acceleration, you also need to execute the following commands. To find out your GPU architecture, refer to [this oerview](https://developer.nvidia.com/cuda-gpus).
```bash
# Dependencies for GPU acceleration
git clone https://github.com/NVIDIA/cutlass.git
# Compile standalone executable for GPU acceleration
cd core/cuda
make -j arch=sm_89 CUDA_PATH=/usr/local/cuda CUTLASS_PATH=/home/user/cutlass # Replace with you architecture, nvcc path and CUTLASS path
cd ../..
# Compile and run MPC executables with GPU acceleration
make -j PARTY=all FUNCTION_IDENTIFIER=57 PROTOCOL=12 USE_CUDA_GEMM=2 # Compile executables for protocol Quad (12) for all parties and a unit tests for matrix multiplication (function 54) with GPU acceleration (USE_CUDA_GEMM=2)
scripts/run.sh -p all -n 4 # Run four parties locally
```

After setting up the framework on each node of a distributed setup, you can run the following commands to run the MPC protocol on a distributed setup.
```bash
make -j PARTY=<party_id>
scripts/run.sh -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> # Run the MPC protocol on a distributed setup. For 3PC protocols, the -d flag has no effect.
```

SplitRoles compiles multiple executables per player to perform load balancing. Running a protocol with Split-Roles can be done by running the following commands. More information on Split-Roles can be found in the section "Scaling MPC to Billions of Gates per Second".
```bash
make -j PARTY=<party_id> SPLITROLES=1 # Compile multiple executables for a 3PC protocol with Split-Roles
scripts/run.sh -s 1 -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> 
``` 

SplitRoles supports multi-GPU setups. To run a protocol with multiple GPUs, you can run the following commands.
```bash 
make -j USE_CUDA_GEMM=2
scripts/run.sh -p <party_id> -s 1 -g 6 # Utilize 6 GPUs for the computation
```

## Project Structure
The framework uses a modular architecture with the following components.
- Core: Implements communication between parties, cryptographic primitives, and techniques for hardware acceleration. Uses Bitslicing, Vectorization, GPU acceleration, and hardware instruction for cryptographic primtitives to accelerate local computation required by the MPC protocols.
- Protocols: Implements MPC protocols and protocol-specific primitives. Each protocol utilizes high-level operations provided by `Core` for commonly used operations such as sampling shared random numbers or exchanging messages.
- Datatypes: Implements different datatypes that serve as a high-level templated interface to compute on MPC shares in a generic manner with overloaded operators.
- Programs: Implements high-level functions, routines and use cases using the custom `datatypes`. Implements several MPC-generic functions such as matrix multiplication and comparisons.
- NN: Implements a templated neural network inference engine that performs the forward pass of a CNN by relying on high-level MPC-generic functions provided by `Programs`. Models and datasets can be exported from PyTorch.

New MPC protocol can be added to `protocols/` by using the operations provided by `Core`.
New functions can be added to `programs/` by using the operations supported by `Datatypes`.
New Model architectures can be added to `nn/FlexNN/architectures/` by using our high-level interface to define model architectures provided by `nn/FlexNN`.

## Scaling MPC to Billions of Gates per Second

The framework offers multiple tweaks to accelerate MPC computation. The following are the most important settings that can be adjusted in `config.h` or by setting the respective flags when compiling.
- Concurrency (`DATTYPE`, `PROCESS_NUM`): `DATTYPE` sets the registersize to use for Bitslicing and Vectorization. Bitslicing and vectorization is supported by the framework on various archiectures (SSE,AVX-2,AVX-512). If your CPU supports AVX-512, seeting `DATTYPE` to 512 vectorize all integers and boolean variables to fully utilize the wide registers. `PROCESS_NUM` sets the number of processes to use for parallel computation.
- Hardware Acceleration (`RANDOM_ALGORITHM`, `USE_SSL_AES`, `ARM`, `USE_CUDA_GEMM`): Different approaches for efficiently implementing cryptographic primitives on various hardware architectures. Matrix operations can be accelerated using CUDA.
- Tweaks (`SEND_BUFFER`, `RECV_BUFFER`, `VERIFY_BUFFER`): Setting buffer sizes for communication and sha hashing to verify messages can accelerate workloads. The default settings should provide a good starting point for most settings.
- Preprocessing (`PRE`): Some protocols support a preprocessing phase that can be enabled to accelerate the online phase. 
- SPLITROLES (`SPLITROLES`): By using the SPLITROLES flag when compiling, the framework compiles n! excutables for a n-PC protocol where each executable has a different player assignment. This allows load balance the communication and computation between the nodes. SPLITROLES 1 compiles all executables for a 3PC protocol, SPLITROLES 2 compiles all executables for a 3PC protocol in a setting with four nodes, and SPLITROLES 3 compiles all executables for a 4PC protocol.

For nodes equipped with a 32-core AVX-512 CPU, a CUDA-enabled GPU, the following example may compile an optimized executables in a distributed setup. Note that this example inherently vectorizes the computation `PROCESS_NUM x DATTYPE/BITLENGTH x SPLITROLES_Factor` times. 
```bash
make -j PARTY=<node_id> FUNCTION_IDENTIFIER=<function_id> PROTOCOL=12 DATTYPE=512 PROCESS_NUM=32 RANDOM_ALGORITHM=2 USE_SSL_AES=0 ARM=0 USE_CUDA_GEMM=2 SEND_BUFFER=10000 RECV_BUFFER=10000 VERIFY_BUFFER=1 PRE=1 SPLITROLES=3.
```

## Protocols

The following protocols currently have full support for all implemented primitives and functions.
- 3PC (semi-honest): Trio (Protocol 5), Trusted Third Party (Protocol 6)
- 4PC Quad (malicious): (Protocol 12), Trusted Third Party (Protocol 7)

The following protocols currently have full support for all basic primitives.
- 3PC (semi-honest): Sharemind (Protocol 1), Replicated (Protocol 2), ASTRA (Protocol 3)
- 4PC (malicious): Tetrad (Protocol 8), Fantastic Four (Protocol 9)

Trio, ASTRA, Quad, and Tetrad support a Preprocessing phase. The preprocessing phase can be enabled in `config.h` or by setting `PRE=1` when compiling. New protocols can be added to `protocols/`and adding a protocol ID to `protocols/Protocols.h`. 

## Functions

Out of the box, the framework provides multiple high-level functions that operate on either Additive or Boolean shares. `programs/functions/` contains unit tests and benchmarks for these functions. An overview of which id corresponds to which function can be found in `protocol_executer.hpp`. In the following, we provide a brief overview of the functions that are currently implemented.
 - Basic Primitives: Secret Sharing, Reconstruction, Addition, Multiplication, Division, etc. 
 - Fixed Point Arithmetic: Fixed Point Addition, Multiplication, Truncation Division, etc.
 - Matrix Operations: Matrix Multiplication, Dot Product, etc.
 - Multi-input Operations: Multi-input Multiplication, Multi-input Scalar Products
 - Comparisons: EQZ, LTZ, MAX, MIN, Argmax, Argmin
 - Use Cases (Benchmarking): Set Intersection, Auction, AES, Logistic Regression, etc.
 - Neural Networks: Forward Pass of CNN, ReLU, Softmax, Pooling, etc.

To implement a custom function, these functions can be used as building blocks. In `programs/tutorials/`, we provide tutorials on how to use different functions. New functions can be added by first implementing the function in `programs/functions/` and then adding a FUNCTION_IDENTIFIER to `protocol_executer.hpp`. The tutorial `programs/tutorials/YourFirstProgram.hpp` should get you started after following the other tutorials.


## The Vectorized Programming Model
Scaling MPC requires a high degree of parallelism to overcome network latency bottlenecks.
HPMPC's architecture is designed to utilize hardware resources proportionally to the degree of parallelism required by the MPC workload. 
By increasing load balancing, registersizes, or number of processes, the framework executes multiple instances of the same function in parallel.
For instance, an arithmetic program that computes a function on 32-bit integers is executed 512 times in parallel by using 32 processes and 512-bit registers.
Similarly, a boolean program that computes a single boolean function is executed 512x32=16384 times in parallel due to Bitslicing.
For mixed circuits, HPMPC automatically groups blocks of arithmetic shares before share conversion to handle these different degrees of parallelism.
The degree of parallelism for arithmetic and mixed circuits can be calculated as `PROCESS_NUM x DATTYPE/BITLENGTH x SPLITROLES_Factor` while for boolean circuits it is `PROCESS_NUM x DATTYPE x SPLITROLES_Factor`.

Thus, setting SPLITROLES=1, PROCESS_NUM=4, and DATTYPE=256 to a program computing 10 AES blocks (boolean circuit) will actually compute 6x4x256x10=61440 AES blocks in parallel by fully utilizing the available hardware resources, while setting DATTYPE=1, SplitRoles=0, and PROCESS_NUM=1 will compute 10 AES blocks on a single core without vectorization. Setting SPLITROLES=1, PROCESS_NUM=4, and DATTYPE=256 to a program computing a single neural network inference (mixed circuit) will compute 6x4x256/32=192 samples in parallel, thus effictively using a batch size of 192.

## Executing MP-SPDZ Bytecode (Experimental)

HPMPC can execute bytecode generated by the MP-SPDZ compiler.


## PIGEON: Private Inference of Neural Networks

This project adds support for private inference of neural networks. The following components are part of PIGEON.

* [FlexNN](https://github.com/chart21/flexNN/tree/hpmpc): A templated neural network inference engine that performs the forward pass of a CNN generically.
* `Programs/functions` contains MPC-generic implementations of functions such as ReLU.
* `Protocols` Implements the MPC protocols and primitives that are required by `Programs/functions`.
* [Pygeon](https://github.com/chart21/Pygeon): Python scripts for exporting models and datsets from PyTorch to the inference engine. 

All protocols that are fully supported by HPMPC can be used with PIGEON. To get started with PIGEON, initialize the submodules to set up FlexNN and Pygeon. 
> git submodule update --init --recursive

A full end-to-end example can be executed as follows. 
1. Use Pygeon to train a model in PyTorch and export its test labels, test images, and model parameters to `.bin` files using the provided scripts. Save the exported files to `nn/FlexNN/dataset` and `nn/FlexNN/model_zoo` respectively. At least one party (say P0) should have the dataset and at least one party (say P1) should have the model parameters. 
If the correct test accuracy should be calculated, all parties should have the test labels.
2. If it does not exist yet, add your model architecture to `nn/FlexNN/architectures/`.
3. If it does not exist yet, add a `FUNCTION_IDENTIFIER` for your model architecture and dataset dimensions in `Programs/functions/NN.hpp`.
4. Specify the path of your model, images, and labels in `config.txt`.
5. Run the following commands to compile the MPC protocol and run the model inference in a distributed setting.
```
make -j PARTY=<party_id> FUNCTION_IDENTIFIER=<function_id> DATAOWNER=P_0 MODELOWNER=P_1 #execute on each node
scripts/run_distributed.sh -p <party_id> -a <ip_address_party_0> -b <ip_address_party_1> -c <ip_address_party_2> -d <ip_address_party_3> 
```

PIGEON provides several options to modify the inference. The following are the most important settings that can be adjusted in `config.h` or by setting the respective flags when compiling.
- Bits (`BITLENGTH`, `FRACTIONAL`): The number of bits used for the total bitlength and the fractional part respectively.
- Truncation (`TRUNC_APPROACH`,`TRUNC_THEN_MULT`,`TRUNC_DELAYED`): There are multiple approaches to truncation. The default approach is to truncate Probabilistically after each multiplication. The different approaches allow switching between several truncation strategies.
- ReLU (`REDUCED_BITLENGTH_m, REDUCED_BITLENGTH_k`): ReLU can be evaluated probabilistically by reducing its bitwidth to save communciation and computation. The default setting is to evaluate ReLU with the same bitwidth as the rest of the computation. 
- Secrecy (`PUBLIC_WEIGHTS`, `COMPUTE_ARGMAX`): The weights can be public or private. The final argmax computation may not be required if parties should learn the probabilities of each class.
- Optimizations ('ONLINE_OPTIMIZED', 'BANDWIDTH_OPTIMIZED'): All layers requiring sign bit extraction such as ReLU, Maxpooling, and Argmax can be evaluated with different types of adders. These have different trade-offs in terms of online/preprocessing communication as well as round complexity and communication complexity.
- Other Optimizations: All default optimizations of HPMPC such as `SPLITROLES`, different buffers, and vectorization can be used with PIGEON. The parties automatically utilze the concurrency to perform inference on multiple independent samples from the dataset in parallel. To benchmark the inference without real data, `MODELOWNER` and `DATAOWNER` can be set to `-1`.

## Measurements

To automate benchmarks and tests of various function and protocols users can define `.conf` files in the `measurements/configs` directory. The following is an example of a configuration file runs a function with different number of inputs and different protocols.
```
PROTOCOL=8,9,12
NUM_INPUTS=10000,100000,1000000
FUNCTION_IDENTIFIER=1
DATTYPE=32
BITLENGTH=32
```

The config can then be executed with 
> measurements/run_config.sh -p <party_id> measurements/configs/<config_file>.conf

The outputs are stored in the `measurements/logs/` directory. The results can be parsed with the `measurements/parse_logs.py` script. The script can be executed as follows. The parsed result contains information such as communication, runtime, throughput, and if applicable unit test passed or accuracy achieved.
> python3 measurements/parse_logs.py measurements/logs/<log_file>.log


## Troubleshooting
HPMPC utilizes different hardware acceleration techniques for a range of hardware architectures. 
In case of compile errors, please check the following:
- Does your CPU support the AES-NI or VAES instruction set? If not, set `USE_SSL_AES=1` or `RANDOM_ALGORITHM=1` in `config.h`.
- Does your CPU support the `SHA` instruction set? If not, setting `ARM=1` in `config.h` may improe performance.
- Does your CPU support SSE (128-bit registers), AVX-2 (256-bit registers), or AVX-512 (512-bit registers)? If not, the maximum supported `DATTYPE` by your architecture is 64.
- Is your `BITLENGTH` compatible with the chosen DATTYPE? UINT64 (DATTYPE 64) does not allow any vectorization of integers with smaller `BITLENGTH`. 
Similarly, out of SSE, AVX-2, and AVX-512, only AVX-512 supports vectorization with a `BITLENGTH` 64. Setting BITLENGTH=DATTYPE for 32-bit or 64-bit inputs should work on all architectures.
- Does your GPU support CUDA? If not, compile without CUDA by setting `USE_CUDA_GEMM=0` in `config.h` or setting the respective flag when compiling. In case your CUDA-enabled GPU does not support datatypes such as UINT8, you can comment out the respective forward declaration in `core/cuda/conv_cutlass_int.cu` and `core/cuda/gemm_cutlass_int.cu`.
- Do you get low accuracy for Neural Network inference? To achieve high accuracy you may need to increase the `BITLENGTH`, increase of reduce the number of `FRACTIONAL` bits, or adjust the truncation strategy to `TRUNC_APPROACH=1` (REDUCED Slack`) or `TRUNC_APPROACH=2` (Exact Truncation), along with `TRUNC_THEN_MULT=1` and `TRUNC_DELAYED=1`. Note that truncation approaches 1 and 2 require setting `TRUNC_DELAYED=1`. Also check if `REDUCED_BITLENGTH_m` and `REDUCED_BITLENGTH_k` are set to 0 and `BITLENGTH` respectively. 
Also inspect the terminal output for any errors regarding reading the model or dataset. PIGEON uses dummy data or model parameters if the files are not found.

## References

Our framework utilizes the following third-party implementations.
- Architecture specific headers for vectorization and Bitslicing adapted from [USUBA](https://github.com/usubalang/usuba), [MIT LICENSE](https://raw.githubusercontent.com/usubalang/usuba/main/LICENSE).
- AES-NI implementation adapted from [AES-Brute-Force](https://github.com/sebastien-riou/aes-brute-force), [Apache 2.0 LICENSE](https://raw.githubusercontent.com/sebastien-riou/aes-brute-force/master/LICENSE)
- SHA-256 implementation adapted from [SHA-Intrinsics](https://github.com/noloader/SHA-Intrinsics/tree/master), No License.
- CUDA GEMM and Convolution implementation adapted from [Cutlass](https://github.com/NVIDIA/cutlass), [LICENSE](https://raw.githubusercontent.com/NVIDIA/cutlass/main/LICENSE.txt) and [Piranha](https://github.com/ucbrise/piranha/tree/main), [MIT LICENSE](https://raw.githubusercontent.com/ucbrise/piranha/main/LICENSE).
- Neural Network Inference engine adapted from [SimpleNN](https://github.com/stnamjef/SimpleNN), [MIT LICENSE](https://raw.githubusercontent.com/stnamjef/SimpleNN/master/LICENSE).
