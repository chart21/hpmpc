# High-Throughput Implementation of Secure Multiparty Computation (MPC) protocols

This project implements multiple MPC protocols in the honest majority setting.
The following protocols are currently supported:
3-PC: Sharemind, Replicated, OECL (Ours), TTP
4-PC: Fantastic Four, Tetrad, OEC-MAL (Ours), TTP

## Getting Started

The only dependency is OpenSSL. Install on your target system, for instance via ```apt install libssl-dev```
You can select a protocol and function in the file `config.h`.
The following commands are a quick way to compile the current configuration for a 3-PC protocol and run all executables locally. This compiles all player executables using g++ with -Ofast and runs all executables on localhost on the same machine.
> ./scripts/config.sh -p all3

> ./scripts/run_loally.sh -n 3

For a 4-PC protocol, you can run.

> ./scripts/config.sh -p all4

> ./scripts/run_loally.sh -n 4

## Configuration and Compilation

Most configuration is contained in the file `config.h`. Here is an overview of the most important settings.

- PROTOCOL: Select the protocol to be used. Options are: 1: Sharemind, 2: Replicated, 3: Astra, 4: ODUP, 5: OECL (3-PC), 6: TTP (3-PC), 7: TTP (4-PC), 8: Tetrad, 9: FantasticFour, 10: OEC-MAL - Base (4-PC), 11: OEC-MAL - Het (4-PC), 12: OEC-MAL: Off/On (4-PC). 
- PARTY: Define the party ID for the current node, starting from 0. 
- FUNCTION_IDENTIFIER: Select the function for computation. Currently includes running secure search (0), AND/Multiplication gates (1-6). Also includes a debug function for boolean/arithemtic circuit to check if all basic functions of a protocol are working correctly (7-9). Matrix Operators require the Eigen library. Dot Products can be tested with function 14.
- DATTYPE: Register size to use for SIMD parallelization (Bitslicing/vectorization). Supported sizes are 0,8,32,64,128(SSE),256(AVX-2),512(AVX-512).
- PRE: Option to use a preprocessing phase. The following protocols support a preprocessing phase: 3,5,8,12
- NUM_INPUTS: Define the number of inputs.
- PROCESS_NUM: Number of parallel processes to use.
- USE_SSL: Use SSL encrypted communication? 
- SEND_BUFFER: Define how many gates should be buffered until sending them to the receiving party. 
- RECV_BUFFER: Define how many receiving messages should be buffered until the main thread is signaled that data is ready.
- VERIFY_BUFFER: Define how many messages should be buffered until a combined hash is calculated. 
- PRINT: Print additional info?

Changes can be applied either directly in the file or via running ```scripts/config.sh```. The script does not assume any default options but always uses the current configuration stored in `config.h` as a basis. In a distributed setup, ensure all configurations are the same (except PARTY).

```
  Script to configure and compile executables for a run.
  Only arguments you want to change have to be set.
   -n Number of elements"
   -b base_port: Needs to be the same for all players for successful networking (e.g. 6000)"
   -d Datatype used for slicing: 1(bool),8(uint8), 16 (uint16), 32(uint32), 64(uint64),128(SSE),256(AVX),512(AVX512)"
   -p Player ID (0/1/2/3). Use all3 or all4 for compiling for all players"
   -f Function Idenftifier (0: search, 1: AND, ...)"
   -c Pack Bool in Char before sending? (0/1). Only used with -d 1"
   -s MPC Protocol (1(Sharemind),2(Replicated),3(Astra),4(OEC DUP),5(OECL),6(TTP),...)"
   -i Initialize circuit separately (0) or at runtime (1)?"
   -l Include the Online Phase in this executable  (0/1)?"
   -e Compile circuit with Preprocessing phase before online phase  (0/1)?"
   -o Use additional assumptions to optimize the sharing phase? (0/1)"
   -u Number of players in total"
   -g Compile flags (other than standard)"
   -x Compiler (g++/clang++/..)"
   -h USE SSL? (0/1)"
   -j Number of parallel processes to use"
   -v Random Number Generator (0: XOR_Shift (insecure)/1 AES Bitsliced/2: AES_NI)"
   -t Total Timeout in seconds for attempting to connect to a player"
   -m VERIFY_BUFFER: How many gates should be buffered until verifying them? 0 means the data of an entire communication round is buffered "
   -k Timeout in milliseconds before attempting to connect again to a socket "
   -y SEND_BUFFER: How many gates should be buffered until sending them to the receiving party? 0 means the data of an entire communication round is buffered"
   -z RECV_BUFFER: How many receiving messages should be buffered until the main thread is signaled that data is ready? 0 means that all data of a communication round needs to be ready before the main thread is signaled.
```

The following configuration compiles an executable for P2, 1024 inputs, sliced 256 times in AVX-2 variables, using Protocol Replicated. All other configuarations are fetched from `config.h`.
> ./scripts/config.sh -p 2 -n 1024 -d 256 -s 2 

The following configuration uses the previous configuration but compiles an executable for all players. This is useful when running the parties on the same host.
> ./scripts/config.sh -p all3

The Split-Roles scripts transform a protocol into a homogeneous protocol by running multiple executables with different player assignments in parallel.

The following script compiles six executables of a 3-PC protocol for player 2 (all player combinations) to run a homogeneous 3-PC protocol on three nodes using Split-Roles.
> ./scripts/split-roles-3-compile.sh -p 2

The following script compiles 18 executables of a 3-PC protocol for player 3 to run a homogeneous 3-PC protocol on four nodes using Split-Roles.
> ./scripts/split-roles-3to4-compile.sh -p 3

The following script compiles 24 executables of a 4-PC protocol for player 0 to run a homogeneous 4-PC protocol on four nodes using Split-Roles.
> ./scripts/split-roles-4-compile.sh -p 0


### Execution

In a distributed setup, you need to specify the IP addresses for each party and run one executable on each node.

Execute P0 executable.
> ./run-P0.o IP_P1 IP_P2

Execute P1 executable.
> ./run-P1.o IP_P0 IP_P2

Execute P2 executable.
> ./run-P2.o IP_P0 IP_P1


Run Split-Roles (3) executables for Player 0.
> ./scripts/split-roles-3-execute.sh -p 0 -a IP_P0 -b IP_P1 -c IP_P2 -d IP_P3

To run all players locally on one machine, omit the IP addresses or set them to 127.0.0.1, and use -p all
> ./scripts/split-roles-3-execute.sh -p all


### Measuring Throughput

To measure the throughput of a specific function such as 64-bit mult, AND, or secure search, first, specify the function in `config.h`. Each process prints a time for running the computation and initialization. The initialization time measures setup costs, such as establishing a connection. When choosing multiple processes or Split-Roles, we recommend timing the whole script or executable with libraries such as /bin/time. To get accurate measurements with this approach, all nodes should connect simultaneously.

The throughput in AND gates per second for instance, can then be calculated as:

(NUM_INPUTS * DATTYPE * PROCESS_NUM * Split_Roles_Multiplier) / Total time measured.

When using the Split-Roles, Split_Roles_Multiplier is 6 for three-node settings and 24 for four-node settings. Otherwise, the multiplier is 1.


### Debugging

To check the correctness of a protocol, the debug function (function 7) checks the correctness of all basic gates in the boolean domain. Function 8-9 do the same in the arithmetic domain using a ring size of $2^{32}$ or $2^{64}$, respectively. Note that BITLENGTH and DATTYPE specified in `config.h` must be compatible with the computation domain. DATTYPE = 128 requires support for SSE, DATTYPE = 256 for AVX-2, and DATTYPE = 512, for AVX-512. The following combinations are valid for 32-bit computation: BITLENGTH = 32, DATTYPE = 32/128/256/512. The following combinations are valid for 64-bit computation: BITLENGTH = 64, DATTYPE = 64//256 (requires AVX-512)/512.
