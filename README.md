# High-Throughput Implementation of Secure Multiparty Computation (MPC) protocols

This project implements multiple MPC protocols in the honest majority setting.

## Getting Started

The only dependency is OpenSSL. Install on your target system, for instance via ```apt install libssl-dev```

The following command is a quick way to compile the current configuration for 3 players and run all executables locally.
> ./scripts/config.sh -p all3

> ./scripts/run_loally.sh -p all3


## Configuration and Compilation

Most configuration is contained in the file config.h. Here is an overview of the most important settings.

- PROTOCOL: Select the protocol to be used. Options are: 1: Sharemind, 2: Replicated, 3: Astra, 4: ODUP, 5: OURS (3-PC), 6: TTP (3-PC), 7: TTP (4-PC), 8: Tetrad, 9: FantasticFour, 10: Ours: Base (4-PC), 11: Ours: Het (4-PC), 12: Ours: Off/On (4-PC).
- PARTY: Define the party ID for the current node, starting from 0.
- FUNCTION_IDENTIFIER: Select the function for computation. Currently includes running secure search, AND gates, and 32-bit/64-bit multiplication gates. 
- DATTYPE: Register size to use for SIMD parallelization (Bitslicing/vectorization). Supported sizes are 0,8,32,64,128(SSE),256(AVX-2),512(AVX-512).
- PRE: Option to use a preprocessing phase. Currently supported by Protocols 4,5,12.
- NUM_INPUTS: Define the number of inputs.
- PROCESS_NUM: Number of parallel processes to use.
- USE_SSL: Use SSL encrypted communication? 
- SEND_BUFFER: Define how many gates should be buffered until sending them to the receiving party. 
- RECV_BUFFER: Define how many receiving messages should be buffered until the main thread is signaled that data is ready.
- VERIFY_BUFFER: Define how many messages should be buffered until a combined hash is calculated. 
- PRINT: Print additional info?

Changes can be applied either directly in the file or via running ```scripts/config.sh```

```
  Script to configure and compile executables for a run.
  Only arguments you want to change have to be set.
   -n Number of elements"
   -a Default Bitlength of integers"
   -b base_port: Needs to be the same for all players for successful networking (e.g. 6000)"
   -d Datatype used for slicing: 1(bool),8(char),64(uint64),128(SSE),256(AVX),512(AVX512)"
   -p Player ID (0/1/2). Use all3 or all4 for compiling for all players"
   -f Function Idenftifier (0: run, 2: AND, ...)"
   -c Pack Bool in Char before sending? (0/1). Only used with -d 1"
   -s MPC Protocol (1(Sharemind),2(Replicated),3(Astra),4(OEC DUP),5(OEC REP),6(TTP))"
   -i Initialize circuit separately (0) or at runtime (1)?"
   -l Include the Online Phase in this executable  (0/1)?"
   -e Compile circuit with Preprocessing phase before online phase  (0/1)?"
   -o Use additional assumptions to optimize the sharing phase? (0/1)"
   -u Number of players in total"
   -g Compile flags (other than standard)"
   -x Compiler (g++/clang++/..)"
   -h USE SSL? (0/1)"
   -j Number of parallel processes to use"
   -v Random Number Generator (0: XOR_Shift/1 AES Bitsliced/2: AES_NI)"
   -t Total Timeout in seconds for attempting to connect to a player"
   -m VERIFY_BUFFER: How many gates should be buffered until verifying them? 0 means the data of an entire communication round is buffered "
   -k Timeout in millisecond before attempting to connect again to a socket "
   -y SEND_BUFFER: How many gates should be buffered until sending them to the receiving party? 0 means the data of an entire communication round is buffered
"
   -z RECV_BUFFER: How many receiving messages should be buffered until the main thread is signaled that data is ready? 0 means that all data of a communication round needs to be ready before the main thread is signaled.
```

The following configuration compiles an executable for P2, 1024 inputs, sliced 256 times in an AVX-2 variable, using Protocol Replicated.
> ./scripts/config.sh -p 2 -n 1024 -d 256 -s 2 

The following configuration uses the previous configuration but compiles an executable for all players. This is useful when running the parties on the same host.
> ./scripts/config.sh -p all3

The following script compiles six executables for player 2 (all player combinations) to run all executables in parallel.
> ./scripts/split-roles-3-compile.sh -p 2

### Execution
Execute P0 executable.
> ./run-P0.o IP_P0 IP_P2

Execute P1 executable.
> ./run-P1.o IP_P0 IP_P2

Run Split-Roles (3) executables for Player 0.
> ./scripts/split-roles-3-execute.sh -p 0 -a IP_0 -b IP_1 -c IP_2 -d IP_3

To run all players locally on one machine, simply omit the IP addresses or set them to 127.0.0.1, and use -p all
> ./scripts/split-roles-3-execute.sh -p all
