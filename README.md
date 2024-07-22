# PIGEON: Private Inference of Neural Networks

This branch extends HPMPC with private inference capabilities. The framework is structured as follows.

* [FlexNN](https://github.com/chart21/flexNN/tree/hpmpc): A templated neural network inference engine that performs the forward pass of a CNN generically.
* `Programs/functions` contains MPC-generic implementations of functions such as ReLU.
* `Protocols` Implements the MPC protocols and primitives that are required by `Programs/functions`.
* [Pygeon](https://github.com/chart21/Pygeon): Python scripts for exporting models and datsets from PyTorch to the inference engine. 

The following protocols are currently fully supported by PIGEON.

3-PC: OECL (Ours, Protocol 5), TTP (Protocol 6)

4-PC: OEC-MAL (Ours, Protocol 12), TTP (Protocol 7)

## Getting Started

You can use the provided Dockerfile or set up the project manually.
The only dependencies are OpenSSL and Eigen. Install on your target system, for instance via ```apt install libssl-dev libeigen3-dev```.

First, initialize the submodules.
> git submodule update --init --recursive

To export a model or dataset from PyTorch use [Pygeon](https://github.com/chart21/pygeon) and save the resulting bin files to `SimpleNN/dataset` or `SimpleNN/model_zoo`. Then define the model architecture in `SimpleNN/architectures/`. Finally, specify your datasetfile and modelfile in `Programs/functions/NN.hpp` and if it does not exist, add a FUNCTION_IDENTIFIER for your function.
```
    cfg.save_dir = "./SimpleNN/model_zoo"; // Your model should be in this folder
    cfg.data_dir = "./SimpleNN/dataset"; // Your dataset should be in this folder
    cfg.image_file = "cifar10-test-images.bin"; // Your test images, exported by Pygeon
    cfg.label_file = "cifar10-test-labels.bin"; // Your test labels, exported by Pygeon
    cfg.pretrained = "AlexNet.bin"; // Your model parameters, exported by Pygeon
```

Existing networks are defined in `SimpleNN/architectures`. `Programs/functions/NN.hpp` includes a FUNCTION_IDENTIFIER for different model architectures and datasets (for instance 70 for RestNet18 on CIFAR-10). 
You can select a protocol and function in the file `config.h`. The config contains numerous settings. Here are just some examples: 
* Should the weights be public or private?
* Which party should share the dataset, Which party should share the model parameters?
* How many bits should be used for the fractional part, how many bits for the total bitlength?
* Which truncation approach should be used? Should ReLUs by default be evaluated with reduced Bitwidth?
* Should the inference be optimized for latency, bandwidth, Online Phase, or total execution time? Should a Preprocessing phase be used?


The following commands are a quick way to compile the current configuration for a 3-PC protocol and run all executables locally. This compiles all player executables using g++ with -Ofast and runs all executables on localhost on the same machine.
> ./scripts/config.sh -p all3

> ./scripts/run_loally.sh -n 3

For a 4-PC protocol, you can run.

> ./scripts/config.sh -p all4

> ./scripts/run_loally.sh -n 4

## Configuration and Compilation

Most configuration is contained in the file `config.h`. Take a careful look at its documentation and different options.

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


# MP-SPDZ

It is possible to run computation with bytecode compiled by [MP-SPDZ](https://github.com/data61/MP-SPDZ). As most of [MP-SPDZ](https://github.com/data61/MP-SPDZ/releases?page=1) 0.3.8 is supported. For this you have to checkout `mp-spdz`:

```sh
git switch mp-spdz
```

## Documentation files

### Setup and successfully run HP-MPC with MP-SPDZ

1. [Install MP-SPDZ](#install-the-mp-spdz-compiler)
2. [Required setup to run HP-MPC with MP-SPDZ as frontend](#setup)
3. [Define the input used for computation](#input)
4. [Add/Run your own functions (.mpc) files using HP-MPC](#run-your-own-functions)

### For developers:

1. [Add support for MP-SPDZ Instructions that are not yet implemented](#add-support-for-mp-spdz-instructions-not-yet-implemented)
2. [Formatting for source files](#formatting)


## Install the MP-SPDZ compiler

You need to install [MP-SPDZ](https://github.com/data61/MP-SPDZ/releases?page=1) 0.3.8 to compile your `<filename>.mpc`
```sh
wget https://github.com/data61/MP-SPDZ/releases/download/v0.3.8/mp-spdz-0.3.8.tar.xz
tar xvf mp-spdz-0.3.8.tar.xz
```

## Setup

### Dependencies

For some MP-SPDZ programs one might require [PyTorch](https://pytorch.org/) or [numpy](https://numpy.org/) to install them you can use our [requirements.txt](/MP-SPDZ/requirements.txt)

```sh
pip install -r ./MP-SPDZ/requirements.txt
```

### 1. Create required Directories

Create two directories in [MP-SPDZ/](/MP-SPDZ/): `Schedules` for the schedule file and `Bytecodes` for the respective bytecode file

```sh
mkdir -p "./MP-SPDZ/Schedules" "./MP-SPDZ/Bytecodes"
```

### 2. Copy .mpc files and Compile them 

In order to compile the `.mpc` files in [MP-SPDZ/Functions/](/MP-SPDZ/Functions/) you have to:

Assuming [MP-SPDZ](https://github.com/data61/MP-SPDZ) is installed at `$MPSPDZ`, copy the desired `<file>.mpc` into `"$MPSPDZ"/Programs/Source` and compile them using their compiler with the bit length specified in [config.h](/config.h).

```sh cp "./MP-SPDZ/Functions/<file.mpc>" "$MPSPDZ"/Programs/Source/ ```

- For arithmetic programs using [Additive_Shares](/protocols/Additive_Share.hpp) use:

```sh
cd "$MPSDZ" && ./compile.py -K LTZ,EQZ -R "<BITLENGTH>" "<file>"
```

where `BITLENGTH` is the same as defined in [config.h](/config.h)

- For boolean programs using [XOR_Shares](/protocols/XOR_Share.hpp)

```sh
cd "$MPSDZ" && ./compile.py -K LTZ,EQZ -B "<bit-length>" "<file>"
```

where `<bit-length>` can be anything **EXCEPT** when operating on int-types (cint, int) $\to$ `<bit-length>` <= `64`

**NOTE**
Adding:
- `-D/--dead-code-elimination` might decrease the size of the bytecode
- `-O/--optimize-hard` might even slow down execution as LTZ/EQZ are replaced by a bit-decomposition approach using random secret bits that are not yet properly supported
- `--budget=<num> -l/--flow-optimization` will prevent the compiler from completely unrolling every loop $\implies$ faster compilation and smaller bytecode **BUT** might slow down execution

### 3. Move the bytecode/schedule file into the respective directory

Rename the schedule file `custom.sch` and use `FUNCTION_IDENTFIER = 501` **OR** alternatively see [here](/MP-SPDZ/docs/add_new_functions.md)

```sh
mv "$MPSDZ/Programs/Schedules/<file>.sch" "./MP-SPDZ/Schedules/custom.sch"
mv "$MPSDZ/Programs/Bytecode/*" "./MP-SPDZ/Bytecodes/"
```

### 4. Run computation for 3 Players

Make sure to use the correct `FUNCTION_IDENTIFIER` and `BITLENGTH`:
```sh
./scripts/config.sh -p all3 -f "<FUNTION_IDENTIFIER>" -a "<BITLENGTH>"
./scripts/run_locally -n 3
```

## Run the example functions

Currently there are multiple example functions in [MP-SPDZ/Functions/](/MP-SPDZ/Functions/)

Mapping from `FUNCTION_IDENTIFIER` $\to$ `.mpc` file:

`FUNCTION_IDENTIFIER` | `.mpc`
----------------------|-------
`500` | [tutorial.mpc](/MP-SPDZ/Functions/tutorial.mpc)
`501` | `custom.mpc` (can be used for your own functions)
`502` | [add.mpc](/MP-SPDZ/Functions/add.mpc)
`503` | [mul.mpc](/MP-SPDZ/Functions/mul.mpc)
`504` | [mul_fix.mpc](/MP-SPDZ/Functions/mul_fix.mpc) (make sure that the precision is set correctly)
`505` | [int_test.mpc](/MP-SPDZ/Functions/int_test.mpc)/[int_test_32.mpc](/MP-SPDZ/Functions/int_test_32.mpc) (depending on `BITLENGTH` (`64` or `32`)) can be used to test public integer operations
`506-534` | functions used for benchmarks (see [here](/MP-SPDZ/Functions/bench)) mapping can be found in [MP-SPDZ/bench_scripts/measurement.sh](/MP-SPDZ/bench_scripts/measurement.sh)


## Input

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


## Run your own functions

As mentioned in [Setup](/MP-SPDZ/docs/run_hpmpc_with_MPSPDZ.md#setup) copy the bytecode file and schedule file into the correct Directory (`./MP-SPDZ/Schedules/`, `./MP-SPDZ/Bytecodes/` respectively)
make sure that for both MP-SPDZ and this project you are using the same bit length for compilation.

### Using function `501`/`custom.mpc`

Rename the schedule file to `custom.sch` and compile with `FUNCTION_IDENTIFIER = 501`
```sh
mv "./MP-SPDZ/Schedules/<file>.sch" "./MP-SPDZ/Schedules/custom.sch"
./scripts/config.sh -p all3 -f 501 -a "<BITLENGTH>"
```

With `FUNCTION_IDENTIFIER` set to `501` the virtual machine will search for a file `custom.sch` in `./MP-SPDZ/Schedules/`

- **NOTE**: bytecode file(-s) do not have to be renamed as their name is referenced in the respective schedule-file

### Adding a new function using mpspdz.hpp

In [programs/functions/mpspdz.hpp](/programs/functions/mpspdz.hpp) are all currently supported functions you'll notice the only thing that changes is the path of the `<schedule-file>`

To add a new `FUNCTION_IDENTIFIER`

1. Create a new header file in [programs](/programs/) you may use [programs/template.hpp](/programs/template.hpp)
2. Choose a number `<your-num>` (`FUNCTION_IDENTFIER`)
    - make sure it does **NOT** exist yet (see [protocol_executer.hpp](/protocol_executer.hpp))
    - make sure that in [protocol_executer.hpp](/protocol_executer.hpp) the correct header file is included

You can do so by adding the following after line 31
```cpp
#elif FUNCTION_IDENTIFIER == `<your-identifier>`
#include "programs/<your header file>.hpp"
```

3. Define the function for a given `FUNCTION_IDENTIFIER`:
    - when using the template make sure to replace the `FUNCTION_IDENTIFIER`, the function name and path to the `<schedule-file>`


## Add support for MP-SPDZ instructions not yet implemented

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

Also may also look at [this commit](https://github.com/aSlunk/hpmpc/commit/d7fd4ec47c58fac9344682701e9052bcf52ef95b) where I added `INPUTPERSONAL` (`0xf5`) and `FIXINPUT` (`0xe8`)

## Formatting

You can use/change the clang-format file in [MP-SPDZ/](/MP-SPDZ/.clang-format)

```sh
clang-format --style=file:MP-SPDZ/.clang-format -i MP-SPDZ/lib/**/*.hpp MP-SPDZ/lib/**/*.cpp
```