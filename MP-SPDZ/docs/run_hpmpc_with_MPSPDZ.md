Setup required to run HPMPC with MP-SPDZ. This assumes you have already Installed [MP-SPDZ](https://github.com/data61/MP-SPDZ/releases?page=1) version 0.3.8.

- If not installed yet see [installation guide](/MP-SPDZ/docs/install_mpspdz.md)

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

```sh
mv "$MPSDZ/Programs/Schedules/<file>.sch" "./MP-SPDZ/Schedules/"
mv "$MPSDZ/Programs/Bytecode/*" "./MP-SPDZ/Bytecodes/"
```

## 2. Run computation for 3 Players

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
`500` | `tutorial.mpc`
`501` | `custom.mpc` (can be used for your own functions)
`502-504` | legacy (used to test simple secure share operations)
`505` | `int_test.mpc/int_test_32.mpc` (depending on `BITLENGTH` (`64` or `32`)) can be used to test public integer operations
`506-534` | functions used for benchmarks (mapping can be found in [MP-SPDZ/bench_scripts/measurement.sh](/MP-SPDZ/bench_scripts/measurement.sh))


> Next Section [define private/public Input](/MP-SPDZ/docs/input.md)
