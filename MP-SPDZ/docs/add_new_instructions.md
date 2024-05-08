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
