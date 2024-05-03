This assumes you know how to compile `.mpc`-files for this project if not check:

- [the setup guide](./run_hpmpc_with_MPSPDZ.md)
- [tutorial.mpc](../Functions/tutorial.mpc) for a more in depth guide 

## Run your own functions

As mentioned in [Setup](./run_hpmpc_with_MPSPDZ.md#setup) copy the bytecode file and schedule file into the correct Directory (`./MP-SPDZ/Schedules/`, `./MP-SPDZ/Bytecodes/` respectively)
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

In [programs/functions/mpspdz.hpp](../../programs/functions/mpspdz.hpp) are all currently supported functions you'll notice the only thing that changes is the path of the `<schedule-file>`

To add a new `FUNCTION_IDENTIFIER`

1. Create a new header file in [programs](../../programs/) you may use [programs/template.hpp](../../programs/template.hpp)
2. Choose a number `<your-num>` (`FUNCTION_IDENTFIER`)
    - make sure it does **NOT** exist yet (see [protocol_executer.hpp](../../protocol_executer.hpp))
    - make sure that in [protocol_executer.hpp](../../protocol_executer.hpp) the correct header file is included

You can do so by adding the following after line 31
```cpp
#elif FUNCTION_IDENTIFIER == `<your-identifier>`
#include "programs/<your header file>.hpp"
```

3. Define the function for a given `FUNCTION_IDENTIFIER`:
    - when using the template make sure to replace the `FUNCTION_IDENTIFIER`, the function name and path to the `<schedule-file>`
