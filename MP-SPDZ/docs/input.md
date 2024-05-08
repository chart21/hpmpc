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

> Next Section: [Run your own .mpc files with HPMPC](/MP-SPDZ/docs/add_new_functions.md)