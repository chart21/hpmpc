#pragma once

// === Include the predefined datatypes that you need ===
#include "../../datatypes/Additive_Share.hpp"  // Additive_Share is a datatype that stores secret shares in the arithmetic domain
// ...
// === Include the predefined functions that you need ===
#include "../functions/GEMM.hpp"
// ...

// === Define the main entry point and resulttype of your program ===
#define FUNCTION YourFirstProgram  // define your main entry point
#define RESULTTYPE \
    DATATYPE  // Each main function should define this to make a result accessible after the MPC protocol has finished

// === Start writing your program ===
template <typename Share>  // Share is defined by the MPC protocol specified in config.h
void YourFirstProgram(
    DATATYPE* res)  // The main entry pint needs to define a result pointer, DATATYPE is specified in config.h
{
    using A = Additive_Share<DATATYPE, Share>;
    // ...
    // Your code goes here
}
