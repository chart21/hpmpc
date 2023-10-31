#pragma once
#include <bitset>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include "../config.h"

void print(const char* format, ...) {
    va_list args;
    va_start(args, format);

    printf("P%i: ", PARTY);
    vprintf(format, args);

    va_end(args);
}

template <typename T>
void print_result(T* var) 
{
    printf("P%i: Result: ", PARTY);
    uint8_t v8val[sizeof(T)];
    std::memcpy(v8val, var, sizeof(v8val));
    for (uint i = sizeof(T); i > 0; i--)
        std::cout << std::bitset<sizeof(uint8_t)*8>(v8val[i-1]);
    printf("\n");
}

