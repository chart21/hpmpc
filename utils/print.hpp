#pragma once
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


