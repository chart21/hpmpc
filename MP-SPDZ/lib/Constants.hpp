#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdint>
#include <string>

namespace IR {

inline constexpr unsigned int REG_TYPES = 9;

inline const std::string ROOT_DIR = "MP-SPDZ";
inline const std::string SCHEDULES_PATH = ROOT_DIR + "/Schedules/";
inline const std::string BYTECODE_PATH = ROOT_DIR + "/Bytecodes/";

enum class Opcode {
    NONE,
    // conf / ring size
    REQBL = 0x12,
    CRASH = 0x1b,

    JMP = 0x90,
    JMPNZ = 0x91,
    JMPEQZ = 0x92,
    JMPI = 0x98,
    LDARG = 0x11,
    PICKS = 0x2e,
    // LOAD
    LDI = 0x01,
    LDSI = 0x02,
    LDMC = 0x03,
    LDMS = 0x04,
    STMC = 0x05,
    STMS = 0x06,
    LDMSI = 0x08,
    STMSI = 0xa,

    LTC = 0x95,
    ADDINT = 0x9b,
    SUBINT = 0x9c,
    STMINT = 0xcb,

    BIT = 0x51,

    MOVS = 0xc,

    // stats
    USE = 0x17,
    USE_INP = 0x18,

    // GF operations i guess ?
    GLDMC = 0x103,
    GLDMS = 0x104,

    ADDS = 0x21,
    ADDSI = 0x24,
    SUBS = 0x26,

    // REVEAL SINT
    OPEN = 0xa5,

    // ARITHM
    MULS = 0xa6,

    LDMCB = 0x217,
    STMCB = 0x218,

    MULSI = 0x33,
    SUBSFI = 0x2c,
    SUBSI = 0x2a,
    SUBML = 0x27,
    SUBMR = 0x28,

    MULCI = 0x32,
    ADDCI = 0x23,

    SUBC = 0x25,
    ADDC = 0x20,
    MULC = 0x30,
    EQC = 0x97,
    ORC = 0x72,
    SHLC = 0x80,
    SHRC = 0x81,
    SUBCFI = 0x2b,
    MOVC = 0xb,

    MULM = 0x31,
    ADDM = 0x22,

    SHLCI = 0x82,
    SHRCI = 0x83,

    CONVINT = 0xc0,
    CONVMODP = 0xc1,
    BITDECINT = 0x99,

    // IO
    PRINT_REG_PLAIN = 0xb3,
    PRINT_CHR = 0xb4,
    PRINT4 = 0xb5,
    PRINT4COND = 0xbf,
    PRINT_COND_PLAIN = 0xe1,
    PRINT_INT = 0x9f,
    PRINT_FLOAT_PREC = 0xe0,  // set precision for print_float_*
    PRINT_FLOAT_PLAIN = 0xbc, // print float

    TRUNC_PR = 0xa9,

    LDMINT = 0xca,
    LDINT = 0x9a,
    ACTIVE = 0xe9,
    INPUTMIXED = 0xf2,

    CONCATS = 0x2f,

    // bit
    LDBITS = 0x20a,
    LDMSB = 0x240,
    STMSB = 0x241,

    XORS = 0x200,
    ANDS = 0x20b,
    NOTS = 0x20f,

    ANDRSVEC = 0x24a, // unstable :D
    BITDECS = 0x203,  // unstable :c

    REVEAL = 0x214,
    TRANSPOSE = 0x20c,
    BITCOMS = 0x204,
    MOVSB = 0x244,

    PRINT_REG_SIGNED = 0x220,
    INPUTBVEC = 0x247,

    MATMULSM = 0xab,
    DOTPRODS = 0xa8,
    INCINT = 0xd1,
    USE_MATMUL = 0x1f,

    CONVCBITVEC = 0x231,
    DABIT = 0x58,
    RANDOMS = 0x5b,
    SPLIT = 0x248, // local share conversion
};

enum class Type {
    INT = 0,
    SBIT = 1,
    CBIT = 2,
    DYN_SBIT = 3,
    SINT = 4,
    CINT = 5,
    SGF2N = 6,
    CGF2N = 7,
    NONE = 8,
};
} // namespace IR

#endif
