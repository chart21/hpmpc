#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdint>
#include <string>

namespace IR {

inline constexpr unsigned int REG_TYPES = 9;

inline const std::string ROOT_DIR = "MP-SPDZ";
inline const std::string SCHEDULES_PATH = ROOT_DIR + "/Schedules/";
inline const std::string BYTECODE_PATH = ROOT_DIR + "/Bytecodes/";

constexpr size_t SIZE_VEC = DATTYPE / BITLENGTH;

enum class Opcode {
    CISC = 0x0,
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
    GTC = 0x96,
    ADDINT = 0x9b,
    SUBINT = 0x9c,
    MULINT = 0x9d,
    DIVINT = 0x9e,
    STMINT = 0xcb,
    LDMINTI = 0xcc,
    STMINTI = 0xcd,
    MOVINT = 0xd0,

    BIT = 0x51,

    MOVS = 0xc,

    // stats
    USE = 0x17,
    USE_INP = 0x18,

    // GF operations i guess ?
    GLDMC = 0x103,
    GLDMS = 0x104,

    ADDS = 0x21,
    PREFIXSUMS = 0x2d,
    ADDSI = 0x24,
    SUBS = 0x26,

    SHUFFLE = 0xd2,

    // REVEAL SINT
    OPEN = 0xa5,

    // ARITHM
    MULS = 0xa6,
    MULRS = 0xa7,

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
    DIVC = 0x34,
    DIVCI = 0x35,
    FLOORDIVC = 0x3b,
    MODC = 0x36,
    MODCI = 0x37,
    EQC = 0x97,
    EQZC = 0x93,
    LTZC = 0x94,
    ORC = 0x72,
    XORC = 0x71,
    ANDC = 0x70,
    ANDCI = 0x73,
    XORCI = 0x74,
    ORCI = 0x75,
    NOTC = 0x76,
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
    PRINT4 = 0xb5,     // PRINTSTR
    PRINT4COND = 0xbf, // CONDPRINTSTR
    PRINT_COND_PLAIN = 0xe1,
    COND_PRINT_STRB = 0x224,
    PRINT_INT = 0x9f,
    PRINT_FLOAT_PREC = 0xe0,  // set precision for print_float_*
    PRINT_FLOAT_PLAIN = 0xbc, // print float
    PRINTREG = 0xb1,

    TRUNC_PR = 0xa9,

    LDMINT = 0xca,
    LDINT = 0x9a,
    ACTIVE = 0xe9,
    INPUTMIXED = 0xf2,
    INPUTMIXEDREG = 0xf3,
    PUBINPUT = 0xb6,
    INTOUTPUT = 0xe6,
    FLOATOUTPUT = 0xe7,
    FIXINPUT = 0xe8, // TODO

    CONCATS = 0x2f,

    // bit
    LDBITS = 0x20a,
    LDMSB = 0x240,
    STMSB = 0x241,
    LDMSBI = 0x242,
    STMSBI = 0x243,

    XORS = 0x200,
    ANDS = 0x20b,
    ANDRS = 0x202,
    NOTS = 0x20f,

    ANDRSVEC = 0x24a,
    BITDECS = 0x203,

    REVEAL = 0x214,
    TRANSPOSE = 0x20c,
    BITCOMS = 0x204,
    MOVSB = 0x244,

    PRINT_REG_SIGNED = 0x220,
    INPUTBVEC = 0x247,

    MATMULS = 0xaa,
    MATMULSM = 0xab,
    DOTPRODS = 0xa8,
    INCINT = 0xd1,
    USE_MATMUL = 0x1f,

    CONVCBITVEC = 0x231,
    CONVCINTVEC = 0x21f,
    CONVCINT = 0x213,
    CONVSINT = 0x205,
    CONVCBIT2S = 0x249,
    ANDM = 0x20e,
    DABIT = 0x58,
    RANDOMS = 0x5b,
    SPLIT = 0x248, // local share conversion
    NONE = 0xff,
    RAND = 0xb2,

    NOTCB = 0x212,
    XORCB = 0x219,
    ADDCB = 0x21a,
    ADDCBI = 0x21b,
    MULCBI = 0x21c,
    XORCBI = 0x210,
    SHRCBI = 0x21d,
    SHLCBI = 0x21e,

    NPLAYERS = 0xe2,
    THRESHOLD = 0xe3,
    PLAYERID = 0xe4,
    TIME = 0x14,
    START = 0x15,
    STOP = 0x16,

    PUSHINT = 0xce, // considered obsolete
    POPINT = 0xcf,  // considered obsolete
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
