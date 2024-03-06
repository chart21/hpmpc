#ifndef PROGRAM_H
#define PROGRAM_H

#include <fcntl.h>  // open file
#include <unistd.h> // close file

#include <cassert>     // assert()
#include <cstdint>     // int_t
#include <iomanip>     // set precision
#include <iostream>    // default precision
#include <random>      // random bit
#include <string>      // bytecode path
#include <string_view> // cisc instruction
#include <utility>     // move
#include <vector>      // register

#include "Constants.hpp"
#include "Shares/Integer.hpp"
#include "help/Input.hpp"
#include "help/Util.hpp"
#include "help/matrix.hpp"

using std::string;
using std::vector;

namespace IR {

template <class sint, template <int, class> class sbit, class BitShare, int N>
class Machine;

template <class sint, template <int, class> class sbit, class BitShare, int N = 64>
class Program {
    using BitType = Integer<INT_TYPE, UINT_TYPE>;
    static constexpr size_t BIT_LEN = N;

    class Instruction {
      public:
        explicit Instruction(const uint32_t& op, const int& vec);

        void execute(Program& p, Machine<sint, sbit, BitShare, N>& m, size_t& pc) const;

        const Opcode& get_opcode() const { return op; }

        const size_t& set_immediate(const size_t& im) {
            n = im;
            return n;
        }
        const int& add_reg(const int& reg) { return regs.emplace_back(reg); }

        bool is_gf2n() const { return (static_cast<unsigned>(op) & 0x100) != 0; }
        Type get_reg_type(const Opcode& op) const;

        inline unsigned get_size() const { return size; }

        string cisc; // for cisc command (LTZ, EQZ, ...)
      private:
        Opcode op;        // opcode
        unsigned size;    // vectorized
        size_t n;         // immediate
        vector<int> regs; // required addresses in order given
    };

  public:
    explicit Program(const string&& path, size_t thread);

    Program(const Program& other) = delete;
    Program(Program&& other) = default;
    Program& operator=(const Program& other) = delete;
    Program& operator=(Program&& other) = default;

    bool load_program(Machine<sint, sbit, BitShare, N>& m); // parse bytecode file
    void setup();                                           // parse bytecode file
    void run(Machine<sint, sbit, BitShare, N>& m,
             const int& arg); // execute all instructions

    inline int get_argument() const { return arg; }

    // sint operations
    void popen(const vector<int>& regs, const size_t& size);
    void muls(const vector<int>& regs);
    void mulm(const vector<int>& regs, const size_t& vec);
    void dotprods(const vector<int>& regs, const size_t& size); // TODO
    void inputmixed(const vector<int>& regs);

    void matmulsm(const vector<int>& regs, Machine<sint, sbit, BitShare, N>& m);
    template <class iterator>
    void matmulsm_prepare(const vector<int>& regs, const int& row_1, const int& j, iterator source1,
                          iterator source2);

    void cisc(const vector<int>& regs, const std::string_view cisc);

    // sbit operations
    void inputbvec(const vector<int>& regs);
    void andrsvec(const vector<int>& regs);

  private:
    int precision;
    const string path;

    size_t thread_id;

    int arg;                  // thread arg
    vector<Instruction> prog; // all instructions

    unsigned max_reg[REG_TYPES]; // to get required register size for all types

    vector<sint> s_register;                         // secret share
    vector<Integer<INT_TYPE, UINT_TYPE>> c_register; // clear share
    vector<long> i_register;                         // integer
    vector<sbit<N, BitShare>> sb_register;           // secret bit
    vector<BitType> cb_register;                     // clear bit

    std::mt19937 rand_engine;

    MatrixCalculus101<sint> matrix;

    void update_max_reg(const Type& reg, const unsigned& sreg, const Opcode& op);
};

template <class sint, template <int, class> class sbit, class BitShare, int N>
bool Program<sint, sbit, BitShare, N>::load_program(Machine<sint, sbit, BitShare, N>& m) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        log(Level::WARNING, "couldn't open file: ", path);
        return false;
    }

    unsigned char buf[8]; // 32bit buffer
    int ans;

    while ((ans = read(fd, buf, 8)) > 0) {
        uint64_t num = to_int_n(buf, 8);
        int cur = 0x3ff & num;
        size_t vec = num >> 10; // 1st 22bits for vectorized command

        auto& inst = prog.emplace_back(cur, vec == 0 ? 1 : vec);

        switch (inst.get_opcode()) {
        // sreg + immediate(32)
        case Opcode::LDSI:
        case Opcode::JMPNZ:
        case Opcode::JMPEQZ:
        case Opcode::LDI:
        case Opcode::LDINT:
        case Opcode::RANDOMS:
        case Opcode::PRINT4COND: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            inst.set_immediate(int(read_next_int(fd, buf, 4)));

            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());
            break;
        }
        // sreg + mem_addr(64)
        case Opcode::LDMS:
        case Opcode::LDMC:
        case Opcode::STMC:
        case Opcode::LDMCB:
        case Opcode::STMCB:
        case Opcode::STMS:
        case Opcode::STMINT:
        case Opcode::LDMSB:
        case Opcode::STMSB:
        case Opcode::GLDMC:
        case Opcode::GLDMS:
        case Opcode::LDMINT: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));         // dest
            size_t mem_addr = inst.set_immediate(read_next_int(fd, buf, 8)); // source

            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());
            m.update_max_mem(inst.get_reg_type(inst.get_opcode()), mem_addr + inst.get_size());
            break;
        }
        // sreg + imm(32) + imm(32)
        case Opcode::LDBITS: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            unsigned bits = inst.add_reg(read_next_int(fd, buf, 4));

            inst.set_immediate(read_next_int(fd, buf, 4));

            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + div_ceil(bits, BIT_LEN),
                           inst.get_opcode());
            break;
        }
        case Opcode::XORS:
        case Opcode::ANDS:
        case Opcode::MULS: {
            unsigned args = read_next_int(fd, buf, 4);

            assert(args % 2 == 0);

            for (size_t i = 1; i < args; i += 4) {
                int size = inst.add_reg(read_next_int(fd, buf, 4)); // vector size

                unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4)); // destination

                size = inst.get_opcode() == Opcode::MULS ? size : div_ceil(size, BIT_LEN);
                update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + size,
                               inst.get_opcode());

                sreg = inst.add_reg(read_next_int(fd, buf, 4)); // factor 1
                update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + size,
                               inst.get_opcode());

                sreg = inst.add_reg(read_next_int(fd, buf, 4)); // factor 2
                update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + size,
                               inst.get_opcode());
            }
            break;
        }
        // im(32) + sreg + sreg
        case Opcode::NOTS: {
            size_t bits = inst.set_immediate(read_next_int(fd, buf, 4));

            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, sreg + div_ceil(bits, BIT_LEN), inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, sreg + div_ceil(bits, BIT_LEN), inst.get_opcode());
            break;
        }
        case Opcode::CONVCBIT2S: {
            size_t bits = inst.set_immediate(read_next_int(fd, buf, 4));

            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, sreg + div_ceil(bits, BIT_LEN), inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::CBIT, sreg + div_ceil(bits, BIT_LEN), inst.get_opcode());
            break;
        }
        case Opcode::OPEN: {
            uint32_t num = inst.set_immediate(read_next_int(fd, buf, 4));
            read_next_int(fd, buf, 4); // check after opening (idk)

            for (size_t i = 1; i < num; i += 2) {
                unsigned creg = inst.add_reg(read_next_int(fd, buf, 4));
                unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));

                update_max_reg(Type::CINT, creg + inst.get_size(), inst.get_opcode());
                update_max_reg(Type::SINT, sreg + inst.get_size(), inst.get_opcode());
            }
            break;
        }
        case Opcode::REVEAL: {
            uint32_t num = read_next_int(fd, buf, 4);

            for (size_t i = 0; i < num; i += 3) {
                unsigned num = inst.add_reg(read_next_int(fd, buf, 4));

                unsigned creg = inst.add_reg(read_next_int(fd, buf, 4)); // des
                unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4)); // source

                update_max_reg(Type::CBIT, creg + div_ceil(num, BIT_LEN), inst.get_opcode());
                update_max_reg(Type::SBIT, sreg + div_ceil(num, sizeof(sint) * 8),
                               inst.get_opcode());
            }
            break;
        }
        // immediate(32)
        case Opcode::PRINT4:
        case Opcode::JMP:
        case Opcode::ACTIVE:
        case Opcode::PRINT_CHR:
        case Opcode::PRINT_FLOAT_PREC:
            inst.set_immediate(read_next_int(fd, buf, 4));
            break;
        case Opcode::REQBL: // requirement for modulus prime calculus
                            // min bit length
        {
            int ring = read_next_int(fd, buf, 4);
            if (ring > 0) {
                log(Level::ERROR, "compiled for fields not rings");
            } else if (-ring != BITLENGTH) {
                log(Level::ERROR, ring, ": compiled for rings 2^", BITLENGTH);
                exit(EXIT_FAILURE);
            }
            break;
        }
        case Opcode::PRINT_FLOAT_PLAIN: {
            assert(inst.get_size() == 1);
            int reg = inst.add_reg(read_next_int(fd, buf, 4)); // significant
            update_max_reg(Type::CINT, reg + inst.get_size(), inst.get_opcode());

            reg = inst.add_reg(read_next_int(fd, buf, 4)); // exponent
            update_max_reg(Type::CINT, reg + inst.get_size(), inst.get_opcode());

            reg = inst.add_reg(read_next_int(fd, buf, 4)); // zero bit (zero if == 1)
            update_max_reg(Type::CINT, reg + inst.get_size(), inst.get_opcode());

            reg = inst.add_reg(read_next_int(fd, buf, 4)); // sign bit (neg if == 1)
            update_max_reg(Type::CINT, reg + inst.get_size(), inst.get_opcode());

            reg = inst.add_reg(read_next_int(fd, buf, 4)); // NaN (reg num if zero)
            update_max_reg(Type::CINT, reg + inst.get_size(), inst.get_opcode());
            break;
        }
        // creg
        case Opcode::PRINT_REG_PLAIN:
        case Opcode::PRINT_INT:
        case Opcode::BIT:
        case Opcode::JMPI:
        case Opcode::CRASH:
        case Opcode::LDARG: {
            unsigned reg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(inst.get_reg_type(inst.get_opcode()), reg + inst.get_size(),
                           inst.get_opcode());
            break;
        }
        // im(32) + CBIT
        case Opcode::PRINT_REG_SIGNED: {
            unsigned im = inst.set_immediate(read_next_int(fd, buf, 4));
            unsigned cbit = inst.add_reg(read_next_int(fd, buf, 4));

            update_max_reg(Type::CBIT, cbit + div_ceil(im, BIT_LEN), inst.get_opcode());
            break;
        }
        case Opcode::BITDECINT: {
            unsigned args = read_next_int(fd, buf, 4);
            unsigned source = inst.add_reg(read_next_int(fd, buf, 4)); // source

            update_max_reg(Type::SINT, source + inst.get_size(), inst.get_opcode());

            for (size_t i = 1; i < args; i++) {
                unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));

                update_max_reg(Type::INT, sreg + inst.get_size(), inst.get_opcode());
            }

            break;
        }
        case Opcode::CONCATS: {
            unsigned args = read_next_int(fd, buf, 4);
            unsigned dest = inst.add_reg(read_next_int(fd, buf, 4)); // dest

            update_max_reg(Type::SINT, dest + 1, inst.get_opcode());

            for (size_t i = 1; i < args; i += 2) {
                unsigned off = inst.add_reg(read_next_int(fd, buf, 4)); // offset
                unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));

                dest += off;

                update_max_reg(Type::SINT, sreg + off, inst.get_opcode());
                update_max_reg(Type::SINT, dest, inst.get_opcode());
            }

            break;
        }
        case Opcode::TRANSPOSE: {
            unsigned num = read_next_int(fd, buf, 4);
            unsigned outs = inst.set_immediate(read_next_int(fd, buf, 4));

            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, sreg + div_ceil(num - 1 - outs, BIT_LEN), inst.get_opcode());

            for (size_t i = 2; i < num; ++i) {
                sreg = inst.add_reg(read_next_int(fd, buf, 4));
                update_max_reg(Type::SBIT, sreg + div_ceil(outs, BIT_LEN), inst.get_opcode());
            }
            break;
        }
        case Opcode::PICKS: {
            unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));
            unsigned source = inst.add_reg(read_next_int(fd, buf, 4));
            unsigned off = inst.add_reg(read_next_int(fd, buf, 4));

            int step(inst.set_immediate(read_next_int(fd, buf, 4)));

            update_max_reg(Type::SINT, dest + inst.get_size(), inst.get_opcode());
            update_max_reg(Type::SINT, source + off + step * vec + 1, inst.get_opcode());
            break;
        }
        case Opcode::USE:
        case Opcode::USE_INP: {
            inst.add_reg(read_next_int(fd, buf, 4));
            inst.add_reg(read_next_int(fd, buf, 4));
            inst.set_immediate(read_next_int(fd, buf, 8));
            break;
        }
        case Opcode::INPUTMIXED: {
            unsigned num = read_next_int(fd, buf, 4);
            for (size_t i = 0; i < num; ++i) {
                uint32_t cur = inst.add_reg(read_next_int(fd, buf, 4));
                if (cur == 2) {
                    log(Level::ERROR, "INPUTMIXED: only int/fix is supported");
                }

                uint32_t dest = inst.add_reg(read_next_int(fd, buf, 4));

                if (cur == 1) {                              // fix-point
                    inst.add_reg(read_next_int(fd, buf, 4)); // precision
                    i++;
                }

                update_max_reg(Type::SINT, dest + inst.get_size(), inst.get_opcode());
                inst.add_reg(read_next_int(fd, buf, 4));

                i += 2;
            }
            break;
        }
        case Opcode::INPUTBVEC: {
            unsigned num = read_next_int(fd, buf, 4);

            for (size_t i = 1; i < num; i += 3) {
                unsigned bits = inst.add_reg(read_next_int(fd, buf, 4)) - 3;
                inst.add_reg(read_next_int(fd, buf, 4)); // 2^n
                inst.add_reg(read_next_int(fd, buf, 4)); // player id

                i += bits;

                for (unsigned j = 0; j < bits; ++j) {
                    unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
                    update_max_reg(Type::SBIT, sreg + 1, inst.get_opcode());
                }
            }

            break;
        }
        // sreg + sreg + im(32)
        case Opcode::MULSI:
        case Opcode::SUBSI:
        case Opcode::SHRCI:
        case Opcode::SHLCI:
        case Opcode::ADDCI:
        case Opcode::MODCI:
        case Opcode::DIVCI:
        case Opcode::ADDSI:
        case Opcode::SUBSFI:
        case Opcode::SUBCFI:
        case Opcode::XORCI:
        case Opcode::ORCI:
        case Opcode::ANDCI:
        case Opcode::NOTC:
        case Opcode::ADDCBI:
        case Opcode::MULCBI:
        case Opcode::XORCBI:
        case Opcode::SHRCBI:
        case Opcode::SHLCBI:
        case Opcode::MULCI: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());

            inst.set_immediate(int(read_next_int(fd, buf, 4)));
            break;
        }
        case Opcode::SUBMR: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SINT, sreg + inst.get_size(), inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::CINT, sreg + inst.get_size(), inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SINT, sreg + inst.get_size(), inst.get_opcode());
            break;
        }
        // sreg + sreg
        case Opcode::MOVC:
        case Opcode::EQZC:
        case Opcode::LTZC:
        case Opcode::RAND:
        case Opcode::MOVS: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());
            break;
        }
        case Opcode::CONVINT: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::CINT, sreg + inst.get_size(), inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::INT, sreg + inst.get_size(), inst.get_opcode());
            break;
        }
        case Opcode::STMSI:
        case Opcode::LDMSI: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SINT, sreg + inst.get_size(), inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::INT, sreg + inst.get_size(), inst.get_opcode());
            break;
        }
        // sreg + sreg + sreg
        case Opcode::XORCB:
            inst.set_immediate(read_next_int(fd, buf, 4)); // + BIT_LEN
        case Opcode::ADDCB:
        case Opcode::ADDS:
        case Opcode::SUBC:
        case Opcode::ADDC:
        case Opcode::FLOORDIVC:
        case Opcode::DIVC:
        case Opcode::MODC:
        case Opcode::MULC:
        case Opcode::ORC:
        case Opcode::ANDC:
        case Opcode::XORC:
        case Opcode::SHLC:
        case Opcode::SHRC:
        case Opcode::EQC:
        case Opcode::LTC:
        case Opcode::GTC:
        case Opcode::SUBINT:
        case Opcode::ADDINT:
        case Opcode::PRINT_COND_PLAIN:
        case Opcode::SUBS: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(inst.get_reg_type(inst.get_opcode()), sreg + inst.get_size(),
                           inst.get_opcode());
            break;
        }
        case Opcode::MULM:
        case Opcode::SUBML:
        case Opcode::ADDM: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SINT, sreg + inst.get_size(),
                           inst.get_opcode()); // dest

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SINT, sreg + inst.get_size(),
                           inst.get_opcode()); // sum1

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::CINT, sreg + inst.get_size(),
                           inst.get_opcode()); // sum2
            break;
        }
        case Opcode::ANDM: {
            inst.add_reg(read_next_int(fd, buf, 4)); // bits

            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, sreg + inst.get_size(),
                           inst.get_opcode()); // dest

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, sreg + inst.get_size(),
                           inst.get_opcode()); // sum1

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::CBIT, sreg + inst.get_size(),
                           inst.get_opcode()); // sum2
            break;
        }
        case Opcode::CONVMODP: {
            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::INT, sreg + inst.get_size(),
                           inst.get_opcode()); // dest

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::CINT, sreg + inst.get_size(),
                           inst.get_opcode()); // sum1

            inst.set_immediate(read_next_int(fd, buf, 4));
            break;
        }
        // im(32) + sreg + sreg
        case Opcode::NOTCB:
        case Opcode::MOVSB: {
            unsigned num = inst.set_immediate(read_next_int(fd, buf, 4));

            unsigned sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, sreg + div_ceil(num, BIT_LEN), inst.get_opcode());

            sreg = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, sreg + div_ceil(num, BIT_LEN), inst.get_opcode());
            break;
        }
        case Opcode::BITDECS:
        case Opcode::BITCOMS: {
            unsigned num = read_next_int(fd, buf, 4);
            unsigned bit = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, bit + div_ceil(num, BIT_LEN), inst.get_opcode());

            for (size_t i = 1; i < num; ++i) {
                bit = inst.add_reg(read_next_int(fd, buf, 4));
                update_max_reg(Type::SBIT, bit + 1, inst.get_opcode());
            }
            break;
        }
        case Opcode::CONVCINTVEC: {
            unsigned bits = read_next_int(fd, buf, 4) - 1;
            inst.add_reg(read_next_int(fd, buf, 4)); // source

            for (size_t i = 0; i < bits; ++i) {
                unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));
                update_max_reg(Type::SBIT, dest + div_ceil(inst.get_size(), BIT_LEN),
                               inst.get_opcode());
            }
            break;
        }
        case Opcode::ANDRSVEC: {
            unsigned num = read_next_int(fd, buf, 4);

            for (size_t i = 0; i < num;) {
                unsigned one_op = inst.add_reg(read_next_int(fd, buf, 4));
                unsigned vec = inst.add_reg(read_next_int(fd, buf, 4)); // vector size

                for (size_t j = 2; j < one_op; ++j) {
                    unsigned reg = inst.add_reg(read_next_int(fd, buf, 4));

                    update_max_reg(Type::SBIT, reg + div_ceil(vec, BIT_LEN), inst.get_opcode());
                }

                i += one_op;
            }
            break;
        }
        case Opcode::MATMULSM: {
            unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));

            inst.add_reg(read_next_int(fd, buf, 4)); // factor 1
            inst.add_reg(read_next_int(fd, buf, 4)); // factor 2

            int rows = inst.add_reg(read_next_int(fd, buf, 4));
            inst.add_reg(read_next_int(fd, buf, 4)); // cols/ropws of 1st/2nd factor
            int cols = inst.add_reg(read_next_int(fd, buf, 4));

            update_max_reg(Type::SINT, dest + rows * cols, inst.get_opcode());

            for (size_t i = 0; i < 6u; ++i) {
                inst.add_reg(read_next_int(fd, buf, 4));
            }

            break;
        }
        case Opcode::USE_MATMUL: {
            inst.add_reg(read_next_int(fd, buf, 4));
            inst.add_reg(read_next_int(fd, buf, 4));
            inst.add_reg(read_next_int(fd, buf, 4));
            inst.set_immediate(read_next_int(fd, buf, 8));
            break;
        }
        case Opcode::INCINT: {
            unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));

            inst.add_reg(read_next_int(fd, buf, 4)); // base
            inst.add_reg(read_next_int(fd, buf, 4)); // increment
            inst.add_reg(read_next_int(fd, buf, 4)); // repeat
            inst.add_reg(read_next_int(fd, buf, 4)); // wrap

            update_max_reg(Type::INT, dest + inst.get_size(), inst.get_opcode());
            break;
        }
        case Opcode::TRUNC_PR: {
            unsigned args = read_next_int(fd, buf, 4);

            for (size_t i = 0; i < args; i += 4) {
                unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));
                unsigned source = inst.add_reg(read_next_int(fd, buf, 4));

                inst.add_reg(read_next_int(fd, buf, 4)); // bits to use
                inst.add_reg(read_next_int(fd, buf, 4)); // bits to truncate

                update_max_reg(Type::SINT, dest + inst.get_size(), inst.get_opcode());
            }
            break;
        }
        case Opcode::DOTPRODS: {
            unsigned args = read_next_int(fd, buf, 4);
            for (size_t i = 0; i < args; ++i) {
                unsigned len = inst.add_reg(read_next_int(fd, buf, 4)) - 2;

                unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));
                update_max_reg(Type::SINT, dest + inst.get_size(), inst.get_opcode());

                for (size_t j = 0; j < len; ++j)
                    inst.add_reg(read_next_int(fd, buf, 4));

                i += len + 1;
            }
            break;
        }
        case Opcode::SPLIT: {
            unsigned args = read_next_int(fd, buf, 4);

            inst.add_reg(read_next_int(fd, buf, 4)); // num player
            inst.add_reg(read_next_int(fd, buf, 4)); // source

            for (size_t i = 2; i < args; ++i) {
                unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));
                update_max_reg(Type::SBIT, dest + div_ceil(vec, BIT_LEN), inst.get_opcode());
            }
            break;
        }
        case Opcode::DABIT: {
            unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SINT, dest + inst.get_size(), inst.get_opcode());
            dest = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SBIT, dest + div_ceil(vec, BIT_LEN), inst.get_opcode());
            break;
        }
        case Opcode::CONVCBITVEC: {
            unsigned bits = inst.set_immediate(read_next_int(fd, buf, 4));

            unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));
            update_max_reg(Type::SINT, dest + bits, inst.get_opcode());
            inst.add_reg(read_next_int(fd, buf, 4)); // source
            break;
        }
        case Opcode::CISC: {
            unsigned args = read_next_int(fd, buf, 4);
            print("reading CISC: %u\n", args);
            char op[16];
            read(fd, op, 16);
            inst.cisc = string(op, 16);
            std::cout << "GOT: " << inst.cisc << "\n";

            if (strncmp(op, "LTZ", 3) == 0 || strncmp(op, "EQZ", 3) == 0) {
                for (size_t i = 0; i < args - 1; i += 6) {
                    unsigned size = inst.add_reg(read_next_int(fd, buf, 4)); // arguments
                    unsigned vec = inst.add_reg(read_next_int(fd, buf, 4));
                    assert(size == 6);
                    unsigned dest = inst.add_reg(read_next_int(fd, buf, 4));
                    inst.add_reg(read_next_int(fd, buf, 4)); // result
                    inst.add_reg(read_next_int(fd, buf, 4)); // bit_length
                    inst.add_reg(read_next_int(fd, buf, 4)); // ignore
                    update_max_reg(Type::SINT, dest + vec, inst.get_opcode());
                }
            } else {
                for (size_t i = 0; i < args - 1; ++i) {
                    unsigned cur = inst.add_reg(read_next_int(fd, buf, 4));
                    std::cout << cur << "\n";
                }
            }
            break;
        }
        default:
            log(Level::WARNING, "unknown operation");
            log(Level::WARNING, "read: ", cur);
            log(Level::WARNING, "vec: ", vec);
            close(fd);
            return false;
        }
    }

    close(fd);
    return true;
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
Type Program<sint, sbit, BitShare, N>::Instruction::get_reg_type(const Opcode& op) const {
    switch (op) {
    case Opcode::LDBITS:
    case Opcode::BITCOMS:
    case Opcode::BITDECS:
    case Opcode::XORS:
    case Opcode::ANDS:
    case Opcode::NOTS:
    case Opcode::LDMSB:
    case Opcode::STMSB:
    case Opcode::MOVSB:
    case Opcode::TRANSPOSE:
    case Opcode::INPUTBVEC:
        return Type::SBIT;
    case Opcode::PRINT_REG_SIGNED:
    case Opcode::LDMCB:
    case Opcode::STMCB:
    case Opcode::NOTCB:
    case Opcode::XORCB:
    case Opcode::ADDCB:
    case Opcode::ADDCBI:
    case Opcode::MULCBI:
    case Opcode::SHRCBI:
    case Opcode::SHLCBI:
    case Opcode::XORCBI:
        return Type::CBIT;
    case Opcode::LDMINT:
    case Opcode::LDARG:
    case Opcode::JMPI:
    case Opcode::CRASH:
    case Opcode::BITDECINT:
    case Opcode::LDINT:
    case Opcode::STMINT:
    case Opcode::LTC:
    case Opcode::GTC:
    case Opcode::SUBINT:
    case Opcode::ADDINT:
    case Opcode::JMPEQZ:
    case Opcode::JMPNZ:
    case Opcode::INCINT:
    case Opcode::EQC:
    case Opcode::EQZC:
    case Opcode::LTZC:
    case Opcode::RAND:
    case Opcode::PRINT_INT:
        return Type::INT;
    case Opcode::LDI:
    case Opcode::LDMC:
    case Opcode::STMC:
    case Opcode::SHRCI:
    case Opcode::SHLCI:
    case Opcode::MULCI:
    case Opcode::ADDCI:
    case Opcode::DIVCI:
    case Opcode::MODCI:
    case Opcode::PRINT_REG_PLAIN:
    case Opcode::PRINT_COND_PLAIN:
    case Opcode::PRINT4COND:
    case Opcode::PRINT_FLOAT_PLAIN:
    case Opcode::ADDC:
    case Opcode::SUBC:
    case Opcode::MULC:
    case Opcode::DIVC:
    case Opcode::MODC:
    case Opcode::FLOORDIVC:
    case Opcode::ORC:
    case Opcode::ANDC:
    case Opcode::XORC:
    case Opcode::NOTC:
    case Opcode::SHLC:
    case Opcode::SHRC:
    case Opcode::SUBCFI:
    case Opcode::MOVC:
    case Opcode::XORCI:
    case Opcode::ORCI:
    case Opcode::ANDCI:
        return Type::CINT;
    case Opcode::LDMS:
    case Opcode::INPUTMIXED:
    case Opcode::LDSI:
    case Opcode::STMS:
    case Opcode::MULS:
    case Opcode::MULSI:
    case Opcode::SUBSI:
    case Opcode::ADDS:
    case Opcode::SUBS:
    case Opcode::CONCATS:
    case Opcode::MOVS:
    case Opcode::BIT:
    case Opcode::SUBSFI:
    case Opcode::DOTPRODS:
    case Opcode::ADDSI:
    case Opcode::RANDOMS:
        return Type::SINT;
    default:
        break;
    }

    if (is_gf2n()) {
        unsigned code = static_cast<unsigned>(op);
        code -= 0x100;
        if (get_reg_type(static_cast<Opcode>(code)) == Type::CINT)
            return Type::CGF2N;
        else
            return Type::SGF2N;
    }

    return Type::SINT;
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
Program<sint, sbit, BitShare, N>::Instruction::Instruction(const uint32_t& opc, const int& vec)
    : size(vec) {
    // only if opc is known
    if (opc == 0x01 || opc == 0x02 || opc == 0x03 || opc == 0x12 || opc == 0x1b || opc == 0x90 ||
        opc == 0x21f || opc == 0xb || opc == 0x91 || opc == 0x92 || opc == 0x98 || opc == 0x11 ||
        opc == 0x2e || opc == 0x249 || opc == 0x04 || opc == 0x05 || opc == 0x06 || opc == 0xcb ||
        opc == 0x08 || opc == 0xa || opc == 0x51 || opc == 0x58 || opc == 0xc || opc == 0xc0 ||
        opc == 0xb2 || opc == 0xc1 || opc == 0x99 || opc == 0x17 || opc == 0x18 || opc == 0x21 ||
        opc == 0x24 || opc == 0x26 || opc == 0x103 || opc == 0x104 || opc == 0xa5 || opc == 0xa6 ||
        opc == 0x23 || opc == 0x31 || opc == 0x22 || opc == 0x32 || opc == 0x20 || opc == 0x33 ||
        opc == 0x231 || opc == 0x2a || opc == 0x2c || opc == 0x27 || opc == 0x28 || opc == 0x25 ||
        opc == 0x82 || opc == 0x35 || opc == 0x83 || opc == 0xb3 || opc == 0xb4 || opc == 0xb5 ||
        opc == 0xbf || opc == 0x30 || opc == 0x34 || opc == 0xe1 || opc == 0xca || opc == 0x9a ||
        opc == 0xf2 || opc == 0x2f || opc == 0x20a || opc == 0x37 || opc == 0x97 || opc == 0x217 ||
        opc == 0x218 || opc == 0x203 || opc == 0x204 || opc == 0x3b || opc == 0x00 || opc == 0x73 ||
        opc == 0x74 || opc == 0x75 || opc == 0x76 || opc == 0x20e || opc == 0x244 || opc == 0x72 ||
        opc == 0x9f || opc == 0x80 || opc == 0x36 || opc == 0x2b || opc == 0x214 || opc == 0x24a ||
        opc == 0x5b || opc == 0x248 || opc == 0x1f || opc == 0x71 || opc == 0xe0 || opc == 0x81 ||
        opc == 0x21e || opc == 0x240 || opc == 0x241 || opc == 0x200 || opc == 0x212 ||
        opc == 0x219 || opc == 0x21d || opc == 0x21a || opc == 0xa9 || opc == 0x70 ||
        opc == 0x247 || opc == 0xab || opc == 0x20c || opc == 0xbc || opc == 0x21b ||
        opc == 0x210 || opc == 0x20b || opc == 0xa8 || opc == 0x94 || opc == 0x20f ||
        opc == 0x220 || opc == 0xd1 || opc == 0x21c || opc == 0xe9 || opc == 0x95 || opc == 0x9c ||
        opc == 0x93 || opc == 0x9b) {
        op = static_cast<Opcode>(opc);
    } else {
        op = Opcode::NONE;
        n = opc;
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
Program<sint, sbit, BitShare, N>::Program(const string&& path, size_t thread)
    : precision(6), path(std::move(path)), thread_id(thread), max_reg(), rand_engine(21), matrix() {
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::update_max_reg(const Type& reg, const unsigned& sreg,
                                                      const Opcode& op [[maybe_unused]]) {
    if (max_reg[static_cast<unsigned>(reg)] < sreg)
        max_reg[static_cast<unsigned>(reg)] = sreg;
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::setup() {
    // prepare register
    s_register.resize(max_reg[static_cast<unsigned>(Type::SINT)]);
    c_register.resize(max_reg[static_cast<unsigned>(Type::CINT)]);
    i_register.resize(max_reg[static_cast<unsigned>(Type::INT)]);
    cb_register.resize(max_reg[static_cast<unsigned>(Type::CBIT)]);
    sb_register.resize(max_reg[static_cast<unsigned>(Type::SBIT)]);
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::run(Machine<sint, sbit, BitShare, N>& m, const int& argi) {
    arg = argi;

    for (size_t pc = 0; pc < prog.size(); ++pc)
        prog[pc].execute(*this, m, pc);
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::Instruction::execute(Program<sint, sbit, BitShare, N>& p,
                                                            Machine<sint, sbit, BitShare, N>& m,
                                                            size_t& pc) const {
    for (size_t vec = 0; vec < size; ++vec) {
        switch (op) {
        case Opcode::LDARG:
            p.i_register[regs[0]] = p.get_argument();
            break;
        case Opcode::LDMC:
            p.c_register[regs[0] + vec] = m.c_mem[n + vec];
            break;
        case Opcode::STMC:
            m.c_mem[n + vec] = p.c_register[regs[0] + vec];
            break;
        case Opcode::LDMCB:
            p.cb_register[regs[0] + vec] = m.cb_mem[long(n) + vec];
            break;
        case Opcode::STMCB:
            m.cb_mem[long(n) + vec] = p.cb_register[regs[0] + vec];
            break;
        case Opcode::LDMS:
            p.s_register[regs[0] + vec] = m.s_mem[n + vec];
            break;
        case Opcode::LDMSI:
            p.s_register[regs[0] + vec] = m.s_mem[p.i_register[regs[1]] + vec];
            break;
        case Opcode::STMSI:
            m.s_mem[vec + p.i_register[regs[1]]] = p.s_register[regs[0] + vec];
            break;
        case Opcode::LDSI:
            p.s_register[regs[0] + vec] = int(n);
            break;
        case Opcode::LDI:
            p.c_register[regs[0] + vec] = int(n);
            break;
        case Opcode::LDINT:
            p.i_register[regs[0] + vec] = int(n);
            break;
        case Opcode::INCINT: {
            auto dest = p.i_register.begin() + regs[0];
            auto base = p.i_register[regs[1]];
            long cur = base;

            const int& inc = regs[2];
            int repeat = regs[3];
            int wrap = regs[4];

            for (size_t i = 0; i < size; ++i) {
                *(dest + i) = cur;

                repeat--;
                if (repeat == 0) {
                    repeat = regs[3];

                    wrap--;
                    if (wrap == 0) {
                        cur = base;
                        wrap = regs[4];
                    } else {
                        cur += inc;
                    }
                }
            }
            return;
        }
        case Opcode::MATMULSM:
            p.matmulsm(regs, m);
            return;
        case Opcode::DOTPRODS:
            p.dotprods(regs, get_size());
            return;
        case Opcode::USE: // for statistics
        case Opcode::USE_INP:
        case Opcode::USE_MATMUL:
            // std::cerr << "DEBUG: USE_* " << regs[0] + vec << "," <<
            // regs[1] + vec << "," << n << "\n";
            break;
        case Opcode::ADDC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] + p.c_register[regs[2] + vec];
            break;
        case Opcode::SUBC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] - p.c_register[regs[2] + vec];
            break;
        case Opcode::MULC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] * p.c_register[regs[2] + vec];
            break;
        case Opcode::ORC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] | p.c_register[regs[2] + vec];
            break;
        case Opcode::ANDC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] & p.c_register[regs[2] + vec];
            break;
        case Opcode::XORC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] ^ p.c_register[regs[2] + vec];
            break;
        case Opcode::ORCI:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] | int(n);
            break;
        case Opcode::XORCI:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] ^ int(n);
            break;
        case Opcode::NOTC:
            p.c_register[regs[0] + vec] = (~p.c_register[regs[1] + vec]) & ((1lu << int(n)) - 1lu);
            break;
        case Opcode::ANDCI:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] & int(n);
            break;
        case Opcode::DIVC:
        case Opcode::FLOORDIVC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] / p.c_register[regs[2] + vec];
            break;
        case Opcode::MODC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] % p.c_register[regs[2] + vec];
            break;
        case Opcode::SHLC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec]
                                          << p.c_register[regs[2] + vec];
            break;
        case Opcode::SHRC:
            p.c_register[regs[0] + vec] =
                p.c_register[regs[1] + vec] >> p.c_register[regs[2] + vec];
            break;
        case Opcode::ADDS:
            p.s_register[regs[0] + vec] = p.s_register[regs[1] + vec] + p.s_register[regs[2] + vec];
            break;
        case Opcode::SUBS:
            p.s_register[regs[0] + vec] = p.s_register[regs[1] + vec] - p.s_register[regs[2] + vec];
            break;
        case Opcode::MULS:
            p.muls(regs);
            return;
        case Opcode::MULSI:
            p.s_register[regs[0] + vec] = p.s_register[regs[1] + vec].mult_public(int(n));
            break;
        case Opcode::SUBSI:
            p.s_register[regs[0] + vec] = p.s_register[regs[1] + vec] - sint(int(n));
            break;
        case Opcode::MULCI:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] * int(n);
            break;
        case Opcode::EQC:
            p.i_register[regs[0] + vec] =
                p.i_register[regs[1] + vec] == p.i_register[regs[2] + vec];
            break;
        case Opcode::EQZC:
            p.i_register[regs[0] + vec] = p.i_register[regs[1] + vec] == 0;
            break;
        case Opcode::LTZC:
            p.i_register[regs[0] + vec] = p.i_register[regs[1] + vec] < 0;
            break;
        case Opcode::RAND: {
            long rand = p.rand_engine();
            rand = p.i_register[regs[1] + vec] >= 64 ? rand
                                                     : rand % (1 << p.i_register[regs[1] + vec]);
            p.i_register[regs[0] + vec] = rand;
            break;
        }
        case Opcode::ADDCI:
            p.c_register[regs[0] + vec] = (p.c_register[regs[1] + vec]) + int(n);
            break;
        case Opcode::DIVCI:
            p.c_register[regs[0] + vec] = (p.c_register[regs[1] + vec]) / int(n);
            break;
        case Opcode::MODCI:
            p.c_register[regs[0] + vec] = (p.c_register[regs[1] + vec]) % int(n);
            break;
        case Opcode::ADDSI:
            p.s_register[regs[0] + vec] = p.s_register[regs[1] + vec] + sint(int(n));
            break;
        case Opcode::SUBSFI:
            p.s_register[regs[0] + vec] = sint{UINT_TYPE(n)} - (p.s_register[regs[1] + vec]);
            break;
        case Opcode::SUBCFI:
            p.c_register[regs[0] + vec] =
                Integer<INT_TYPE, UINT_TYPE>(n) - (p.c_register[regs[1] + vec]);
            break;
        case Opcode::SHRCI:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] >> int(n);
            break;
        case Opcode::SHLCI:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec] << n;
            break;
        case Opcode::LTC:
            p.i_register[regs[0] + vec] =
                p.i_register[regs[1] + vec] < p.i_register[regs[2] + vec] ? 1 : 0;
            break;
        case Opcode::GTC:
            p.i_register[regs[0] + vec] =
                p.i_register[regs[1] + vec] > p.i_register[regs[2] + vec] ? 1 : 0;
            break;
        case Opcode::SUBINT:
            p.i_register[regs[0] + vec] = p.i_register[regs[1] + vec] - p.i_register[regs[2] + vec];
            break;
        case Opcode::ADDINT:
            p.i_register[regs[0] + vec] = p.i_register[regs[1] + vec] + p.i_register[regs[2] + vec];
            break;
        case Opcode::MULM:
            p.mulm(regs, get_size());
            return;
        case Opcode::ANDM:
            for (int i = 0; i < regs[0]; ++i) {
                p.sb_register[regs[1] + i / BIT_LEN][i % BIT_LEN] =
                    p.sb_register[regs[2] + i / BIT_LEN][i % BIT_LEN].prepare_and(BitShare(
                        ((p.cb_register[regs[3] + i / BIT_LEN] >> (i % BIT_LEN)) & 1).get()));
            }
            BitShare::communicate();
            for (int i = 0; i < regs[0]; ++i)
                p.sb_register[regs[1] + i / BIT_LEN][i % BIT_LEN].complete_and();
            return;
        case Opcode::SUBML:
            p.s_register[regs[0] + vec] =
                p.s_register[regs[1] + vec] - sint{UINT_TYPE(p.c_register[regs[2] + vec].get())};
            break;
        case Opcode::SUBMR:
            p.s_register[regs[0] + vec] =
                sint{UINT_TYPE(p.c_register[regs[1] + vec].get())} - p.s_register[regs[2] + vec];
            break;
        case Opcode::ADDM:
            p.s_register[regs[0] + vec] =
                p.s_register[regs[1] + vec] + sint{UINT_TYPE(p.c_register[regs[2] + vec].get())};
            break;
        case Opcode::OPEN:
            p.popen(regs, get_size());
            return;
        case Opcode::TRUNC_PR:
            print("TODO: TRUNC_PR\n");
            assert(regs[3] == FRACTIONAL);
            break;
        case Opcode::STMS:
            m.s_mem[n + vec] = p.s_register[regs[0] + vec];
            break;
        case Opcode::PRINT4:
            m.get_out() << string((char*)&n, 4);
            break;
        case Opcode::PRINT4COND:
            if (p.c_register[regs[0]] != 0)
                m.get_out() << string((char*)&n, 4);
            break;
        case Opcode::PRINT_CHR: {
            m.get_out() << static_cast<char>(n);
            break;
        }
        case Opcode::PRINT_INT:
            m.get_out() << p.i_register[regs[0]];
            return;
        case Opcode::PRINT_REG_PLAIN:
            if (size > 1)
                m.get_out() << "[";

            m.get_out() << p.c_register[regs[0]].get();
            for (size_t i = 1; i < size; ++i) {
                m.get_out() << ", " << p.c_register[regs[0] + i].get();
            }
            if (size > 1)
                m.get_out() << "]";
            return;
        case Opcode::PRINT_COND_PLAIN:
            if (p.c_register[regs[0]] != 0) {
                m.get_out()
                    << (p.c_register[regs[1]] * std::pow(2, p.c_register[regs[2]].get())).get();
            }
            break;
        case Opcode::INPUTMIXED: // fine
            p.inputmixed(regs);
            return;
        case Opcode::CONCATS: {
            auto dest = p.s_register.begin() + regs[0];

            for (size_t i = 1; i < regs.size(); i += 2) {
                auto source = p.s_register.begin() + regs[i + 1];

                for (int j = 0; j < regs[i]; ++j)
                    *dest++ = *source++;
            }
            return;
        }
        case Opcode::LDBITS: // fine
            if (regs[1] >= int(BIT_LEN))
                log(Level::ERROR, "public val too long :c");

            for (int j = 0; j < regs[1]; j++) {
                p.sb_register[regs[0]][j] = BitShare(((n >> j) & 1) == 1 ? 1 : ZERO);
            }

            return;
        case Opcode::STMSB:
            m.sb_mem[n] = p.sb_register[regs[0]];
            break;
        case Opcode::LDMSB:
            p.sb_register[regs[0]] = m.sb_mem[n];
            break;
        case Opcode::XORS:
            for (size_t i = 0; i < regs.size(); i += 4) {
                for (int j = 0; j < regs[i]; ++j) {
                    p.sb_register[regs[i + 1] + j / BIT_LEN][j % BIT_LEN] =
                        p.sb_register[regs[i + 2] + j / BIT_LEN][j % BIT_LEN] ^
                        p.sb_register[regs[i + 3] + j / BIT_LEN][j % BIT_LEN];
                }
            }
            return;
        case Opcode::ANDS:
            for (size_t i = 0; i < regs.size(); i += 4) {
                for (int j = 0; j < regs[i]; ++j) {
                    p.sb_register[regs[i + 1] + j / BIT_LEN][j % BIT_LEN] =
                        p.sb_register[regs[i + 2] + j / BIT_LEN][j % BIT_LEN] &
                        p.sb_register[regs[i + 3] + j / BIT_LEN][j % BIT_LEN];
                }
            }

            BitShare::communicate();

            for (size_t i = 0; i < regs.size(); i += 4) {
                for (int j = 0; j < regs[i]; ++j) {
                    p.sb_register[regs[i + 1] + j / BIT_LEN][j % BIT_LEN].complete_and();
                }
            }
            return;
        case Opcode::NOTS:
            for (int i = 0; i < int(n); ++i)
                p.sb_register[regs[0] + i / BIT_LEN][i % BIT_LEN] =
                    ~(p.sb_register[regs[1] + i / BIT_LEN][i % BIT_LEN]);
            return;
        case Opcode::NOTCB: {
            long cur = n;
            for (size_t i = 0; i < div_ceil(n, BIT_LEN); ++i) {
                long bits = std::min(cur, long(BIT_LEN));
                auto num = ~p.cb_register[regs[1] + i];
                p.cb_register[regs[0] + i] = bits == BIT_LEN ? num : num & ((1lu << bits) - 1lu);
                cur -= BIT_LEN;
            }
            return;
        }
        case Opcode::XORCB: {
            long cur = n;
            for (size_t i = 0; i < div_ceil(n, BIT_LEN); ++i) {
                long bits = std::min(cur, long(BIT_LEN));
                auto num = p.cb_register[regs[1] + i] ^ p.cb_register[regs[2] + i];
                p.cb_register[regs[0] + i] = bits == BIT_LEN ? num : num & ((1lu << bits) - 1lu);
                cur -= BIT_LEN;
            }
            return;
        }
        case Opcode::ADDCB:
            p.cb_register[regs[0] + vec] =
                p.cb_register[regs[1] + vec] + p.cb_register[regs[2] + vec];
            return;
        case Opcode::ADDCBI:
            p.cb_register[regs[0] + vec] = p.cb_register[regs[1] + vec] + int(n);
            break;
        case Opcode::MULCBI:
            p.cb_register[regs[0] + vec] = p.cb_register[regs[1] + vec] * int(n);
            break;
        case Opcode::XORCBI:
            p.cb_register[regs[0] + vec] = p.cb_register[regs[1] + vec] ^ int(n);
            break;
        case Opcode::SHRCBI:
            p.cb_register[regs[0] + vec] = p.cb_register[regs[1] + vec] >> int(n);
            break;
        case Opcode::SHLCBI:
            p.cb_register[regs[0] + vec] = p.cb_register[regs[1] + vec] << int(n);
            break;
        case Opcode::REVEAL:
            for (size_t i = 0; i < regs.size(); i += 3) {
                for (int j = 0; j < regs[i]; ++j) {
                    p.sb_register[regs[i + 2] + j / BIT_LEN][j % BIT_LEN].prepare_reveal_to_all();
                }
            }

            BitShare::communicate();

            for (size_t i = 0; i < regs.size(); i += 3) {
                for (int j = 0; j < div_ceil(regs[i], BIT_LEN); ++j)
                    p.cb_register[regs[i + 1] + j] = 0;
                for (int j = 0; j < regs[i]; ++j) {
                    p.cb_register[regs[i + 1] + j / BIT_LEN] |=
                        (p.sb_register[regs[i + 2] + j / BIT_LEN][j % BIT_LEN]
                             .complete_reveal_to_all() &
                         1)
                        << (j % BIT_LEN);
                }
            }
            break;
        case Opcode::PRINT_REG_SIGNED: {
            long cur = 0;
            assert(n <= BIT_LEN);

            for (size_t j = 0; j < div_ceil(n, BIT_LEN); ++j) {
                cur |= long(mask(p.cb_register[j + regs[0]], n)) << (BIT_LEN * j);
            }

            long res;
            if (n < BIT_LEN && n > 1 && cur & (long(1l) << (n - 1))) {
                res = static_cast<long>(cur | (~((long(1l) << n) - 1l)));
            } else {
                res = static_cast<long>(cur);
            }

            m.get_out() << static_cast<long long>(res);
            break;
        }
        case Opcode::PRINT_FLOAT_PREC:
            p.precision = int(n);
            return;
        case Opcode::PRINT_FLOAT_PLAIN: {
            // m.get_out() << "significant: " << p.c_register[regs[0]].get() << "\n";
            // m.get_out() << "exponent: " << p.c_register[regs[1]].get() << "\n";
            // m.get_out() << "zero bit: " << p.c_register[regs[2]].get() << "\n";
            // m.get_out() << "sign bit: " << p.c_register[regs[3]].get() << "\n";
            // m.get_out() << "NaN bit: " << p.c_register[regs[4]].get() << "\n";

            if (p.c_register[regs[4]].get()) {
                m.get_out() << "NaN";
                return;
            } else if (p.c_register[regs[2]].get()) {
                m.get_out() << "0";
                return;
            }

            double res = p.c_register[regs[0]].get() * powf(2, p.c_register[regs[1]].get()) *
                         (p.c_register[regs[3]].get() == 1 ? -1.f : 1.f);
            m.get_out() << std::setprecision(p.precision) << res;
            return;
        }
        case Opcode::BITCOMS: {
            size_t dest = regs[0];
            for (size_t i = 0; i < BIT_LEN; i++) {
                p.sb_register[dest][i] = 0;
            }

            if (BIT_LEN <= regs.size() - 2) {
                log(Level::WARNING, "Note: bits should be compiled with BITLENGTH = 64");
            }
            for (size_t i = 1; i < regs.size(); i++) {
                p.sb_register[dest][i - 1] = p.sb_register[regs[i]][0];
            }
            break;
        }
        case Opcode::BITDECS: {
            size_t source = regs[0];

            if (BIT_LEN <= regs.size() - 2) {
                log(Level::WARNING, "Note: bits should be compiled with BITLENGTH = 64");
            }
            for (size_t i = 1; i < regs.size(); i++) {
                p.sb_register[regs[i]] = p.sb_register[regs[i]] ^ p.sb_register[regs[i]];
                p.sb_register[regs[i]][0] = p.sb_register[source][i - 1];
            }
            break;
        }
        case Opcode::TRANSPOSE: {
            size_t outs = n;
            size_t bits = regs.size() - outs;
            for (size_t i = 0; i < outs; ++i) {
                for (size_t j = 0; j < bits; ++j)
                    p.sb_register[regs[i] + j / BIT_LEN][j % BIT_LEN] =
                        p.sb_register[regs[outs + j] + i / BIT_LEN][i % BIT_LEN];
            }
            return;
        }
        case Opcode::PICKS:
            p.s_register[regs[0] + vec] = p.s_register[regs[1] + regs[2] + int(n) * vec];
            break;
        case Opcode::MOVS:
            p.s_register[regs[0] + vec] = p.s_register[regs[1] + vec];
            break;
        case Opcode::MOVC:
            p.c_register[regs[0] + vec] = p.c_register[regs[1] + vec];
            break;
        case Opcode::CONVINT:
            p.c_register[regs[0] + vec] = p.i_register[regs[1] + vec];
            break;
        case Opcode::MOVSB:
            for (size_t i = 0; i < div_ceil(n, BIT_LEN); ++i)
                p.sb_register[i + regs[0]] = p.sb_register[i + regs[1]];
            break;
        case Opcode::INPUTBVEC:
            p.inputbvec(regs);
            return;
        case Opcode::JMP:
            pc += (signed int)n;
            break;
        case Opcode::JMPI:
            pc += (signed int)p.i_register[regs[0] + vec];
            break;
        case Opcode::JMPNZ:
            if (p.i_register[regs[0]] != 0)
                pc += (signed int)n;
            return;
        case Opcode::JMPEQZ:
            if (p.i_register[regs[0]] == 0)
                pc += (signed int)n;
            return;
        case Opcode::BIT:
            p.s_register[regs[0] + vec] = p.rand_engine() & 1; // welp for testing/simulation
            break;
        case Opcode::DABIT: { // TODO
            unsigned bit = p.rand_engine() & 1;
            p.s_register[regs[0] + vec] = bit;
            p.sb_register[regs[1] + vec / BIT_LEN][vec % BIT_LEN] = BitShare(bit);
            break;
        }
        case Opcode::CONVCBITVEC:
            for (size_t i = 0; i < n; ++i) {
                p.i_register[regs[0] + i] =
                    ((p.cb_register[regs[1] + i / BIT_LEN] >> (i % BIT_LEN)) & 1).get();
            }
            break;
        case Opcode::CONVCINTVEC: {
            for (size_t i = 0; i < get_size(); ++i) {
                auto source = p.c_register[regs[0] + i];
                for (size_t j = 1; j < regs.size(); ++j) {
                    if (i % BIT_LEN == 0)
                        p.cb_register[regs[j] + i / BIT_LEN] = 0;
                    p.cb_register[regs[j] + i / BIT_LEN] ^= ((source >> (j - 1)) & 1)
                                                            << (i % BIT_LEN);
                }
            }
            return;
        }
        case Opcode::CONVCBIT2S:
            for (int i = 0; i < int(n); ++i) {
                p.sb_register[regs[0] + i / BIT_LEN][i % BIT_LEN] =
                    ((p.cb_register[regs[1] + i / BIT_LEN] >> (i % BIT_LEN)) & 1).get();
            }
            return;
        case Opcode::CONVMODP:
            if (n == 0) { // unsigned conversion
                p.i_register[regs[0] + vec] = (unsigned long)p.c_register[regs[1] + vec].get();
            } else if (n <= BIT_LEN) {
                auto dest = p.i_register.begin() + regs[0] + vec;
                auto x = p.c_register[regs[1] + vec];
                if (n == 1) {
                    *dest = (x & 1).get();
                } else if (n == BIT_LEN) {
                    *dest = x.get();
                } else {
                    Integer a = x.abs();
                    a &= ~(uint64_t(-1) << (n - 1) << 1);
                    if (x < 0)
                        a = -a;

                    *dest = a.get();
                }
            } else {
                log(Level::WARNING, "CONVMODP with bit size > 64 is not possible");
            }
            break;
        case Opcode::BITDECINT: {
            long x = p.i_register[regs[0] + vec];

            for (size_t i = 1; i < regs.size(); ++i) {
                p.i_register[regs[i] + vec] = (x >> (i - 1)) & 1;
            }
            break;
        }
        case Opcode::ANDRSVEC:
            p.andrsvec(regs);
            return;
        case Opcode::LDMINT:
            p.i_register[regs[0] + vec] = m.ci_mem[n + vec];
            break;
        case Opcode::STMINT:
            m.ci_mem[n + vec] = p.i_register[regs[0] + vec];
            break;
        case Opcode::CRASH:
            if (p.i_register[regs[0]] != 0)
                log(Level::ERROR, "CRASH");
            break;
        case Opcode::ACTIVE:
        case Opcode::REQBL:
            // std::cerr << "DEBUG: ACTIVE " << n << "\n";
            break;
        case Opcode::GLDMC:
            // std::cerr << "DEBUG: GLDMC\n";
            break;
        case Opcode::GLDMS:
            // std::cerr << "DEBUG: GLDMS\n";
            break;
        case Opcode::CISC: {
            p.cisc(regs, cisc);
            return;
        }
        case Opcode::NONE:
            log(Level::WARNING, "unknown opcode: ", n);
            break;
        default:
            log(Level::WARNING, "not implemented: ", static_cast<unsigned>(op));
        }
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::inputbvec(const vector<int>& regs) {
    for (size_t i = 0; i < regs.size(); i += 3) {
        unsigned bits = regs[i] - 3;
        assert(bits == BITLENGTH && "BITLENGTH must equal -B <int>");

        for (size_t j = 0; j < bits; ++j) {
            auto input = get_next_bit(regs[i + 2]);
            if (j >= BIT_LEN)
                log(Level::ERROR, "input is too big");
            switch (regs[i + 2]) {
            case 0:
                sb_register[regs[i + 3 + j]][0].template prepare_receive_from<P_0>();
                break;
            case 1:
                sb_register[regs[i + 3 + j]][0].template prepare_receive_from<P_1>();
                break;
            case 2:
                sb_register[regs[i + 3 + j]][0].template prepare_receive_from<P_2>();
                break;
            }
        }
        i += bits;
    }

    BitShare::communicate();

    for (size_t i = 0; i < regs.size(); i += 3) {
        unsigned bits = regs[i] - 3;
        for (size_t j = 0; j < bits; ++j) {
            switch (regs[i + 2]) {
            case 0:
                sb_register[regs[i + 3 + j]][0].template complete_receive_from<P_0>();
                break;
            case 1:
                sb_register[regs[i + 3 + j]][0].template complete_receive_from<P_1>();
                break;
            case 2:
                sb_register[regs[i + 3 + j]][0].template complete_receive_from<P_2>();
                break;
            }
        }
        i += bits;
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::inputmixed(const vector<int>& regs) {
    for (size_t i = 0; i < regs.size(); i += 3) {
        INT_TYPE input = 0;
        int dest = regs[i + 1];

        switch (regs[i]) {
        case 0: // int
            input = next_input(regs[i + 2], thread_id)[0];
            break;
        case 1: { // fix
            float tmp = next_input_f(regs[i + 3], thread_id)[0];
            input = static_cast<INT_TYPE>(tmp * (1u << regs[i + 2]));
            i++;
            break;
        }
        case 2: // float
            break;
        }

        switch (regs[i + 2]) {
        case 0:
            s_register[dest].template prepare_receive_and_replicate<P_0>(input);
            break;
        case 1:
            s_register[dest].template prepare_receive_and_replicate<P_1>(input);
            break;
        case 2:
            s_register[dest].template prepare_receive_and_replicate<P_2>(input);
            break;
        }
    }

    sint::communicate();

    for (size_t i = 0; i < regs.size(); i += 3) {
        int dest = regs[i + 1];

        if (regs[i] == 1) {
            i += 1;
        }

        switch (regs[i + 2]) {
        case 0:
            s_register[dest].template complete_receive_from<P_0>();
            break;
        case 1:
            s_register[dest].template complete_receive_from<P_1>();
            break;
        case 2:
            s_register[dest].template complete_receive_from<P_2>();
            break;
        }
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::popen(const vector<int>& regs, const size_t& size) {
    for (size_t i = 0; i < regs.size(); i += 2) {
        for (size_t vec = 0; vec < size; ++vec) {
            s_register[regs[i + 1] + vec].prepare_reveal_to_all();
        }
    }

    sint::communicate();

    for (size_t i = 0; i < regs.size(); i += 2)
        for (size_t vec = 0; vec < size; ++vec)
            c_register[regs[i] + vec] =
                s_register[regs[i + 1] + vec].complete_reveal_to_all_single();
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::mulm(const vector<int>& regs, const size_t& size) {
    for (size_t vec = 0; vec < size; ++vec)
        s_register[regs[0] + vec] =
            s_register[regs[1] + vec].mult_public(UINT_TYPE(c_register[regs[2] + vec].get()));
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::muls(const vector<int>& regs) {
    for (size_t i = 0; i < regs.size(); i += 4) {
        for (int j = 0; j < regs[i]; j++) {
            s_register[regs[i + 1] + j] =
                (s_register[regs[i + 2] + j].prepare_mult(s_register[regs[i + 3] + j]));
        }
    }

    sint::communicate();

    for (size_t i = 0; i < regs.size(); i += 4) {
        for (int j = 0; j < regs[i]; j++) {
            s_register[regs[i + 1] + j].complete_mult_without_trunc();
        }
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::andrsvec(const vector<int>& regs) {
    auto it = regs.begin();
    while (it < regs.end()) {
        int total = *it++;
        int n_args = (total - 3) / 2;
        int vec = *it++;

        for (int i = 0; i < vec; ++i) {
            for (int j = 0; j < n_args; ++j) {
                sb_register[*(it + j) + i / BIT_LEN][i % BIT_LEN] =
                    sb_register[*(it + n_args) + i / BIT_LEN][i % BIT_LEN] &
                    sb_register[*(it + n_args + 1 + j) + i / BIT_LEN][i % BIT_LEN];
            }
        }
        it += total - 2;
    }

    BitShare::communicate();

    it = regs.begin();
    while (it < regs.end()) {
        int total = *it++;
        int n_args = (total - 3) / 2;
        int vec = *it++;

        for (int i = 0; i < vec; ++i) {
            for (int j = 0; j < n_args; ++j) {
                sb_register[*(it + j) + i / BIT_LEN][i % BIT_LEN].complete_and();
            }
        }
        it += total - 2;
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::matmulsm(const vector<int>& regs,
                                                Machine<sint, sbit, BitShare, N>& m) {
    auto res = s_register.begin() + regs[0];
    auto source1 = m.s_mem.begin() + i_register[regs[1]];
    auto source2 = m.s_mem.begin() + i_register[regs[2]];

    int rows = regs[3]; // for 1st but also final
    int cols = regs[5]; // for 2nd but also final

    for (int i = 0; i < rows; ++i) {
        auto row_1 = i_register[regs[6] + i]; // rows to use what ever that means

        for (int j = 0; j < cols; ++j) {
            matmulsm_prepare(regs, row_1, j, source1, source2); // calculate dotprod
        }
    }

    sint::communicate();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            *(res + i * cols + j) = matrix.get_next();
        }
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
template <class iterator>
void Program<sint, sbit, BitShare, N>::matmulsm_prepare(const vector<int>& regs, const int& row_1,
                                                        const int& j, iterator source1,
                                                        iterator source2) {
    auto col_2 = i_register[regs[9] + j]; // column of 2nd factor

    matrix.next_dotprod();
    for (int k = 0; k < regs[4]; ++k) {       // length of dot_prod
        auto col_1 = i_register[regs[7] + k]; // column of first factor
        auto row_2 = i_register[regs[8] + k]; // row of 2nd factor

        iterator cur_1 = source1 + row_1 * regs[10] + col_1;
        iterator cur_2 = source2 + row_2 * regs[11] + col_2;

        matrix.add_mul(*cur_1, *cur_2);
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::dotprods(const vector<int>& regs, const size_t& size) {
    for (size_t vec = 0; vec < size; ++vec) {
        for (auto it = regs.begin(); it != regs.end();) {
            auto next = it + *it;
            it += 2;

            matrix.next_dotprod();
            while (it != next) {
                matrix.add_mul(s_register[*(it++) + vec], s_register[*(it++) + vec]);
            }
        }
    }

    sint::communicate();

    for (size_t vec = 0; vec < size; ++vec) {
        for (auto it = regs.begin(); it != regs.end();) {
            auto next = it + *it;
            it++;
            s_register[*it + vec] = matrix.get_next();
            it = next;
        }
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Program<sint, sbit, BitShare, N>::cisc(const vector<int>& regs, const std::string_view cisc) {
    if (cisc.starts_with("LTZ")) {
        print("LTZ\n");
        for (size_t i = 0; i < regs.size(); i += 6) {
            std::cout << "s" << regs[i + 2] << " = s" << regs[i + 3] << "(" << regs[i + 4]
                      << ") < 0"
                      << "\n";
        }
    } else if (cisc.starts_with("EQZ")) {
        print("EQZ\n");
        for (size_t i = 0; i < regs.size(); i += 6) {
            std::cout << "s" << regs[i + 2] << " = s" << regs[i + 3] << "(" << regs[i + 4]
                      << ") == 0"
                      << "\n";
        }
    }
}

} // namespace IR

#endif
