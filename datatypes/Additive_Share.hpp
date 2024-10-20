#pragma once
#include "../protocols/Protocols.h"
template <typename Datatype, typename Share_Type>
class Additive_Share : public Share_Type
{
public:
    Additive_Share() {}

    Additive_Share(const Share_Type& s) : Share_Type(s) {}

    Additive_Share(UINT_TYPE a) : Share_Type() {
        *this = Share_Type::public_val(PROMOTE(a));
    }

    static Additive_Share get_share_from_public_dat(Datatype a) {
        return Share_Type::public_val(a);
    }

    Additive_Share operator+(const Additive_Share<Datatype,Share_Type>& b) const
    {
#if PROTOCOL == 2
        return Additive_Share(Share_Type::Add(b, OP_SUB)); //Replicated needs substraction
#else
        return Additive_Share(Share_Type::Add(b, OP_ADD));
#endif
    }

    void operator+=(const Additive_Share<Datatype,Share_Type>& b)
    {
        *this = *this + b;
    }
    
    Additive_Share operator-(const Additive_Share<Datatype,Share_Type>& b) const
    {
#if PROTOCOL == 2
        return Additive_Share(Share_Type::Add(b, OP_ADD)); //Replicated needs addition
#else
        return Additive_Share(Share_Type::Add(b, OP_SUB));
#endif
    }

    Additive_Share operator*(const Additive_Share<Datatype, Share_Type>& b) const
    {
        return Additive_Share(Share_Type::prepare_dot(b, OP_ADD, OP_SUB, OP_MULT));
    }
        
        Additive_Share operator*(const UINT_TYPE other) const
        {
        return Additive_Share(Share_Type::prepare_mult_public_fixed(PROMOTE(other), OP_MULT, OP_ADD, OP_SUB, FUNC_TRUNC));
        }

        void operator*=(const UINT_TYPE other) 
        {
        *this = this->prepare_mult_public_fixed(other);
        }

        void operator*=(const Additive_Share<Datatype, Share_Type>& other) 
        {
        *this = *this * other;
        }

    Additive_Share mult_public(const UINT_TYPE b) const
    {
        return Additive_Share(Share_Type::mult_public(PROMOTE(b), OP_MULT));
    }
    
    Additive_Share mult_public_dat(const Datatype b) const
    {
        return Additive_Share(Share_Type::mult_public(b, OP_MULT));
    }

    Additive_Share prepare_mult_public_fixed(const UINT_TYPE b, int fractional_bits = FRACTIONAL) const
    {
        return Share_Type::prepare_mult_public_fixed(PROMOTE(b), OP_MULT, OP_ADD, OP_SUB, OP_TRUNCF, fractional_bits);
    }
    
    Additive_Share prepare_trunc_share(int fractional_bits = FRACTIONAL) const
    {
        return Share_Type::prepare_trunc_share(OP_MULT, OP_ADD, OP_SUB, OP_TRUNCF, fractional_bits);
    }
    
    Additive_Share prepare_div_exp2(const int b) const
    {
        return Additive_Share(Share_Type::prepare_div_exp2(b, OP_MULT, OP_ADD, OP_SUB, OP_TRUNC2));
    }

    Additive_Share prepare_mult_public_fixed_dat(const Datatype b, int fractional_bits = FRACTIONAL) const
    {
        return Share_Type::prepare_mult_public_fixed(b, OP_MULT, OP_ADD, OP_SUB, OP_TRUNCF, fractional_bits);
    }

    void complete_public_mult_fixed()
    {
        Share_Type::complete_public_mult_fixed(OP_ADD, OP_SUB);
    }
    
    
    bool operator==(const Additive_Share& b) const
    {
        return false; // Needed for Eigen optimizations
    }

    template <int id>
    void prepare_receive_from()
    {
        Share_Type::template prepare_receive_from<id>(OP_ADD, OP_SUB);
    }
    
    template <int id>
    void prepare_receive_from(DATATYPE val)
    {
        Share_Type::template prepare_receive_from<id>(val, OP_ADD, OP_SUB);
    }

    template<int id>
    void prepare_receive_and_replicate(UINT_TYPE value) {
        if constexpr (id == PSELF || PROTOCOL == 13) {
          if (current_phase != PHASE_INIT) { //TODO: Should only happen either in PRE or in live pahse
            /* prepare_receive_from<id>(PROMOTE(value)); */
            prepare_receive_from<id>(PROMOTE(value)); 
            return;
          }
        }
            prepare_receive_from<id>();
        }


    template <int id>
    void complete_receive_from()
    {
        Share_Type::template complete_receive_from<id>(OP_ADD, OP_SUB);
    }
    
    void prepare_reveal_to_all() const
    {
        Share_Type::prepare_reveal_to_all();
    }
    void prepare_trunc_2k_inputs(Share_Type& rmk2, Share_Type& rmsb, Share_Type& c, Share_Type& c_prime, int fractional_bits = FRACTIONAL)
    {
       Share_Type::prepare_trunc_2k_inputs(OP_ADD, OP_SUB, FUNC_XOR, FUNC_AND, OP_TRUNCF, rmk2, rmsb, c, c_prime, fractional_bits);
    }

    void complete_trunc_2k_inputs(Share_Type& rmk2, Share_Type& rmsb, Share_Type& c, Share_Type& c_prime)
    {
        Share_Type::complete_trunc_2k_inputs(OP_ADD, OP_SUB, FUNC_XOR, FUNC_AND, FUNC_TRUNC, rmk2, rmsb, c, c_prime);
    }


    Datatype complete_reveal_to_all() const
    {
        return Share_Type::complete_Reveal(OP_ADD, OP_SUB);
    }

    void complete_reveal_to_all(UINT_TYPE output[]) const
    {
        auto res = Share_Type::complete_Reveal(OP_ADD, OP_SUB); 
        unorthogonalize_arithmetic(&res, output,1);
    }
    
        
    UINT_TYPE complete_reveal_to_all_single() const {
        auto res = Share_Type::complete_Reveal(OP_ADD, OP_SUB); //TODO: Use extract method from Intrinsics
        alignas(DATATYPE) UINT_TYPE ret[DATTYPE/BITLENGTH];
        unorthogonalize_arithmetic(&res, ret,1);
        return ret[0];
        }

    Additive_Share prepare_mult(const Additive_Share<Datatype, Share_Type>& b) const
    {
        return Additive_Share(Share_Type::prepare_mult(b, OP_ADD, OP_SUB, OP_MULT));
    }

    Additive_Share prepare_mult3(const Additive_Share<Datatype, Share_Type>& b, const Additive_Share<Datatype, Share_Type>& c) const
    {
        return Additive_Share(Share_Type::prepare_mult3(b, c, OP_ADD, OP_SUB, OP_MULT));
    }
        
    void prepare_XOR(const Additive_Share &a, const Additive_Share &b) {
                *this = a.prepare_mult(b);
        }

    void complete_XOR(const Additive_Share &a, const Additive_Share &b) {
                this->complete_mult_without_trunc();
                *this = a + b - *this - *this;
            }
    
    void complete_mult3()
    {
        Share_Type::complete_mult3(OP_ADD, OP_SUB);
    }
    
    Additive_Share prepare_mult4(const Additive_Share<Datatype, Share_Type>& b, const Additive_Share<Datatype, Share_Type>& c, const Additive_Share<Datatype, Share_Type>& d) const
    {
        return Additive_Share(Share_Type::prepare_mult4(b, c, d, OP_ADD, OP_SUB, OP_MULT));
    }
    
    void complete_mult4()
    {
        Share_Type::complete_mult4(OP_ADD, OP_SUB);
    }
    
    Additive_Share prepare_dot(const Additive_Share<Datatype, Share_Type>& b) const
    {
        return Additive_Share(Share_Type::prepare_dot(b, OP_ADD, OP_SUB, OP_MULT));
    }
#if FUSE_DOT != 1
    Additive_Share prepare_dot(const Additive_Share<Datatype, Share_Type>& b, int i) const
    {
        return Additive_Share(Share_Type::prepare_dot(b, i, OP_ADD, OP_SUB, OP_MULT));
    }
    void join_dots(Additive_Share<Datatype, Share_Type> b[]) 
    {
        Share_Type::join_dots(b, OP_ADD, OP_SUB);
    }
#endif

    Additive_Share prepare_dot3(const Additive_Share<Datatype, Share_Type>& b, const Additive_Share<Datatype, Share_Type>& c) const
    {
        return Additive_Share(Share_Type::prepare_dot3(b, c, OP_ADD, OP_SUB, OP_MULT));
    }

    Additive_Share prepare_dot4(const Additive_Share<Datatype, Share_Type>& b, const Additive_Share<Datatype, Share_Type>& c, const Additive_Share<Datatype, Share_Type>& d) const
    {
        return Additive_Share(Share_Type::prepare_dot4(b, c, d, OP_ADD, OP_SUB, OP_MULT));
    }
    
    void mask_and_send_dot() 
    {
        #if FRACTIONAL > 0
        Share_Type::mask_and_send_dot_with_trunc(OP_ADD, OP_SUB, FUNC_TRUNC);
        #else
        #if PROTOCOL == 2
        Share_Type::mask_and_send_dot(OP_SUB, OP_MULT); // Replicated needs custom overloads because division by 3 is required
        #else
        Share_Type::mask_and_send_dot(OP_ADD, OP_SUB);
        #endif
        #endif
    }
    
    void mask_and_send_dot_without_trunc() 
    {
        #if PROTOCOL == 2
        Share_Type::mask_and_send_dot(OP_SUB, OP_MULT); // Replicated needs custom overloads because division by 3 is required
        #else
        Share_Type::mask_and_send_dot(OP_ADD, OP_SUB);
        #endif
    }
    
    void complete_mult()
    {
    #if PROTOCOL == 1 || (PROTOCOL == 4 && PRE == 0)// Sharemind and Additive need custom overload
        Share_Type::complete_mult(OP_ADD, OP_SUB, OP_MULT);
    #else
        #if FRACTIONAL > 0
        Share_Type::complete_mult_with_trunc(OP_ADD, OP_SUB, FUNC_TRUNC);
        #else
        Share_Type::complete_mult(OP_ADD, OP_SUB);
        #endif
    #endif
    }

        void complete_mult_without_trunc()
    {
    #if PROTOCOL == 1 || (PROTOCOL == 4 && PRE == 0)// Sharemind and Additive need custom overload
        Share_Type::complete_mult(OP_ADD, OP_SUB, OP_MULT);
    #else
        Share_Type::complete_mult(OP_ADD, OP_SUB);
        #endif
    }

    Share_Type get_share() const
    {
        return (Share_Type)*this;
    }

    void set_share(const Share_Type& s)
    {
        *this = Additive_Share(s);
    }
    
   Additive_Share  prepare_trunc_exact_xmod2t(int fractional_bits = FRACTIONAL) const
   {
       return Share_Type::prepare_trunc_exact_xmod2t(OP_ADD, OP_SUB, OP_TRUNCF, FUNC_AND, fractional_bits);
   }
};


