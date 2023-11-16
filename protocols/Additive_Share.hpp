#pragma once
/* #include "Share.hpp" */
#include <functional>
#include "../arch/DATATYPE.h"

template <typename Datatype, typename Share_Type>
class Additive_Share : public Share_Type
{
public:
    Additive_Share() {}

    Additive_Share(const Share_Type& s) : Share_Type(s) {}

    Additive_Share operator+(const Additive_Share<Datatype,Share_Type>& b) const
    {
        return Additive_Share(Share_Type::Add(b, OP_ADD));
    }
    
    Additive_Share operator-(const Additive_Share<Datatype,Share_Type>& b) const
    {
        return Additive_Share(Share_Type::Add(b, OP_SUB));
    }

    Additive_Share operator*(const Additive_Share<Datatype, Share_Type>& b) const
    {
        return Additive_Share(Share_Type::prepare_mult(b, OP_ADD, OP_SUB, OP_MULT));
    }

    void operator*=(const DATATYPE b)
    {
        *this = Share_Type::mult_public(b, OP_MULT);
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


    template <int id>
    void complete_receive_from()
    {
        Share_Type::template complete_receive_from<id>(OP_ADD, OP_SUB);
    }
    
    void prepare_reveal_to_all() const
    {
        Share_Type::prepare_reveal_to_all();
    }

    Datatype complete_reveal_to_all() const
    {
        return Share_Type::complete_Reveal(OP_ADD, OP_SUB);
    }

    Additive_Share prepare_mult3(const Additive_Share<Datatype, Share_Type>& b, const Additive_Share<Datatype, Share_Type>& c) const
    {
        return Additive_Share(Share_Type::prepare_mult3(b, c, OP_ADD, OP_SUB, OP_MULT));
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
        Share_Type::mask_and_send_dot_with_trunc(OP_ADD, OP_SUB, OP_TRUNC);
        #else
        #if PROTOCOL == 2
        Share_Type::mask_and_send_dot(OP_SUB, OP_MULT); // Replicated needs custom overloads because division by 3 is required
        #else
        Share_Type::mask_and_send_dot(OP_ADD, OP_SUB);
        #endif
        #endif
    }
    
    void complete_mult()
    {
    #if PROTOCOL == 1 // Sharemind needs custom overload
        Share_Type::complete_mult(OP_ADD, OP_SUB, OP_MULT);
    #else
        #if FRACTIONAL > 0
        Share_Type::complete_mult_with_trunc(OP_ADD, OP_SUB, OP_TRUNC);
        #else
        Share_Type::complete_mult(OP_ADD, OP_SUB);
        #endif
    #endif
    }


};


