#pragma once
#include <functional>
#include "../arch/DATATYPE.h"

template <typename Datatype, typename Share_Type>
class Matrix_Share : public Share_Type
{
public:
    Matrix_Share() {}
    Matrix_Share(int tmp) {}

    Matrix_Share(const Share_Type& s) : Share_Type(s) {}

    Matrix_Share operator+(const Matrix_Share<Datatype,Share_Type>& b) const
    {
        return Matrix_Share(Share_Type::Add(b, OP_ADD));
    }
    
    Matrix_Share operator-(const Matrix_Share<Datatype,Share_Type>& b) const
    {
        return Matrix_Share(Share_Type::Add(b, OP_SUB));
    }

    Matrix_Share operator*(const Matrix_Share<Datatype, Share_Type>& b) const
    {
        return Matrix_Share(Share_Type::prepare_dot(b, OP_ADD, OP_SUB, OP_MULT));
    }

    void operator+=(const Matrix_Share<Datatype, Share_Type>& b)
    {
        *this = *this + b;
    }

    bool operator==(const Matrix_Share<Datatype, Share_Type>& b) const
    {
        return false; // Needed for Eigen optimizations
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
        #if FRACTIONAL > 0
        Share_Type::complete_mult_with_trunc(OP_ADD, OP_SUB, OP_TRUNC);
        #else
        Share_Type::complete_mult(OP_ADD, OP_SUB);
        #endif
    }




    template <int id>
    void prepare_receive_from()
    {
        Share_Type::template prepare_receive_from<id>(OP_ADD, OP_SUB);
    }

    template <int id>
    void complete_receive_from()
    {
        Share_Type::template complete_receive_from<id>(OP_ADD, OP_SUB);
    }
    
    void prepare_reveal_to_all()
    {
        Share_Type::prepare_reveal_to_all();
    }

    Datatype complete_reveal_to_all()
    {
        return Share_Type::complete_Reveal(OP_ADD, OP_SUB);
    }
};


