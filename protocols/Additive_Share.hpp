#pragma once
#include "Share.hpp"
#include <functional>

template <typename Datatype, typename Share_Type>
class Additive_Share : public Share_Type
{
public:
    Additive_Share() {}

    Additive_Share(const Share_Type& s) : Share_Type(s) {}

    Additive_Share operator+(const Additive_Share<Datatype,Share_Type>& b) const
    {
        return Additive_Share(Share_Type::Add(b, std::plus<Datatype>()));
    }
    
    Additive_Share operator-(const Additive_Share<Datatype,Share_Type>& b) const
    {
        return Additive_Share(Share_Type::Add(b, std::minus<Datatype>()));
    }

    Additive_Share operator*(const Additive_Share<Datatype, Share_Type>& b) const
    {
        return Additive_Share(Share_Type::prepare_mult(b, std::plus<Datatype>(), std::minus<Datatype>(), std::multiplies<Datatype>()));
    }

    void complete_mult()
    {
        Share_Type::complete_mult(std::plus<Datatype>(), std::minus<Datatype>());
    }

    template <int id>
    void prepare_receive_from()
    {
        Share_Type::template prepare_receive_from<id>(std::plus<Datatype>(), std::minus<Datatype>());
    }

    template <int id>
    void complete_receive_from()
    {
        Share_Type::template complete_receive_from<id>(std::plus<Datatype>(), std::minus<Datatype>());
    }
    
    void prepare_reveal_to_all()
    {
        Share_Type::prepare_reveal_to_all();
    }

    Datatype complete_reveal_to_all()
    {
        return Share_Type::complete_Reveal(std::plus<Datatype>(), std::minus<Datatype>());
    }
};


