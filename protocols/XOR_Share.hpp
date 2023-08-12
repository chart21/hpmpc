#pragma once
#include "Share.hpp"
#include <functional>

template <typename Datatype, typename Share_Type>
class XOR_Share : public Share_Type
{
public:
    XOR_Share() {}

    XOR_Share(const Share_Type& s) : Share_Type(s) {}

    XOR_Share operator~() const
    {
        return XOR_Share(Share_Type::Not());
    }

    XOR_Share operator^(const XOR_Share<Datatype,Share_Type>& b) const
    {
        return XOR_Share(Share_Type::Add(b, std::bit_xor<Datatype>()));
    }

    XOR_Share operator&(const XOR_Share<Datatype, Share_Type>& b) const
    {
        return XOR_Share(Share_Type::prepare_mult(b, std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>()));
    }

    void complete_and()
    {
        Share_Type::complete_mult(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
    }

    template <int id>
    void prepare_receive_from()
    {
        Share_Type::template prepare_receive_from<id>(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
    }

    template <int id>
    void complete_receive_from()
    {
        Share_Type::template complete_receive_from<id>(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
    }
    
    void prepare_reveal_to_all()
    {
        Share_Type::prepare_reveal_to_all();
    }

    Datatype complete_reveal_to_all()
    {
        return Share_Type::complete_Reveal(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
    }
};

