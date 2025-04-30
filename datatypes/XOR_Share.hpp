#pragma once
#include "../protocols/Protocols.h"

template <typename Datatype, typename Share_Type>
class XOR_Share : public Share_Type
{
  public:
    XOR_Share() {}

    XOR_Share(const Share_Type& s) : Share_Type(s) {}

    XOR_Share(const Datatype& d) { *this = Share_Type::public_val(d); }

    XOR_Share operator~() const { return XOR_Share(Share_Type::Not()); }

    XOR_Share operator!() const { return XOR_Share(Share_Type::Not()); }

    XOR_Share operator^(const XOR_Share<Datatype, Share_Type>& b) const
    {
        return XOR_Share(Share_Type::Add(b, std::bit_xor<Datatype>()));
    }

    XOR_Share operator&(const XOR_Share<Datatype, Share_Type>& b) const
    {
        return XOR_Share(
            Share_Type::prepare_mult(b, std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>()));
    }

    XOR_Share prepare_and(const XOR_Share<Datatype, Share_Type>& b) const
    {
        return XOR_Share(
            Share_Type::prepare_mult(b, std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>()));
    }

    XOR_Share and_public(const Datatype& b) const
    {
        return XOR_Share(Share_Type::mult_public(b, std::bit_and<Datatype>()));
    }

    void complete_and()
    {
#if PROTOCOL == 1  // Sharemind needs custom overload
        Share_Type::complete_mult(std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>());
#else
        Share_Type::complete_mult(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
#endif
    }

    template <int id>
    void prepare_receive_from()
    {
        Share_Type::template prepare_receive_from<id>(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
    }

    template <int id>
    void prepare_receive_from(
        Datatype val)  // If all bits should be 0 use SET_ALL_ZERO(), if all bits should be 1 use SET_ALL_ONE()
    {
        Share_Type::template prepare_receive_from<id>(val, std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
    }

    template <int id>
    void complete_receive_from()
    {
        Share_Type::template complete_receive_from<id>(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
    }

    void prepare_reveal_to_all() { Share_Type::prepare_reveal_to_all(); }

    Datatype complete_reveal_to_all()
    {
        return Share_Type::complete_Reveal(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
    }

    int get_p1() { return 0; }

    XOR_Share prepare_dot(const XOR_Share<Datatype, Share_Type>& b) const
    {
        return XOR_Share(
            Share_Type::prepare_dot(b, std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>()));
    }

    XOR_Share prepare_dot3(const XOR_Share<Datatype, Share_Type>& b, const XOR_Share<Datatype, Share_Type>& c) const
    {
        return XOR_Share(Share_Type::prepare_dot3(
            b, c, std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>()));
    }
    XOR_Share prepare_and3(const XOR_Share<Datatype, Share_Type>& b, const XOR_Share<Datatype, Share_Type>& c) const
    {
        return XOR_Share(Share_Type::prepare_mult3(
            b, c, std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>()));
    }

    void complete_and3() { Share_Type::complete_mult3(std::bit_xor<Datatype>(), std::bit_xor<Datatype>()); }
    XOR_Share prepare_dot4(const XOR_Share<Datatype, Share_Type>& b,
                           const XOR_Share<Datatype, Share_Type>& c,
                           const XOR_Share<Datatype, Share_Type>& d) const
    {
        return XOR_Share(Share_Type::prepare_dot4(
            b, c, d, std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>()));
    }

    XOR_Share prepare_and4(const XOR_Share<Datatype, Share_Type>& b,
                           const XOR_Share<Datatype, Share_Type>& c,
                           const XOR_Share<Datatype, Share_Type>& d) const
    {
        return XOR_Share(Share_Type::prepare_mult4(
            b, c, d, std::bit_xor<Datatype>(), std::bit_xor<Datatype>(), std::bit_and<Datatype>()));
    }
    void complete_and4() { Share_Type::complete_mult4(std::bit_xor<Datatype>(), std::bit_xor<Datatype>()); }

    void mask_and_send_dot()
    {
#if PROTOCOL == 2
        Share_Type::mask_and_send_dot(
            std::bit_xor<Datatype>(),
            std::bit_and<Datatype>());  // Replicated needs custom overloads because division by 3 is required
#else
        Share_Type::mask_and_send_dot(std::bit_xor<Datatype>(), std::bit_xor<Datatype>());
#endif
    }
};
