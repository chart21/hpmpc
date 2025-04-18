#pragma once
#include "../protocols/Protocols.h"
template <typename Share>
class sint_t
{
  private:
    Share shares[BITLENGTH];

  public:
    // temporary constructor
    sint_t() {}

    sint_t(UINT_TYPE value)
    {
        for (int i = 0; i < BITLENGTH; i++)
            shares[i] = Share::public_val(PROMOTE(value));
    }

    template <int id>
    sint_t(UINT_TYPE value)
    {
        alignas(sizeof(DATATYPE)) UINT_TYPE temp_u[DATTYPE] = {value};
        init(temp_u);
    }

    template <int id>
    sint_t(UINT_TYPE value[DATTYPE])
    {
        init(value);
    }
    template <int id>
    void prepare_receive_from()
    {
        for (int i = 0; i < BITLENGTH; i++)
            shares[i].template prepare_receive_from<id>();
    }

    template <int id>
    void prepare_receive_from(DATATYPE values[BITLENGTH])
    {
        for (int i = 0; i < BITLENGTH; i++)
            shares[i].template prepare_receive_from<id>(values[i]);
    }

    template <int id>
    void prepare_receive_and_replicate(UINT_TYPE value)
    {
        if constexpr (id == PSELF || PROTOCOL == 13)
        {
            if (current_phase != PHASE_INIT)
            {  // TODO: Should only happen either in PRE or in live phase
                DATATYPE temp_u[BITLENGTH];
                for (int i = 0; i < BITLENGTH; i++)
                    shares[i].template prepare_receive_from<id>(PROMOTE(value));
                return;
            }
        }
        prepare_receive_from<id>();
    }

    template <int id>
    void complete_receive_from()
    {
        for (int i = 0; i < BITLENGTH; i++)
            shares[i].template complete_receive_from<id>();
    }

    template <int id>
    void init(UINT_TYPE value[DATTYPE])
    {
        if constexpr (id == PSELF)
        {
            if (current_phase != PHASE_INIT)
            {  // TODO: Should only happen either in PRE or in live phase

                DATATYPE temp_d[BITLENGTH];
                orthogonalize_arithmetic(value, temp_d);
                for (int i = 0; i < BITLENGTH; i++)
                    shares[i].template prepare_receive_from<id>(temp_d[i]);
            }
        }

        for (int i = 0; i < BITLENGTH; i++)
        {
            shares[i].template prepare_receive_from<id>();
        }
    }

    Share& operator[](int idx) { return shares[idx]; }

    const Share& operator[](int idx) const { return shares[idx]; }

    sint_t operator+(const sint_t& other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i] + other[i];
        }
        return result;
    }

    sint_t operator+(const Share& other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i] + other;
        }
        return result;
    }

    sint_t operator-(const sint_t& other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i] - other[i];
        }
        return result;
    }

    sint_t operator-(const Share& other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i] - other;
        }
        return result;
    }

    sint_t operator*(const sint_t& other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i] * other[i];
        }
        return result;
    }

    sint_t operator*(const Share& other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i] * other;
        }
        return result;
    }

    void operator+=(const sint_t& other)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i] = shares[i] + other[i];
        }
    }

    void operator+=(const Share other)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i] = shares[i] + other;
        }
    }

    void operator-=(const sint_t& other)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i] = shares[i] - other[i];
        }
    }

    void operator-=(const Share other)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i] = shares[i] - other;
        }
    }

    sint_t operator*(const UINT_TYPE other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i] * other;
        }
        return result;
    }

    sint_t prepare_mult_public_fixed_dat(const DATATYPE other, int fractional_bits = FRACTIONAL) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].prepare_mult_public_fixed_dat(other, fractional_bits);
        }
        return result;
    }

    sint_t prepare_mult_public_fixed(const UINT_TYPE other, int fractional_bits = FRACTIONAL) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].prepare_mult_public_fixed(other, fractional_bits);
        }
        return result;
    }

    sint_t prepare_trunc_share(int fractional_bits = FRACTIONAL) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].prepare_trunc_share(fractional_bits);
        }
        return result;
    }

    void operator*=(const UINT_TYPE other)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i] = shares[i].prepare_mult_public_fixed(other);
        }
    }

    void complete_public_mult_fixed()
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i].complete_public_mult_fixed();
        }
    }

    sint_t mult_public(const UINT_TYPE other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].mult_public(other);
        }
        return result;
    }

    sint_t mult_public_dat(const DATATYPE other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].mult_public_dat(other);
        }
        return result;
    }

    sint_t prepare_mult(const sint_t& other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].prepare_mult(other[i]);
        }
        return result;
    }

    bool operator==(const sint_t& b) const
    {
        return false;  // Needed for Eigen optimizations
    }

    void operator*=(const sint_t& other)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i] = shares[i] * other[i];
        }
    }

    void complete_mult()
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i].complete_mult();
        }
    }

    void complete_mult_without_trunc()
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i].complete_mult_without_trunc();
        }
    }

    sint_t prepare_dot(const sint_t& other) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].prepare_dot(other[i]);
        }
        return result;
    }
#if MULTI_INPUT == 1
    sint_t prepare_dot3(const sint_t& other, const sint_t& other2) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].prepare_dot3(other[i], other2[i]);
        }
        return result;
    }

    sint_t prepare_dot4(const sint_t& other, const sint_t& other2, const sint_t& other3) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
        {
            result[i] = shares[i].prepare_dot4(other[i], other2[i], other3[i]);
        }
        return result;
    }
#endif
    void complete_receive_from(int id)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i].template complete_receive_from<id>();
        }
    }

    void prepare_reveal_to_all() const
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i].prepare_reveal_to_all();
        }
    }

    void complete_reveal_to_all(UINT_TYPE result[DATTYPE]) const
    {
        DATATYPE temp[BITLENGTH];
        for (int i = 0; i < BITLENGTH; ++i)
        {
            temp[i] = shares[i].complete_reveal_to_all();
        }
        unorthogonalize_arithmetic(temp, result);
    }

    UINT_TYPE complete_reveal_to_all_single() const
    {
        DATATYPE temp[BITLENGTH];
        alignas(sizeof(DATATYPE)) UINT_TYPE result[DATTYPE];
        for (int i = 0; i < BITLENGTH; ++i)
        {
            temp[i] = shares[i].complete_reveal_to_all();
        }
        unorthogonalize_arithmetic(temp, result);
        return result[0];
    }

    Share* get_share_pointer() { return shares; }

    Share get_share(int idx) const { return shares[idx]; }

    static sint_t<Share> load_shares(int l, const Share shares[])
    {
        sint_t<Share> result;
        for (int i = 0; i < l; ++i)
        {
            result[i] = shares[i];
        }
        for (int i = l; i < BITLENGTH; ++i)
        {
            result[i] = Share::public_val(PROMOTE(0));
        }
        return result;
    }

    static sint_t<Share> load_shares(const Share shares[BITLENGTH]) { return load_shares(BITLENGTH, shares); }

    static sint_t<Share> move_shares(Share* shares)
    {
        sint_t<Share> result;
        std::move(shares, shares + BITLENGTH, result.shares);
        return result;
    }

    void prepare_XOR(const sint_t<Share>& a, const sint_t<Share>& b)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            shares[i] = a[i].prepare_mult(b[i]);
        }
    }

    void complete_XOR(const sint_t<Share>& a, const sint_t<Share>& b)
    {
        for (int i = 0; i < BITLENGTH; ++i)
        {
            /* shares[i].complete_XOR(a[i], b[i]); */
            shares[i].complete_mult_without_trunc();
            shares[i] = a[i] + b[i] - shares[i] - shares[i];
        }
    }

    void complete_bit_injection_S1() { Share::complete_bit_injection_S1(shares); }

    void complete_opt_bit_injection()
    {
        for (int i = 0; i < BITLENGTH; ++i)
            shares[i].complete_opt_bit_injection();
    }

    void complete_bit2a()
    {
        for (int i = 0; i < BITLENGTH; ++i)
            shares[i].complete_bit2a();
    }

    void mask_and_send_dot()
    {
        for (int i = 0; i < BITLENGTH; ++i)
            shares[i].mask_and_send_dot();
    }

    void mask_and_send_dot_without_trunc()
    {
        for (int i = 0; i < BITLENGTH; ++i)
            shares[i].mask_and_send_dot_without_trunc();
    }

    void complete_bit_injection_S2() { Share::complete_bit_injection_S2(shares); }

    static void communicate() { Share::communicate(); }

    sint_t relu() const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
            result.shares[i] = shares[i].relu();
        return result;
    }

    void prepare_trunc_2k_inputs(sint_t& rmk2,
                                 sint_t& rmsb,
                                 sint_t& c,
                                 sint_t& c_prime,
                                 int fractional_bits = FRACTIONAL)
    {
        for (int i = 0; i < BITLENGTH; ++i)
            shares[i].prepare_trunc_2k_inputs(
                rmk2.shares[i], rmsb.shares[i], c.shares[i], c_prime.shares[i], fractional_bits);
    }

    void complete_trunc_2k_inputs(sint_t& rmk2, sint_t& rmsb, sint_t& c, sint_t& c_prime)
    {
        for (int i = 0; i < BITLENGTH; ++i)
            shares[i].complete_trunc_2k_inputs(rmk2.shares[i], rmsb.shares[i], c.shares[i], c_prime.shares[i]);
    }

    /* #if TRUNC_APPROACH == 2 */
    template <typename X, typename A>
    static void prepare_B2A(X z[], X r[], A out[])
    {
        Share::prepare_B2A(z, r, out);
    }

    template <typename X, typename A>
    static void complete_B2A(X z[], A out[])
    {
        Share::complete_B2A(z, out);
    }
    template <typename X, typename A>
    static void complete_B2A2(X z[], A out[])
    {
        Share::complete_B2A2(z, out);
    }
    /* #endif */

    sint_t prepare_trunc_exact_xmod2t(int fractional_bits = FRACTIONAL) const
    {
        sint_t result;
        for (int i = 0; i < BITLENGTH; ++i)
            result[i] = shares[i].prepare_trunc_exact_xmod2t(fractional_bits);
        return result;
    }
};
