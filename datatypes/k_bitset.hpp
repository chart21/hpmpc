#pragma once
#include "../protocols/Protocols.h"
template <int k, typename Share>
class sbitset_t
{
  private:
    Share shares[k];

  public:
    // temporary constructor
    sbitset_t() {}

    template <int id>
    sbitset_t(UINT_TYPE value)
    {
        UINT_TYPE temp_u[DATTYPE] = {value};
        init(temp_u);
    }

    sbitset_t(UINT_TYPE value[DATTYPE])
    {
        DATATYPE temp_d[BITLENGTH];

        orthogonalize_boolean(value, temp_d);
        for (int i = 0; i < k; i++)
            shares[i] = Share(temp_d[i]);
    }

    template <int id>
    sbitset_t(UINT_TYPE value[DATTYPE])
    {
        init(value);
    }

    template <int id>
    void prepare_receive_from()
    {
        for (int i = 0; i < k; i++)
            shares[i].template prepare_receive_from<id>();
    }

    template <int id>
    void complete_receive_from()
    {
        for (int i = 0; i < k; i++)
            shares[i].template complete_receive_from<id>();
    }

    template <int id>
    void init(UINT_TYPE value[DATTYPE])
    {
        if constexpr (id == PSELF)
        {
            if (current_phase != PHASE_INIT)
            {  // TODO: Should only happen either in PRE or in live pahse

                DATATYPE temp_d[BITLENGTH];
                orthogonalize_boolean(value, temp_d);
                for (int i = 0; i < k; i++)
                    shares[i] = Share(temp_d[i]);
            }
        }
        for (int i = 0; i < k; i++)
        {
            shares[i].template prepare_receive_from<id>();
        }
    }

    Share& operator[](int idx) { return shares[idx]; }

    const Share& operator[](int idx) const { return shares[idx]; }

    sbitset_t operator^(const sbitset_t& other) const
    {
        sbitset_t result;
        for (int i = 0; i < k; ++i)
        {
            result[i] = shares[i] ^ other[i];
        }
        return result;
    }

    sbitset_t operator~() const
    {
        sbitset_t result;
        for (int i = 0; i < k; ++i)
        {
            result[i] = ~shares[i];
        }
        return result;
    }

    sbitset_t operator&(const sbitset_t& other) const
    {
        sbitset_t result;
        for (int i = 0; i < k; ++i)
        {
            result[i] = shares[i] & other[i];
        }
        return result;
    }

    sbitset_t prepare_and(const sbitset_t& other) const
    {
        sbitset_t result;
        for (int i = 0; i < k; ++i)
        {
            result[i] = shares[i].prepare_and(other[i]);
        }
        return result;
    }

    sbitset_t and_public(const UINT_TYPE other) const
    {
        sbitset_t result;
        for (int i = 0; i < k; ++i)
        {
            result[i] = shares[i].and_public(other);
        }
        return result;
    }

    void complete_and()
    {
        for (int i = 0; i < k; ++i)
        {
            shares[i].complete_and();
        }
    }

    void complete_receive_from(int id)
    {
        for (int i = 0; i < k; ++i)
        {
            shares[i].template complete_receive_from<id>();
        }
    }

    void prepare_reveal_to_all()
    {
        for (int i = 0; i < k; ++i)
        {
            shares[i].prepare_reveal_to_all();
        }
    }

    void complete_reveal_to_all(UINT_TYPE result[DATTYPE])
    {
        DATATYPE temp[BITLENGTH]{0};
        /* DATATYPE* temp = (DATATYPE*) result; */
        for (int i = 0; i < k; ++i)
        {
            temp[i] = shares[i].complete_reveal_to_all();
        }
        unorthogonalize_boolean((DATATYPE*)temp, result);
    }

    Share* get_share_pointer() { return shares; }

    static sbitset_t<k, Share> load_shares(Share shares[k])
    {
        sbitset_t<k, Share> result;
        for (int i = 0; i < k; ++i)
        {
            result[i] = shares[i];
        }
        return result;
    }

    static sbitset_t prepare_A2B_S1(int m, Share s[k])
    {
        sbitset_t<k, Share> result;
        Share::prepare_A2B_S1(m, k + m, s, result.get_share_pointer());
        return result;
    }

    static sbitset_t prepare_A2B_S2(int m, Share s[k])
    {
        sbitset_t<k, Share> result;
        Share::prepare_A2B_S2(m, k + m, s, result.get_share_pointer());
        return result;
    }

    static sbitset_t prepare_A2B_S1(Share s[k]) { return prepare_A2B_S1(0, s); }

    static sbitset_t prepare_A2B_S2(Share s[k]) { return prepare_A2B_S2(0, s); }

    void complete_A2B_S1(int m) { Share::complete_A2B_S1(k + m, shares); }

    void complete_A2B_S2(int m) { Share::complete_A2B_S2(k + m, shares); }

    void complete_A2B_S1() { Share::complete_A2B_S1(k, shares); }

    void complete_A2B_S2() { Share::complete_A2B_S2(k, shares); }

    // TODO: does this really belong here? -> complete result should be in sint
    void complete_bit_injection_S1() { Share::complete_bit_injection_S1(shares); }

    void complete_bit_injection_S2() { Share::complete_bit_injection_S2(shares); }

    static constexpr int get_bitlength() { return k; }
};
