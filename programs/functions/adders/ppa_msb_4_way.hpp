#pragma once
#include "../../../datatypes/k_bitset.hpp"
#include "../../../protocols/Protocols.h"
template <int k, typename Share>
class PPA_MSB_4Way
{
    using Bitset = sbitset_t<k, Share>;

  private:
    Bitset& a;
    Bitset& b;
    Share& msb;
    int level;
    std::vector<Share> v;

  public:
    // constructor

    // g1 ⊕p1g2 ⊕p1p2g3 ⊕p1p2p3g4 ⊕p1p2p3p4g5
    // 5way
    Share prepare_W5(Share g1,
                     Share p1,
                     Share g2,
                     Share p2,
                     Share g3,
                     Share p3,
                     Share g4,
                     Share g5,
                     Share p1234_1,
                     Share p1234_2,
                     Share p1234_3)
    {
        /* return g1 ^ (p1 & g2) ^ p1.prepare_and3(p2, g3) ^ p1.prepare_and4(p2, p3, g4) ^ p1234_1.prepare_and4(p1234_2,
         * p1234_3, g5); */
        auto val = p1.prepare_dot(g2) ^ p1.prepare_dot3(p2, g3) ^ p1.prepare_dot4(p2, p3, g4) ^
                   p1234_1.prepare_dot4(p1234_2, p1234_3, g5);
        val.mask_and_send_dot();
        val = val ^ g1;
        return val;
    }

    void complete_W5(Share& val)

    {
        /* val.complete_and(); */
        /* val.complete_and3(); */
        /* val.complete_and4(); */
        /* val.complete_and4(); */
        val.complete_and();
    }
    // g1 ⊕p1g2 ⊕p1p2g3 ⊕p1p2p3g4
    Share prepare_W4_3(Share g1, Share p1, Share g2, Share g3, Share p3, Share g4, Share p12)
    {
        /* return g1 ^ (p1 & g2) ^ (p12 & g3) ^ p12.prepare_and3(p3, g4); */
        auto val = p1.prepare_dot(g2) ^ p12.prepare_dot(g3) ^ p12.prepare_dot3(p3, g4);
        val.mask_and_send_dot();
        val = val ^ g1;
        return val;
    }

    void complete_W4_3(Share& val)

    {
        /* val.complete_and(); */
        /* val.complete_and(); */
        /* val.complete_and3(); */
        val.complete_and();
    }

    // 4-way

    // g1 ⊕p1g2 ⊕p1p2g3 ⊕p1p2p3g4
    Share prepare_W4_S(Share a1, Share b1, Share g2, Share p2, Share g3, Share p3, Share g4)
    {
        /* return g1 ^ (p1 & g2) ^ p1.prepare_and3(p2, g3) ^ p1.prepare_and4(p2, p3, g4); */
        auto val = a1.prepare_dot(b1) ^ (a1 ^ b1).prepare_dot(g2) ^ (a1 ^ b1).prepare_dot3(p2, g3) ^
                   (a1 ^ b1).prepare_dot4(p2, p3, g4);
        val.mask_and_send_dot();
        return val;
    }
    Share prepare_W4(Share g1, Share p1, Share g2, Share p2, Share g3, Share p3, Share g4)
    {
        /* return g1 ^ (p1 & g2) ^ p1.prepare_and3(p2, g3) ^ p1.prepare_and4(p2, p3, g4); */
        auto val = p1.prepare_dot(g2) ^ p1.prepare_dot3(p2, g3) ^ p1.prepare_dot4(p2, p3, g4);
        val.mask_and_send_dot();
        val = val ^ g1;
        return val;
    }

    void complete_W4(Share& val)

    {
        /* val.complete_and(); */
        /* val.complete_and3(); */
        /* val.complete_and4(); */
        val.complete_and();
    }

    Share prepare_B4_G(Share g1, Share p1, Share g2, Share p2, Share g3, Share p3, Share g4)
    {
        return prepare_W4(g1, p1, g2, p2, g3, p3, g4);
    }

    void complete_B4_G(Share& val) { complete_W4(val); }

    Share prepare_B4_P(Share p1, Share p2, Share p3, Share p4) { return p1.prepare_and4(p2, p3, p4); }

    void complete_B4_P(Share& val) { val.complete_and4(); }

    // 3-way

    // g1 ⊕p1g2 ⊕p1p2g3
    Share prepare_W3_S(Share x1, Share y1, Share g2, Share p2, Share g3)
    {
        /* return g1 ^ (p1 & g2) ^ p1.prepare_and3(p2, g3); */
        auto val = x1.prepare_dot(y1) ^ (x1 ^ y1).prepare_dot(g2) ^ (x1 ^ y1).prepare_dot3(p2, g3);
        val.mask_and_send_dot();
        return val;
    }
    Share prepare_W3(Share g1, Share p1, Share g2, Share p2, Share g3)
    {
        /* return g1 ^ (p1 & g2) ^ p1.prepare_and3(p2, g3); */
        auto val = p1.prepare_dot(g2) ^ p1.prepare_dot3(p2, g3);
        val.mask_and_send_dot();
        val = val ^ g1;
        return val;
    }
    void complete_W3(Share& val)

    {
        /* val.complete_and(); */
        /* val.complete_and3(); */
        val.complete_and();
    }

    Share prepare_B3_G(Share g1, Share p1, Share g2, Share p2, Share g3) { return prepare_W3(g1, p1, g2, p2, g3); }

    void complete_B3_G(Share& val) { complete_W3(val); }

    Share prepare_B3_P(Share p1, Share p2, Share p3) { return p1.prepare_and3(p2, p3); }

    void complete_B3_P(Share& val) { val.complete_and3(); }

    // gi = ai · bi, pi = ai ⊕ bi
    // g1 ⊕p1g2 ⊕p1p2g3
    Share prepare_W3L1(Share x1, Share y1, Share x2, Share y2, Share x3, Share y3)
    {
        /* return (x1 & y1) ^ (x1^y1).prepare_and3(x2, y2) ^ (x1^y1).prepare_and4((x2^y2), x3, y3); */
        auto val = x1.prepare_dot(y1) ^ (x1 ^ y1).prepare_dot3(x2, y2) ^ (x1 ^ y1).prepare_dot4((x2 ^ y2), x3, y3);
        val.mask_and_send_dot();
        return val;
    }
    void complete_W3L1(Share& val)

    {
        /* val.complete_and(); */
        /* val.complete_and3(); */
        /* val.complete_and4(); */
        val.complete_and();
    }

    Share prepare_B3L1_G(Share x1, Share y1, Share x2, Share y2, Share x3, Share y3)
    {
        return prepare_W3L1(x1, y1, x2, y2, x3, y3);
    }

    void complete_B3L1_G(Share& val) { complete_W3L1(val); }

    // p1p2p3
    Share prepare_B3L1_P(Share x1, Share y1, Share x2, Share y2, Share x3, Share y3)
    {
        return (x1 ^ y1).prepare_and3((x2 ^ y2), (x3 ^ y3));
    }

    void complete_B3L1_P(Share& val) { val.complete_and3(); }

    // 2-way

    Share prepare_B2L1_P(Share x1, Share y1, Share x2, Share y2) { return (x1 ^ y1) & (x2 ^ y2); }

    void complete_B2L1_P(Share& val) { val.complete_and(); }

    // g1 ⊕p1g2
    Share prepare_B2L1_G(Share x1, Share y1, Share x2, Share y2)
    {
        /* return (x1 & y1) ^ (x1^y1).prepare_and3( x2, y2); */
        auto val = x1.prepare_dot(y1) ^ (x1 ^ y1).prepare_dot3(x2, y2);
        val.mask_and_send_dot();
        return val;
    }

    void complete_B2L1_G(Share& val)

    {
        /* val.complete_and(); */
        /* val.complete_and3(); */
        val.complete_and();
    }

    // L0: W3L1, 2 x B2 L1
    // L1: W3
    /* template<int m = k, typename std::enable_if<(m == 8), int>::type = 0> */
    /* void prepare_step() { */
    /* switch(level) { */
    /*     case 0: */
    /*         //reverse order of inputs */
    /*         v.push_back( prepare_B2L1_G(a[1], b[1], a[2], b[2])  ); */
    /*         v.push_back( prepare_B2L1_P(a[1], b[1], a[2], b[2])  ); */
    /*         v.push_back( prepare_B2L1_G(a[3], b[3], a[4], b[4])  ); */
    /*         v.push_back( prepare_B2L1_P(a[3], b[3], a[4], b[4])  ); */
    /*         v.push_back( prepare_W3L1(a[5], b[5], a[6], b[6], a[7], b[7])  ); */

    /* break; */
    /*     case 1: */
    /*         msb = (a[0] ^ b[0]) ^ prepare_W3(v[0], v[1], v[2], v[3], v[4]); */
    /* break; */
    /*     default: */
    /* break; */
    /* } */
    /* } */

    /* template<int m = k, typename std::enable_if<(m == 8), int>::type = 0> */
    /* void complete_Step() { */
    /* switch(level) { */
    /*     case 1: */
    /*         complete_B2L1_G(v[0]); */
    /*         complete_B2L1_P(v[1]); */
    /*         complete_B2L1_G(v[2]); */
    /*         complete_B2L1_P(v[3]); */
    /*         complete_W3L1(v[4]); */
    /* break; */
    /*     case 2: */
    /*         complete_W3(msb); */
    /* break; */
    /* default: */
    /* break; */
    /* } */
    /* } */
    template <int m = k, typename std::enable_if<(m == 8), int>::type = 0>
    void prepare_step()
    {
        switch (level)
        {
            case 0:
                v.push_back(prepare_B3L1_G(a[2], b[2], a[3], b[3], a[4], b[4]));
                v.push_back(prepare_B3L1_P(a[2], b[2], a[3], b[3], a[4], b[4]));
                v.push_back(prepare_W3L1(a[5], b[5], a[6], b[6], a[7], b[7]));

                break;
            case 1:
                msb = prepare_W3_S(a[1], b[1], v[0], v[1], v[2]);
                break;
            default:
                break;
        }
    }

    template <int m = k, typename std::enable_if<(m == 8), int>::type = 0>
    void complete_Step()
    {
        switch (level)
        {
            case 1:
                complete_B3L1_G(v[0]);
                complete_B3L1_P(v[1]);
                complete_W3(v[2]);
                break;
            case 2:
                complete_W4_3(msb);
                msb = msb ^ (a[0] ^ b[0]);
                break;
            default:
                break;
        }
    }

    /* template<int m = k, typename std::enable_if<(m == 8), int>::type = 0> */
    /* void prepare_step() { */
    /* switch(level) { */
    /*     case 0: */
    /*         //reverse order of inputs */
    /*         v.push_back( a[1] & b[1] ); // single g */
    /*         v.push_back( a[1] ^ b[1] ); // single p */
    /*         v.push_back( prepare_B2L1_G(a[2], b[2], a[3], b[3])  ); */
    /*         /1* v.push_back( prepare_B2L1_P(a[2], b[2], a[3], b[3])  );  // not needed since p1p2 is calculated *1/
     */
    /*         v.push_back( prepare_B2L1_G(a[4], b[4], a[5], b[5])  ); */
    /*         v.push_back( prepare_B2L1_P(a[4], b[4], a[5], b[5])  ); */
    /*         v.push_back( prepare_B2L1_G(a[6], b[6], a[7], b[7])  ); // white */
    /*         v.push_back( (a[1]^b[1]).prepare_and3((a[2]^b[2]), (a[3]^b[3])) ); //p12 */

    /* break; */
    /*     case 1: */
    /*         msb = prepare_W4_3(v[0], v[1], v[2], v[3], v[4], v[5], v[6]);   // g1, p1, g2, g3, p3, g4, p12 */
    /* break; */
    /*     default: */
    /* break; */
    /* } */
    /* } */

    /* template<int m = k, typename std::enable_if<(m == 8), int>::type = 0> */
    /* void complete_Step() { */
    /* switch(level) { */
    /*     case 1: */
    /*         v[0].complete_and(); */
    /*         complete_B2L1_G(v[2]); */
    /*         complete_B2L1_G(v[3]); */
    /*         complete_B2L1_P(v[4]); */
    /*         complete_B2L1_G(v[5]); */
    /*         v[6].complete_and3(); */
    /* break; */
    /*     case 2: */
    /*         complete_W4_3(msb); */
    /*         msb = (a[0] ^ b[0]) ^ msb; */
    /* break; */
    /* default: */
    /* break; */
    /* } */
    /* } */

    template <int m = k, typename std::enable_if<(m == 16), int>::type = 0>
    void prepare_step()
    {
        switch (level)
        {
            case 0:
                v.push_back(prepare_B3L1_G(a[1], b[1], a[2], b[2], a[3], b[3]));
                v.push_back(prepare_B3L1_P(a[1], b[1], a[2], b[2], a[3], b[3]));
                v.push_back(prepare_B3L1_G(a[4], b[4], a[5], b[5], a[6], b[6]));
                v.push_back(prepare_B3L1_P(a[4], b[4], a[5], b[5], a[6], b[6]));
                v.push_back(prepare_B3L1_G(a[7], b[7], a[8], b[8], a[9], b[9]));
                v.push_back(prepare_B3L1_P(a[7], b[7], a[8], b[8], a[9], b[9]));
                v.push_back(prepare_B3L1_G(a[10], b[10], a[11], b[11], a[12], b[12]));
                /* v.push_back( prepare_B3L1_P(a[10],b[10],a[11],b[11],a[12],b[12])  ); //TODO: remove since p1234 is
                 * used instead */
                v.push_back(prepare_W3L1(a[13], b[13], a[14], b[14], a[15], b[15]));
                v.push_back((a[1] ^ b[1]).prepare_and4(a[2] ^ b[2], a[3] ^ b[3], a[4] ^ b[4]));        // p1234_1
                v.push_back((a[5] ^ b[5]).prepare_and4(a[6] ^ b[6], a[7] ^ b[7], a[8] ^ b[8]));        // p1234_2
                v.push_back((a[9] ^ b[9]).prepare_and4(a[10] ^ b[10], a[11] ^ b[11], a[12] ^ b[12]));  // p1234_3
                // reverse input order

                break;
            case 1:
                msb = prepare_W5(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
                break;
            default:
                break;
        }
    }

    template <int m = k, typename std::enable_if<(m == 16), int>::type = 0>
    void complete_Step()
    {
        switch (level)
        {
            case 1:
                complete_B3L1_G(v[0]);
                complete_B3L1_P(v[1]);
                complete_B3L1_G(v[2]);
                complete_B3L1_P(v[3]);
                complete_B3L1_G(v[4]);
                complete_B3L1_P(v[5]);
                complete_B3L1_G(v[6]);
                /* complete_B3L1_P(v[7]); */
                complete_W3L1(v[7]);
                v[8].complete_and4();
                v[9].complete_and4();
                v[10].complete_and4();

                break;
            case 2:
                complete_W5(msb);
                msb = (a[0] ^ b[0]) ^ msb;
                break;
            default:
                break;
        }
    }

    template <int m = k, typename std::enable_if<(m == 32), int>::type = 0>
    void prepare_step()
    {
        switch (level)
        {
            case 0:
                // reverse input order above
                v.push_back(prepare_B3L1_G(a[1], b[1], a[2], b[2], a[3], b[3]));
                v.push_back(prepare_B3L1_P(a[1], b[1], a[2], b[2], a[3], b[3]));
                v.push_back(prepare_B3L1_G(a[4], b[4], a[5], b[5], a[6], b[6]));
                v.push_back(prepare_B3L1_P(a[4], b[4], a[5], b[5], a[6], b[6]));
                v.push_back(prepare_B3L1_G(a[7], b[7], a[8], b[8], a[9], b[9]));
                v.push_back(prepare_B3L1_P(a[7], b[7], a[8], b[8], a[9], b[9]));
                v.push_back(prepare_B3L1_G(a[10], b[10], a[11], b[11], a[12], b[12]));
                v.push_back(prepare_B3L1_P(a[10], b[10], a[11], b[11], a[12], b[12]));
                v.push_back(prepare_B3L1_G(a[13], b[13], a[14], b[14], a[15], b[15]));
                v.push_back(prepare_B3L1_P(a[13], b[13], a[14], b[14], a[15], b[15]));
                v.push_back(prepare_B3L1_G(a[16], b[16], a[17], b[17], a[18], b[18]));
                v.push_back(prepare_B3L1_P(a[16], b[16], a[17], b[17], a[18], b[18]));
                v.push_back(prepare_B3L1_G(a[19], b[19], a[20], b[20], a[21], b[21]));
                v.push_back(prepare_B3L1_P(a[19], b[19], a[20], b[20], a[21], b[21]));
                v.push_back(prepare_B3L1_G(a[23], b[23], a[24], b[24], a[25], b[25]));
                v.push_back(prepare_B3L1_P(a[23], b[23], a[24], b[24], a[25], b[25]));
                v.push_back(prepare_B3L1_G(a[26], b[26], a[27], b[27], a[28], b[28]));
                v.push_back(prepare_B3L1_P(a[26], b[26], a[27], b[27], a[28], b[28]));
                v.push_back(prepare_W3L1(a[29], b[29], a[30], b[30], a[31], b[31]));

                break;
            case 1:
                v.push_back(prepare_B3_G(v[0], v[1], v[2], v[3], v[4]));
                v.push_back(prepare_B3_P(v[1], v[3], v[5]));
                v.push_back(prepare_B4_G(v[6], v[7], v[8], v[9], v[10], v[11], v[12]));
                v.push_back(prepare_B4_P(v[7], v[9], v[11], v[13]));
                v.push_back(prepare_W4_S(a[22], b[22], v[14], v[15], v[16], v[17], v[18]));
                break;
            case 2:
                msb = prepare_W3(v[19], v[20], v[21], v[22], v[23]);
                break;
            default:
                break;
        }
    }

    template <int m = k, typename std::enable_if<(m == 32), int>::type = 0>
    void complete_Step()
    {
        switch (level)
        {
            case 1:
                complete_B3L1_G(v[0]);
                complete_B3L1_P(v[1]);
                complete_B3L1_G(v[2]);
                complete_B3L1_P(v[3]);
                complete_B3L1_G(v[4]);
                complete_B3L1_P(v[5]);
                complete_B3L1_G(v[6]);
                complete_B3L1_P(v[7]);
                complete_B3L1_G(v[8]);
                complete_B3L1_P(v[9]);
                complete_B3L1_G(v[10]);
                complete_B3L1_P(v[11]);
                complete_B3L1_G(v[12]);
                complete_B3L1_P(v[13]);
                complete_B3L1_G(v[14]);
                complete_B3L1_P(v[15]);
                complete_B3L1_G(v[16]);
                complete_B3L1_P(v[17]);
                complete_W3L1(v[18]);
                break;
            case 2:
                complete_B3_G(v[19]);
                complete_B3_P(v[20]);
                complete_B4_G(v[21]);
                complete_B4_P(v[22]);
                complete_W4(v[23]);
                break;
            case 3:
                complete_W3(msb);
                msb = msb ^ a[0] ^ b[0];
                break;

            default:
                break;
        }
    }

    template <int m = k, typename std::enable_if<(m == 64), int>::type = 0>
    void prepare_step()
    {
        switch (level)
        {
            case 0:
            {
                for (int i = 0; i < 32; i++)
                {
                    std::swap(a[i], a[63 - i]);
                    std::swap(b[i], b[63 - i]);
                }
                v.push_back(a[2].prepare_dot(b[2]) ^ (a[2] ^ b[2]).prepare_dot3(a[1], b[1]) ^
                            (a[2] ^ b[2]).prepare_dot4((a[1] ^ b[1]), a[0], b[0]));  // G2
                v[0].mask_and_send_dot();
                v.push_back(a[6].prepare_dot(b[6]) ^ (a[6] ^ b[6]).prepare_dot3(a[5], b[5]) ^
                            (a[6] ^ b[6]).prepare_dot4((a[5] ^ b[5]), a[4], b[4]));  // G6
                v[1].mask_and_send_dot();
                v.push_back(a[10].prepare_dot(b[10]) ^ (a[10] ^ b[10]).prepare_dot3(a[9], b[9]) ^
                            (a[10] ^ b[10]).prepare_dot4((a[9] ^ b[9]), a[8], b[8]));  // G10
                v[2].mask_and_send_dot();
                v.push_back(a[14].prepare_dot(b[14]) ^ (a[14] ^ b[14]).prepare_dot3(a[13], b[13]) ^
                            (a[14] ^ b[14]).prepare_dot4((a[13] ^ b[13]), a[12], b[12]));  // G14
                v[3].mask_and_send_dot();
                v.push_back(a[18].prepare_dot(b[18]) ^ (a[18] ^ b[18]).prepare_dot3(a[17], b[17]) ^
                            (a[18] ^ b[18]).prepare_dot4((a[17] ^ b[17]), a[16], b[16]));  // G18
                v[4].mask_and_send_dot();
                v.push_back(a[22].prepare_dot(b[22]) ^ (a[22] ^ b[22]).prepare_dot3(a[21], b[21]) ^
                            (a[22] ^ b[22]).prepare_dot4((a[21] ^ b[21]), a[20], b[20]));  // G22
                v[5].mask_and_send_dot();
                v.push_back(a[26].prepare_dot(b[26]) ^ (a[26] ^ b[26]).prepare_dot3(a[25], b[25]) ^
                            (a[26] ^ b[26]).prepare_dot4((a[25] ^ b[25]), a[24], b[24]));  // G26
                v[6].mask_and_send_dot();
                v.push_back(a[30].prepare_dot(b[30]) ^ (a[30] ^ b[30]).prepare_dot3(a[29], b[29]) ^
                            (a[30] ^ b[30]).prepare_dot4((a[29] ^ b[29]), a[28], b[28]));  // G30
                v[7].mask_and_send_dot();
                v.push_back(a[34].prepare_dot(b[34]) ^ (a[34] ^ b[34]).prepare_dot3(a[33], b[33]) ^
                            (a[34] ^ b[34]).prepare_dot4((a[33] ^ b[33]), a[32], b[32]));  // G34
                v[8].mask_and_send_dot();
                v.push_back(a[38].prepare_dot(b[38]) ^ (a[38] ^ b[38]).prepare_dot3(a[37], b[37]) ^
                            (a[38] ^ b[38]).prepare_dot4((a[37] ^ b[37]), a[36], b[36]));  // G38
                v[9].mask_and_send_dot();
                v.push_back(a[42].prepare_dot(b[42]) ^ (a[42] ^ b[42]).prepare_dot3(a[41], b[41]) ^
                            (a[42] ^ b[42]).prepare_dot4((a[41] ^ b[41]), a[40], b[40]));  // G42
                v[10].mask_and_send_dot();
                v.push_back(a[46].prepare_dot(b[46]) ^ (a[46] ^ b[46]).prepare_dot3(a[45], b[45]) ^
                            (a[46] ^ b[46]).prepare_dot4((a[45] ^ b[45]), a[44], b[44]));  // G46
                v[11].mask_and_send_dot();
                v.push_back(a[50].prepare_dot(b[50]) ^ (a[50] ^ b[50]).prepare_dot3(a[49], b[49]) ^
                            (a[50] ^ b[50]).prepare_dot4((a[49] ^ b[49]), a[48], b[48]));  // G50
                v[12].mask_and_send_dot();
                v.push_back(a[54].prepare_dot(b[54]) ^ (a[54] ^ b[54]).prepare_dot3(a[53], b[53]) ^
                            (a[54] ^ b[54]).prepare_dot4((a[53] ^ b[53]), a[52], b[52]));  // G54
                v[13].mask_and_send_dot();
                v.push_back(a[58].prepare_dot(b[58]) ^ (a[58] ^ b[58]).prepare_dot3(a[57], b[57]) ^
                            (a[58] ^ b[58]).prepare_dot4((a[57] ^ b[57]), a[56], b[56]));  // G58
                v[14].mask_and_send_dot();
                v.push_back(a[62].prepare_dot(b[62]) ^ (a[62] ^ b[62]).prepare_dot3(a[61], b[61]) ^
                            (a[62] ^ b[62]).prepare_dot4((a[61] ^ b[61]), a[60], b[60]));  // G62
                v[15].mask_and_send_dot();
                v.push_back((a[7] ^ b[7]).prepare_dot4((a[6] ^ b[6]), (a[5] ^ b[5]), (a[4] ^ b[4])));  // P7
                v[16].mask_and_send_dot();
                v.push_back((a[11] ^ b[11]).prepare_dot4((a[10] ^ b[10]), (a[9] ^ b[9]), (a[8] ^ b[8])));  // P11
                v[17].mask_and_send_dot();
                v.push_back((a[15] ^ b[15]).prepare_dot4((a[14] ^ b[14]), (a[13] ^ b[13]), (a[12] ^ b[12])));  // P15
                v[18].mask_and_send_dot();
                v.push_back((a[19] ^ b[19]).prepare_dot4((a[18] ^ b[18]), (a[17] ^ b[17]), (a[16] ^ b[16])));  // P19
                v[19].mask_and_send_dot();
                v.push_back((a[23] ^ b[23]).prepare_dot4((a[22] ^ b[22]), (a[21] ^ b[21]), (a[20] ^ b[20])));  // P23
                v[20].mask_and_send_dot();
                v.push_back((a[27] ^ b[27]).prepare_dot4((a[26] ^ b[26]), (a[25] ^ b[25]), (a[24] ^ b[24])));  // P27
                v[21].mask_and_send_dot();
                v.push_back((a[31] ^ b[31]).prepare_dot4((a[30] ^ b[30]), (a[29] ^ b[29]), (a[28] ^ b[28])));  // P31
                v[22].mask_and_send_dot();
                v.push_back((a[35] ^ b[35]).prepare_dot4((a[34] ^ b[34]), (a[33] ^ b[33]), (a[32] ^ b[32])));  // P35
                v[23].mask_and_send_dot();
                v.push_back((a[39] ^ b[39]).prepare_dot4((a[38] ^ b[38]), (a[37] ^ b[37]), (a[36] ^ b[36])));  // P39
                v[24].mask_and_send_dot();
                v.push_back((a[43] ^ b[43]).prepare_dot4((a[42] ^ b[42]), (a[41] ^ b[41]), (a[40] ^ b[40])));  // P43
                v[25].mask_and_send_dot();
                v.push_back((a[47] ^ b[47]).prepare_dot4((a[46] ^ b[46]), (a[45] ^ b[45]), (a[44] ^ b[44])));  // P47
                v[26].mask_and_send_dot();
                v.push_back((a[51] ^ b[51]).prepare_dot4((a[50] ^ b[50]), (a[49] ^ b[49]), (a[48] ^ b[48])));  // P51
                v[27].mask_and_send_dot();
                v.push_back((a[55] ^ b[55]).prepare_dot4((a[54] ^ b[54]), (a[53] ^ b[53]), (a[52] ^ b[52])));  // P55
                v[28].mask_and_send_dot();
                v.push_back((a[59] ^ b[59]).prepare_dot4((a[58] ^ b[58]), (a[57] ^ b[57]), (a[56] ^ b[56])));  // P59
                v[29].mask_and_send_dot();
                v.push_back((a[62] ^ b[62]).prepare_dot3((a[61] ^ b[61]), (a[60] ^ b[60])));  // P62
                v[30].mask_and_send_dot();
                v.push_back((a[62] ^ b[62]).prepare_dot4((a[61] ^ b[61]), (a[60] ^ b[60]), (a[15] ^ b[15])));  // P62_15
                v[31].mask_and_send_dot();
                v.push_back((a[62] ^ b[62]).prepare_dot4((a[61] ^ b[61]), (a[60] ^ b[60]), (a[3] ^ b[3])));  // P62_3
                v[32].mask_and_send_dot();
                v.push_back(a[3].prepare_dot(b[3]));  // g3
                v[33].mask_and_send_dot();
                v.push_back(a[15].prepare_dot(b[15]));  // g15
                v[34].mask_and_send_dot();
                v.push_back(a[19].prepare_dot(b[19]));  // g19
                v[35].mask_and_send_dot();
                break;
            }
            case 1:
            {
                auto G2 = v[0];
                auto G6 = v[1];
                auto G10 = v[2];
                auto G14 = v[3];
                auto G18 = v[4];
                auto G22 = v[5];
                auto G26 = v[6];
                auto G30 = v[7];
                auto G34 = v[8];
                auto G38 = v[9];
                auto G42 = v[10];
                auto G46 = v[11];
                auto G50 = v[12];
                auto G54 = v[13];
                auto G58 = v[14];
                auto G62 = v[15];
                auto P7 = v[16];
                auto P11 = v[17];
                auto P15 = v[18];
                auto P19 = v[19];
                auto P23 = v[20];
                auto P27 = v[21];
                auto P31 = v[22];
                auto P35 = v[23];
                auto P39 = v[24];
                auto P43 = v[25];
                auto P47 = v[26];
                auto P51 = v[27];
                auto P55 = v[28];
                auto P59 = v[29];
                auto P62 = v[30];
                auto P62_15 = v[31];
                auto P62_3 = v[32];
                auto g3 = v[33];
                auto g15 = v[34];
                auto g19 = v[35];

                v.push_back(a[27].prepare_dot(b[27]) ^ (a[27] ^ b[27]).prepare_dot(G26));  // G27
                v[36].mask_and_send_dot();
                v.push_back(a[39].prepare_dot(b[39]) ^ (a[39] ^ b[39]).prepare_dot(G38));  // G39
                v[37].mask_and_send_dot();
                v.push_back(a[43].prepare_dot(b[43]) ^ (a[43] ^ b[43]).prepare_dot(G42));  // G43
                v[38].mask_and_send_dot();
                v.push_back(a[55].prepare_dot(b[55]) ^ (a[55] ^ b[55]).prepare_dot(G54));  // G55
                v[39].mask_and_send_dot();
                v.push_back(a[59].prepare_dot(b[59]) ^ (a[59] ^ b[59]).prepare_dot(G58));  // G59
                v[40].mask_and_send_dot();
                v.push_back(P27.prepare_dot4(P23, P19, P15));  // P27_2
                v[41].mask_and_send_dot();
                v.push_back(P43.prepare_dot4(P39, P35, P31));  // P43_2
                v[42].mask_and_send_dot();
                v.push_back(P59.prepare_dot4(P55, P51, P47));  // P59_2

                v[43].mask_and_send_dot();
                v.push_back(P39.prepare_dot3(a[35], b[35]) ^ P39.prepare_dot3((a[35] ^ b[35]), G34) ^
                            P39.prepare_dot4(P35, a[31], b[31]) ^
                            P39.prepare_dot4(P35, (a[31] ^ b[31]), G30));  // T1, need to add G39 when complete
                v[44].mask_and_send_dot();
                v.push_back(P55.prepare_dot3(a[51], b[51]) ^ P55.prepare_dot3((a[51] ^ b[51]), G50) ^
                            P55.prepare_dot4(P51, a[47], b[47]) ^
                            P55.prepare_dot4(P51, (a[47] ^ b[47]), G46));  // T2, need to add G55 when complete
                v[45].mask_and_send_dot();
                v.push_back(P62.prepare_dot3(a[11], b[11]) ^ P62.prepare_dot3((a[11] ^ b[11]), G10) ^
                            P62.prepare_dot4(P11, a[7], b[7]) ^ P62.prepare_dot4(P11, (a[7] ^ b[7]), G6) ^
                            P62.prepare_dot4(P11, P7, g3) ^
                            P62_3.prepare_dot4(P11, P7, G2));  // T3, need to prepare g[3]
                v[46].mask_and_send_dot();
                v.push_back(
                    P62.prepare_dot3(a[23], b[23]) ^ P62.prepare_dot3((a[23] ^ b[23]), G22) ^
                    P62.prepare_dot4(P23, a[19], b[19]) ^ P62.prepare_dot4(P23, g19, G18) ^
                    P62.prepare_dot4(P23, P19, g15) ^
                    P62_15.prepare_dot4(P23, P19, G14));  // T4, need to prepare g[19], verify if 2nd G19 should be P19
                v[47].mask_and_send_dot();
                break;
            }
            case 2:
            {
                auto P27 = v[21];
                auto P43 = v[25];
                auto P59 = v[29];
                auto P62 = v[30];
                auto G62 = v[15];
                auto G27 = v[36];
                auto G39 = v[37];
                auto G43 = v[38];
                auto G55 = v[39];
                auto G59 = v[40];
                auto P27_2 = v[41];
                auto P43_2 = v[42];
                auto P59_2 = v[43];
                auto T1 = v[44];
                auto T2 = v[45];
                auto T3 = v[46];
                auto T4 = v[47];
                msb =
                    P62.prepare_dot(G59) ^ P62.prepare_dot3(P59, T2) ^ P62.prepare_dot3(P59_2, G43) ^
                    P62.prepare_dot4(P59_2, P43, T1) ^ T4.prepare_dot4(P59_2, P43_2, P27) ^
                    P59_2.prepare_dot4(P43_2, P27_2, T3);  // T4, need to prepare g[19], verify if 2nd G19 should be P19
                msb.mask_and_send_dot();
                break;
            }
            default:
                break;
        }
    }
    template <int m = k, typename std::enable_if<(m == 64), int>::type = 0>
    void complete_Step()
    {
        switch (level)
        {
            case 1:
                for (int i = 0; i < 36; i++)
                    v[i].complete_and();
                break;
            case 2:
                for (int i = 36; i < 48; i++)
                    v[i].complete_and();
                // add G39 and G55 to T1 and T2
                v[44] = v[44] ^ v[37];
                v[45] = v[45] ^ v[39];
                break;
            case 3:
                msb.complete_and();
                msb = msb ^ a[63] ^ b[63] ^ v[15];  // add G62
                break;

            default:
                break;
        }
    }

    /* template<int m = k, typename std::enable_if<(m == 32), int>::type = 0> */
    /* void prepare_step() { */
    /* switch(level) { */
    /*     case 0: */
    /*         //reverse input order above */
    /*         v.push_back( prepare_B3L1_G(a[1], b[1], a[2], b[2], a[3], b[3])  ); */
    /*         v.push_back( prepare_B3L1_P(a[1], b[1], a[2], b[2], a[3], b[3])  ); */
    /*         v.push_back( prepare_B3L1_G(a[4], b[4], a[5], b[5], a[6], b[6])  ); */
    /*         v.push_back( prepare_B3L1_P(a[4], b[4], a[5], b[5], a[6], b[6])  ); */
    /*         v.push_back( prepare_B3L1_G(a[7], b[7], a[8], b[8], a[9], b[9])  ); */
    /*         v.push_back( prepare_B3L1_P(a[7], b[7], a[8], b[8], a[9], b[9])  ); */
    /*         v.push_back( prepare_B3L1_G(a[10], b[10], a[11], b[11], a[12], b[12])  ); */
    /*         v.push_back( prepare_B3L1_P(a[10], b[10], a[11], b[11], a[12], b[12])  ); */
    /*         v.push_back( prepare_B3L1_G(a[13], b[13], a[14], b[14], a[15], b[15])  ); */
    /*         v.push_back( prepare_B3L1_P(a[13], b[13], a[14], b[14], a[15], b[15])  ); */
    /*         v.push_back( prepare_B3L1_G(a[16], b[16], a[17], b[17], a[18], b[18])  ); */
    /*         v.push_back( prepare_B3L1_P(a[16], b[16], a[17], b[17], a[18], b[18])  ); */
    /*         v.push_back( prepare_B3L1_G(a[19], b[19], a[20], b[20], a[21], b[21])  ); */
    /*         v.push_back( prepare_B3L1_P(a[19], b[19], a[20], b[20], a[21], b[21])  ); */
    /*         v.push_back( prepare_B3L1_G(a[22], b[22], a[23], b[23], a[24], b[24])  ); */
    /*         v.push_back( prepare_B3L1_P(a[22], b[22], a[23], b[23], a[24], b[24])  ); */
    /*         v.push_back( prepare_B3L1_G(a[25], b[25], a[26], b[26], a[27], b[27])  ); */
    /*         v.push_back( prepare_B3L1_P(a[25], b[25], a[26], b[26], a[27], b[27])  ); */
    /*         v.push_back( a[28] & b[28]  ); //single G */
    /*         v.push_back( a[28] ^ b[28]  );  //single P */
    /*         v.push_back( prepare_W3L1(a[29], b[29], a[30], b[30], a[31], b[31])  ); */

    /*         break; */
    /*     case 1: */
    /*         v.push_back( prepare_B3_G(v[0], v[1], v[2], v[3], v[4])  ); */
    /*         v.push_back( prepare_B3_P(v[1], v[3], v[5])  ); */
    /*         v.push_back( prepare_B4_G(v[6], v[7], v[8], v[9], v[10], v[11], v[12])  ); */
    /*         v.push_back( prepare_B4_P(v[7], v[9], v[11], v[13])  ); */
    /*         v.push_back( prepare_W4(v[14], v[15], v[16], v[17], v[18], v[19], v[20])  ); */
    /* break; */
    /*     case 2: */
    /*         msb = (a[0] ^ b[0]) ^ prepare_W3(v[21], v[22], v[23], v[24], v[25]); */
    /* break; */
    /*     default: */
    /* break; */
    /* } */
    /* } */

    /* template<int m = k, typename std::enable_if<(m == 32), int>::type = 0> */
    /* void complete_Step() { */
    /* switch(level) { */
    /*     case 1: */
    /*         complete_B3L1_G(v[0]); */
    /*         complete_B3L1_P(v[1]); */
    /*         complete_B3L1_G(v[2]); */
    /*         complete_B3L1_P(v[3]); */
    /*         complete_B3L1_G(v[4]); */
    /*         complete_B3L1_P(v[5]); */
    /*         complete_B3L1_G(v[6]); */
    /*         complete_B3L1_P(v[7]); */
    /*         complete_B3L1_G(v[8]); */
    /*         complete_B3L1_P(v[9]); */
    /*         complete_B3L1_G(v[10]); */
    /*         complete_B3L1_P(v[11]); */
    /*         complete_B3L1_G(v[12]); */
    /*         complete_B3L1_P(v[13]); */
    /*         complete_B3L1_G(v[14]); */
    /*         complete_B3L1_P(v[15]); */
    /*         complete_B3L1_G(v[16]); */
    /*         complete_B3L1_P(v[17]); */
    /*         v[18].complete_and(); */
    /*         //skip v[19] because no mult */
    /*         complete_W3L1(v[20]); */
    /* break; */
    /*     case 2: */
    /*         complete_B3_G(v[21]); */
    /*         complete_B3_P(v[22]); */
    /*         complete_B4_G(v[23]); */
    /*         complete_B4_P(v[24]); */
    /*         complete_W4(v[25]); */
    /* break; */
    /*     case 3: */
    /*         complete_W3(msb); */
    /* default: */
    /* break; */
    /* } */
    /* } */

    void step()
    {
        switch (level)
        {
            case 0:
                prepare_step();
                level++;
                break;
            case 1:
                complete_Step();
                prepare_step();
                level++;
                break;
            case 2:
                complete_Step();
                if constexpr (k > 16)
                {
                    prepare_step();
                }
                level++;
                break;
            case 3:
                complete_Step();
                level++;
                break;

            default:
                break;
        }
    }

    PPA_MSB_4Way(Bitset& x0, Bitset& x1, Share& y0) : a(x0), b(x1), msb(y0)
    {
        level = 0;
        if constexpr (k == 8)
            v.reserve(3);  // 4 * online phase
        else if constexpr (k == 16)
            v.reserve(11);  // 12 * online phase
        else if constexpr (k == 32)
            v.reserve(24);  // 25 * online phase
        else if constexpr (k == 64)
            v.reserve(48);  // 49 * online phase
    }

    int get_rounds() { return level; }

    int get_total_rounds()
    {
        switch (k)
        {
            case 8:
                return 2;
            case 16:
                return 2;
            case 32:
                return 3;
            case 64:
                return 3;
            default:
                return 0;
        }
    }

    bool is_done() { return level == get_total_rounds() + 1; }
};
