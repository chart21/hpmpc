#pragma once
#include "../../protocols/Protocols.h"
#include "../../datatypes/k_bitset.hpp"
#include <cstring>
#include <iostream>
#include <vector>

template<int k, typename Share>
class PPA_MSB_4Way{
    using Bitset = sbitset_t<Share>;
private:
    Bitset &a;
    Bitset &b;
    Share &msb;
    int level;
    std::vector<Share> v;
public:
//constructor




// g1 ⊕p1g2 ⊕p1p2g3 ⊕p1p2p3g4 ⊕p1p2p3p4g5
// 5way
Share prepare_W5(Share g1, Share p1, Share g2, Share p2, Share g3, Share p3, Share g4, Share g5, Share p1234_1, Share p1234_2, Share p1234_3)
{
   return g1 ^ (p1 & g2) ^ p1.prepare_and3(p2, g3) ^ p1.prepare_and4(p2, p3, g4) ^ p1234_1.prepare_and4(p1234_2, p1234_3, g5);
}

void complete_W5(Share &val)

{
   val.complete_and();
   val.complete_and3();
   val.complete_and4();
   val.complete_and4();
}

// 4-way

// g1 ⊕p1g2 ⊕p1p2g3 ⊕p1p2p3g4
Share prepare_W4(Share g1, Share p1, Share g2, Share p2, Share g3, Share p3, Share g4)
{
   return g1 ^ (p1 & g2) ^ p1.prepare_and3(p2, g3) ^ p1.prepare_and4(p2, p3, g4);
}

void complete_W4(Share &val)

{
   val.complete_and();
   val.complete_and3();
   val.complete_and4();
}

Share prepare_B4_G(Share g1, Share p1, Share g2, Share p2, Share g3, Share p3, Share g4)
{
    return prepare_W4(g1, p1, g2, p2, g3, p3, g4);
}

void complete_B4_G(Share &val)
{
    complete_W4(val);
}

Share prepare_B4_P(Share p1, Share p2, Share p3, Share p4)
{
    return p1.prepare_and4(p2, p3, p4);
}

void complete_B4_P(Share &val)
{
    val.complete_and4();
}


// 3-way


// g1 ⊕p1g2 ⊕p1p2g3
Share prepare_W3(Share g1, Share p1, Share g2, Share p2, Share g3)
{
   return g1 ^ (p1 & g2) ^ p1.prepare_and3(p2, g3);
}
void complete_W3(Share &val)

{
   val.complete_and();
   val.complete_and3();
}

Share prepare_B3_G(Share g1, Share p1, Share g2, Share p2, Share g3)
{
    return prepare_W3(g1, p1, g2, p2, g3);
}

void complete_B3_G(Share &val)
{
    complete_W3(val);
}

Share prepare_B3_P(Share p1, Share p2, Share p3)
{
    return p1.prepare_and3(p2, p3);
}

void complete_B3_P(Share &val)
{
    val.complete_and3();
}




// gi = ai · bi, pi = ai ⊕ bi
// g1 ⊕p1g2 ⊕p1p2g3
Share prepare_W3L1(Share x1, Share y1, Share x2, Share y2, Share x3, Share y3)
{
   return (x1 & y1) ^ (x1^y1).prepare_and3(x2, y2) ^ (x1^y1).prepare_and4((x2^y2), x3, y3);
}
void complete_W3L1(Share &val)

{
   val.complete_and();
   val.complete_and3();
   val.complete_and4();
}

Share prepare_B3L1_G(Share x1, Share y1, Share x2, Share y2, Share x3, Share y3)
{
    return prepare_W3L1(x1, y1, x2, y2, x3, y3);
}

void complete_B3L1_G(Share &val)

{
    complete_W3L1(val);
}

// p1p2p3
Share prepare_B3L1_P(Share x1, Share y1, Share x2, Share y2, Share x3, Share y3)
{
    return (x1 ^ y1).prepare_and3((x2 ^ y2), (x3 ^ y3));
}

void complete_B3L1_P(Share &val)
{
    val.complete_and3();
}



// 2-way

Share prepare_B2L1_P(Share x1, Share y1, Share x2, Share y2)
{
    return (x1 ^ y1) & (x2 ^ y2);
}

void complete_B2L1_P(Share &val)

{
   val.complete_and();
}

// g1 ⊕p1g2 
Share prepare_B2L1_G(Share x1, Share y1, Share x2, Share y2)
{
    return (x1 & y1) ^ (x1^y1).prepare_and3( x2, y2);
}

void complete_B2L1_G(Share &val)

{
   val.complete_and();
   val.complete_and3();
}


//L0: W3L1, 2 x B2 L1
//L1: W3
template<int m = k, typename std::enable_if<(m == 8), int>::type = 0>
void prepare_step() {
switch(level) {
    case 0:
        //reverse order of inputs
        v.push_back( prepare_B2L1_G(a[1], b[1], a[2], b[2])  );
        v.push_back( prepare_B2L1_P(a[1], b[1], a[2], b[2])  );
        v.push_back( prepare_B2L1_G(a[3], b[3], a[4], b[4])  );
        v.push_back( prepare_B2L1_P(a[3], b[3], a[4], b[4])  );
        v.push_back( prepare_W3L1(a[5], b[5], a[6], b[6], a[7], b[7])  );

break;
    case 1:
        msb = (a[0] ^ b[0]) ^ prepare_W3(v[0], v[1], v[2], v[3], v[4]);
break;
    default:
break;
}
}

template<int m = k, typename std::enable_if<(m == 8), int>::type = 0>
void complete_Step() {
switch(level) {
    case 1:
        complete_B2L1_G(v[0]);
        complete_B2L1_P(v[1]);
        complete_B2L1_G(v[2]);
        complete_B2L1_P(v[3]);
        complete_W3L1(v[4]);
break;
    case 2:
        complete_W3(msb);
break;
default:
break;
}
}

template<int m = k, typename std::enable_if<(m == 16), int>::type = 0>
void prepare_step() {
switch(level) {
    case 0:
        v.push_back( prepare_B3L1_G(a[1],b[1],a[2],b[2],a[3],b[3])  );
        v.push_back( prepare_B3L1_P(a[1],b[1],a[2],b[2],a[3],b[3])  );
        v.push_back( prepare_B3L1_G(a[4],b[4],a[5],b[5],a[6],b[6])  );
        v.push_back( prepare_B3L1_P(a[4],b[4],a[5],b[5],a[6],b[6])  );
        v.push_back( prepare_B3L1_G(a[7],b[7],a[8],b[8],a[9],b[9])  );
        v.push_back( prepare_B3L1_P(a[7],b[7],a[8],b[8],a[9],b[9])  );
        v.push_back( prepare_B3L1_G(a[10],b[10],a[11],b[11],a[12],b[12])  );
        v.push_back( prepare_B3L1_P(a[10],b[10],a[11],b[11],a[12],b[12])  ); //TODO: remove since p1234 is used instead 
        v.push_back( prepare_W3L1(a[13],b[13],a[14],b[14],a[15],b[15])  );
        v.push_back( (a[1] ^ b[1]).prepare_and4(a[2] ^ b[2], a[3] ^ b[3], a[4] ^ b[4])); //p1234_1
        v.push_back( (a[5] ^ b[5]).prepare_and4(a[6] ^ b[6], a[7] ^ b[7], a[8] ^ b[8])); //p1234_2
        v.push_back( (a[9] ^ b[9]).prepare_and4(a[10] ^ b[10], a[11] ^ b[11], a[12] ^ b[12])); //p1234_3
        //reverse input order

break;
    case 1:
        msb = (a[0] ^ b[0]) ^ prepare_W5(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[8], v[9], v[10], v[11]);
break;
    default:
break;
}
}

template<int m = k, typename std::enable_if<(m == 16), int>::type = 0>
void complete_Step() {
switch(level) {
    case 1:
        complete_B3L1_G(v[0]);
        complete_B3L1_P(v[1]);
        complete_B3L1_G(v[2]);
        complete_B3L1_P(v[3]);
        complete_B3L1_G(v[4]);
        complete_B3L1_P(v[5]);
        complete_B3L1_G(v[6]);
        complete_B3L1_P(v[7]);
        complete_W3L1(v[8]);
        v[9].complete_and4();
        v[10].complete_and4();
        v[11].complete_and4();

break;
    case 2:
        complete_W5(msb);
break;
default:
break;
}
}

template<int m = k, typename std::enable_if<(m == 32), int>::type = 0>
void prepare_step() {
switch(level) {
    case 0:
        //reverse input order above
        v.push_back( prepare_B3L1_G(a[1], b[1], a[2], b[2], a[3], b[3])  );
        v.push_back( prepare_B3L1_P(a[1], b[1], a[2], b[2], a[3], b[3])  );
        v.push_back( prepare_B3L1_G(a[4], b[4], a[5], b[5], a[6], b[6])  );
        v.push_back( prepare_B3L1_P(a[4], b[4], a[5], b[5], a[6], b[6])  );
        v.push_back( prepare_B3L1_G(a[7], b[7], a[8], b[8], a[9], b[9])  );
        v.push_back( prepare_B3L1_P(a[7], b[7], a[8], b[8], a[9], b[9])  );
        v.push_back( prepare_B3L1_G(a[10], b[10], a[11], b[11], a[12], b[12])  );
        v.push_back( prepare_B3L1_P(a[10], b[10], a[11], b[11], a[12], b[12])  );
        v.push_back( prepare_B3L1_G(a[13], b[13], a[14], b[14], a[15], b[15])  );
        v.push_back( prepare_B3L1_P(a[13], b[13], a[14], b[14], a[15], b[15])  );
        v.push_back( prepare_B3L1_G(a[16], b[16], a[17], b[17], a[18], b[18])  );
        v.push_back( prepare_B3L1_P(a[16], b[16], a[17], b[17], a[18], b[18])  );
        v.push_back( prepare_B3L1_G(a[19], b[19], a[20], b[20], a[21], b[21])  );
        v.push_back( prepare_B3L1_P(a[19], b[19], a[20], b[20], a[21], b[21])  );
        v.push_back( prepare_B3L1_G(a[22], b[22], a[23], b[23], a[24], b[24])  );
        v.push_back( prepare_B3L1_P(a[22], b[22], a[23], b[23], a[24], b[24])  );
        v.push_back( prepare_B3L1_G(a[25], b[25], a[26], b[26], a[27], b[27])  );
        v.push_back( prepare_B3L1_P(a[25], b[25], a[26], b[26], a[27], b[27])  );
        v.push_back( a[28] & b[28]  ); //single G
        v.push_back( a[28] ^ b[28]  );  //single P
        v.push_back( prepare_W3L1(a[29], b[29], a[30], b[30], a[31], b[31])  );

        break;
    case 1:
        v.push_back( prepare_B3_G(v[0], v[1], v[2], v[3], v[4])  );
        v.push_back( prepare_B3_P(v[1], v[3], v[5])  );
        v.push_back( prepare_B4_G(v[6], v[7], v[8], v[9], v[10], v[11], v[12])  );
        v.push_back( prepare_B4_P(v[7], v[9], v[11], v[13])  );
        v.push_back( prepare_W4(v[14], v[15], v[16], v[17], v[18], v[19], v[20])  );
break;
    case 2:
        msb = (a[0] ^ b[0]) ^ prepare_W3(v[21], v[22], v[23], v[24], v[25]);
break;
    default:
break;
}
}

template<int m = k, typename std::enable_if<(m == 32), int>::type = 0>
void complete_Step() {
switch(level) {
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
        v[18].complete_and();
        //skip v[19] because no mult
        complete_W3L1(v[20]);
break;
    case 2:
        complete_B3_G(v[21]);
        complete_B3_P(v[22]);
        complete_B4_G(v[23]);
        complete_B4_P(v[24]);
        complete_W4(v[25]);
break;
    case 3:
        complete_W3(msb);
default:
break;
}
}


void step() {
switch(level) {
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
       if constexpr (k > 16) {
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

PPA_MSB_4Way(Bitset &x0, Bitset &x1, Share &y0) : a(x0), b(x1), msb(y0) 
    {
        level = 0;
    }




int get_rounds() {
    return level;
}

int get_total_rounds() {
    return LOG4_BITLENGTH;
}

bool is_done() {
    return level == (LOG4_BITLENGTH+1);
}

};
