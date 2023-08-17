#pragma once
#include "../../protocols/Protocols.h"
#include "../../datatypes/k_bitset.hpp"
#include <cstring>
#include <iostream>

template<typename Share>
class BooleanAdder {
    using Bitset = sbitset_t<Share>;
private:
    int r;
    Bitset &x0__;
    Bitset &x1__;
    Bitset &y0__;
    Share temp__[312];
   
public:
//constructor

BooleanAdder()
    {
        r = 0;
    }

BooleanAdder(Bitset &x0, Bitset &x1, Bitset &y0) : x0__(x0), x1__(x1), y0__(y0) 
    {
        r = 0;
    }

void set_values(Bitset &x0, Bitset &x1, Bitset &y0) 
    {
        x0__ = x0;
        x1__ = x1;
        y0__ = y0;
    }

int get_rounds() {
    return r;
}

int get_total_rounds() {
    return BITLENGTH;
}

bool is_done() {
    return r == BITLENGTH;
}

void step() 
{
  switch(r) {
    case 0:
      temp__[248] = x0__[0] ^ x1__[0];   
      temp__[247] = x0__[1] ^ x1__[1];
      temp__[246] = x0__[2] ^ x1__[2];
      temp__[245] = x0__[3] ^ x1__[3];
      temp__[244] = x0__[4] ^ x1__[4];
      temp__[243] = x0__[5] ^ x1__[5];
      temp__[242] = x0__[6] ^ x1__[6];
      temp__[241] = x0__[7] ^ x1__[7];
      temp__[240] = x0__[8] ^ x1__[8];
      temp__[239] = x0__[9] ^ x1__[9];
      temp__[238] = x0__[10] ^ x1__[10];
      temp__[237] = x0__[11] ^ x1__[11];
      temp__[236] = x0__[12] ^ x1__[12];
      temp__[235] = x0__[13] ^ x1__[13];
      temp__[234] = x0__[14] ^ x1__[14];
      temp__[233] = x0__[15] ^ x1__[15];
      temp__[232] = x0__[16] ^ x1__[16];
      temp__[231] = x0__[17] ^ x1__[17];
      temp__[230] = x0__[18] ^ x1__[18];
      temp__[229] = x0__[19] ^ x1__[19];
      temp__[228] = x0__[20] ^ x1__[20];
      temp__[227] = x0__[21] ^ x1__[21];
      temp__[226] = x0__[22] ^ x1__[22];
      temp__[225] = x0__[23] ^ x1__[23];
      temp__[224] = x0__[24] ^ x1__[24];
      temp__[223] = x0__[25] ^ x1__[25];
      temp__[222] = x0__[26] ^ x1__[26];
      temp__[221] = x0__[27] ^ x1__[27];
      temp__[220] = x0__[28] ^ x1__[28];
      temp__[219] = x0__[29] ^ x1__[29];
      temp__[218] = x0__[30] ^ x1__[30];
      temp__[217] = x0__[31] ^ x1__[31];
      temp__[216] = x0__[32] ^ x1__[32];
      temp__[215] = x0__[33] ^ x1__[33];
      temp__[214] = x0__[34] ^ x1__[34];
      temp__[213] = x0__[35] ^ x1__[35];
      temp__[212] = x0__[36] ^ x1__[36];
      temp__[211] = x0__[37] ^ x1__[37];
      temp__[210] = x0__[38] ^ x1__[38];
      temp__[209] = x0__[39] ^ x1__[39];
      temp__[208] = x0__[40] ^ x1__[40];
      temp__[207] = x0__[41] ^ x1__[41];
      temp__[206] = x0__[42] ^ x1__[42];
      temp__[205] = x0__[43] ^ x1__[43];
      temp__[204] = x0__[44] ^ x1__[44];
      temp__[203] = x0__[45] ^ x1__[45];
      temp__[202] = x0__[46] ^ x1__[46];
      temp__[201] = x0__[47] ^ x1__[47];
      temp__[200] = x0__[48] ^ x1__[48];
      temp__[199] = x0__[49] ^ x1__[49];
      temp__[198] = x0__[50] ^ x1__[50];
      temp__[197] = x0__[51] ^ x1__[51];
      temp__[196] = x0__[52] ^ x1__[52];
      temp__[195] = x0__[53] ^ x1__[53];
      temp__[194] = x0__[54] ^ x1__[54];
      temp__[193] = x0__[55] ^ x1__[55];
      temp__[192] = x0__[56] ^ x1__[56];
      temp__[191] = x0__[57] ^ x1__[57];
      temp__[190] = x0__[58] ^ x1__[58];
      temp__[189] = x0__[59] ^ x1__[59];
      temp__[188] = x0__[60] ^ x1__[60];
      temp__[187] = x0__[61] ^ x1__[61];
      temp__[186] = x0__[62] ^ x1__[62];
      y0__[63] = x0__[63] ^ x1__[63];
      temp__[249] = x0__[63] & x1__[63];
      break;
    
    case 1:
      temp__[249].complete_and();
      temp__[1] = x1__[62] ^ temp__[249];
      temp__[0] = x0__[62] ^ temp__[249];
      temp__[2] = temp__[0] & temp__[1];
      break;
    
    case 2:
      temp__[2].complete_and();
      temp__[250] = temp__[2] ^ temp__[249];
      temp__[4] = x1__[61] ^ temp__[250];
      temp__[3] = x0__[61] ^ temp__[250];
      temp__[5] = temp__[3] & temp__[4];
      break;
    
    case 3:
      temp__[5].complete_and();
      temp__[251] = temp__[5] ^ temp__[250];
      temp__[7] = x1__[60] ^ temp__[251];
      temp__[6] = x0__[60] ^ temp__[251];
      temp__[8] = temp__[6] & temp__[7];
      break;
    
    case 4:
      temp__[8].complete_and();
      temp__[252] = temp__[8] ^ temp__[251];
      temp__[10] = x1__[59] ^ temp__[252];
      temp__[9] = x0__[59] ^ temp__[252];
      temp__[11] = temp__[9] & temp__[10];
      break;
    
    case 5:
      temp__[11].complete_and();
      temp__[253] = temp__[11] ^ temp__[252];
      temp__[13] = x1__[58] ^ temp__[253];
      temp__[12] = x0__[58] ^ temp__[253];
      temp__[14] = temp__[12] & temp__[13];
      break;
    
    case 6:
      temp__[14].complete_and();
      temp__[254] = temp__[14] ^ temp__[253];
      temp__[16] = x1__[57] ^ temp__[254];
      temp__[15] = x0__[57] ^ temp__[254];
      temp__[17] = temp__[15] & temp__[16];
      break;
    
    case 7:
      temp__[17].complete_and();
      temp__[255] = temp__[17] ^ temp__[254];
      temp__[19] = x1__[56] ^ temp__[255];
      temp__[18] = x0__[56] ^ temp__[255];
      temp__[20] = temp__[18] & temp__[19];
      break;
    
    case 8:
      temp__[20].complete_and();
      temp__[256] = temp__[20] ^ temp__[255];
      temp__[22] = x1__[55] ^ temp__[256];
      temp__[21] = x0__[55] ^ temp__[256];
      temp__[23] = temp__[21] & temp__[22];
      break;
    
    case 9: 
      temp__[23].complete_and();
      temp__[257] = temp__[23] ^ temp__[256];
      temp__[25] = x1__[54] ^ temp__[257];
      temp__[24] = x0__[54] ^ temp__[257];
      temp__[26] = temp__[24] & temp__[25];
      break;
    
    case 10:
      temp__[26].complete_and();
      temp__[258] = temp__[26] ^ temp__[257];
      temp__[28] = x1__[53] ^ temp__[258];
      temp__[27] = x0__[53] ^ temp__[258];
      temp__[29] = temp__[27] & temp__[28];
      break;
    
    case 11:
      temp__[29].complete_and();
      temp__[259] = temp__[29] ^ temp__[258];
      temp__[31] = x1__[52] ^ temp__[259];
      temp__[30] = x0__[52] ^ temp__[259];
      temp__[32] = temp__[30] & temp__[31];
      break;
    
    case 12:
      temp__[32].complete_and();
      temp__[260] = temp__[32] ^ temp__[259];
      temp__[34] = x1__[51] ^ temp__[260];
      temp__[33] = x0__[51] ^ temp__[260];
      temp__[35] = temp__[33] & temp__[34];
      break;
    
    case 13:
      temp__[35].complete_and();
      temp__[261] = temp__[35] ^ temp__[260];
      temp__[37] = x1__[50] ^ temp__[261];
      temp__[36] = x0__[50] ^ temp__[261];
      temp__[38] = temp__[36] & temp__[37];
      break;
    
    case 14:
      temp__[38].complete_and();
      temp__[262] = temp__[38] ^ temp__[261];
      temp__[40] = x1__[49] ^ temp__[262];
      temp__[39] = x0__[49] ^ temp__[262];
      temp__[41] = temp__[39] & temp__[40];
      break;
    
    case 15:
      temp__[41].complete_and();
      temp__[263] = temp__[41] ^ temp__[262];
      temp__[43] = x1__[48] ^ temp__[263];
      temp__[42] = x0__[48] ^ temp__[263];
      temp__[44] = temp__[42] & temp__[43];
      break;
    
    case 16:
      temp__[44].complete_and();
      temp__[264] = temp__[44] ^ temp__[263];
      temp__[46] = x1__[47] ^ temp__[264];
      temp__[45] = x0__[47] ^ temp__[264];
      temp__[47] = temp__[45] & temp__[46];
      break;
    
    case 17:
      temp__[47].complete_and();
      temp__[265] = temp__[47] ^ temp__[264];
      temp__[49] = x1__[46] ^ temp__[265];
      temp__[48] = x0__[46] ^ temp__[265];
      temp__[50] = temp__[48] & temp__[49];
      break;
    
    case 18:
      temp__[50].complete_and();
      temp__[266] = temp__[50] ^ temp__[265];
      temp__[52] = x1__[45] ^ temp__[266];
      temp__[51] = x0__[45] ^ temp__[266];
      temp__[53] = temp__[51] & temp__[52];
      break;
    
    case 19:
      temp__[53].complete_and();
      temp__[267] = temp__[53] ^ temp__[266];
      temp__[55] = x1__[44] ^ temp__[267];
      temp__[54] = x0__[44] ^ temp__[267];
      temp__[56] = temp__[54] & temp__[55];
      break;
    
    case 20:
      temp__[56].complete_and();
      temp__[268] = temp__[56] ^ temp__[267];
      temp__[58] = x1__[43] ^ temp__[268];
      temp__[57] = x0__[43] ^ temp__[268];
      temp__[59] = temp__[57] & temp__[58];
      break;
    
    case 21:
      temp__[59].complete_and();
      temp__[269] = temp__[59] ^ temp__[268];
      temp__[61] = x1__[42] ^ temp__[269];
      temp__[60] = x0__[42] ^ temp__[269];
      temp__[62] = temp__[60] & temp__[61];
      break;
    
    case 22:
      temp__[62].complete_and();
      temp__[270] = temp__[62] ^ temp__[269];
      temp__[64] = x1__[41] ^ temp__[270];
      temp__[63] = x0__[41] ^ temp__[270];
      temp__[65] = temp__[63] & temp__[64];
      break;
    
    case 23:
      temp__[65].complete_and();
      temp__[271] = temp__[65] ^ temp__[270];
      temp__[67] = x1__[40] ^ temp__[271];
      temp__[66] = x0__[40] ^ temp__[271];
      temp__[68] = temp__[66] & temp__[67];
      break;
    
    case 24:
      temp__[68].complete_and();
      temp__[272] = temp__[68] ^ temp__[271];
      temp__[70] = x1__[39] ^ temp__[272];
      temp__[69] = x0__[39] ^ temp__[272];
      temp__[71] = temp__[69] & temp__[70];
      break;
    
    case 25:
      temp__[71].complete_and();
      temp__[273] = temp__[71] ^ temp__[272];
      temp__[73] = x1__[38] ^ temp__[273];
      temp__[72] = x0__[38] ^ temp__[273];
      temp__[74] = temp__[72] & temp__[73];
      break;
    
    case 26:
      temp__[74].complete_and();
      temp__[274] = temp__[74] ^ temp__[273];
      temp__[76] = x1__[37] ^ temp__[274];
      temp__[75] = x0__[37] ^ temp__[274];
      temp__[77] = temp__[75] & temp__[76];
      break;
    
    case 27:
      temp__[77].complete_and();
      temp__[275] = temp__[77] ^ temp__[274];
      temp__[79] = x1__[36] ^ temp__[275];
      temp__[78] = x0__[36] ^ temp__[275];
      temp__[80] = temp__[78] & temp__[79];
      break;
    
    case 28:
      temp__[80].complete_and();
      temp__[276] = temp__[80] ^ temp__[275]; 
      y0__[35] = temp__[213] ^ temp__[276];
      temp__[82] = x1__[35] ^ temp__[276];
      temp__[81] = x0__[35] ^ temp__[276];
      temp__[83] = temp__[81] & temp__[82];
      break;
    
    case 29:
      temp__[83].complete_and();
      temp__[277] = temp__[83] ^ temp__[276];
      y0__[34] = temp__[214] ^ temp__[277];
      y0__[36] = temp__[212] ^ temp__[275];
      temp__[85] = x1__[34] ^ temp__[277];
      temp__[84] = x0__[34] ^ temp__[277];
      temp__[86] = temp__[84] & temp__[85];
      break;
    
    case 30:
      temp__[86].complete_and();
      temp__[278] = temp__[86] ^ temp__[277];
      y0__[33] = temp__[215] ^ temp__[278];
      y0__[37] = temp__[211] ^ temp__[274];  
      temp__[88] = x1__[33] ^ temp__[278];
      temp__[87] = x0__[33] ^ temp__[278];
      temp__[89] = temp__[87] & temp__[88];
      break;
    
    case 31:
      temp__[89].complete_and();
      temp__[279] = temp__[89] ^ temp__[278];
      y0__[38] = temp__[210] ^ temp__[273];
      temp__[90] = x0__[32] ^ temp__[279];
      y0__[32] = temp__[216] ^ temp__[279];
      y0__[39] = temp__[209] ^ temp__[272];
      temp__[91] = x1__[32] ^ temp__[279];
      temp__[92] = temp__[90] & temp__[91];
      break;
    
    case 32:
      temp__[92].complete_and();
      temp__[280] = temp__[92] ^ temp__[279];
      y0__[31] = temp__[217] ^ temp__[280];  
      y0__[40] = temp__[208] ^ temp__[271];
      temp__[94] = x1__[31] ^ temp__[280];
      temp__[93] = x0__[31] ^ temp__[280];
      temp__[95] = temp__[93] & temp__[94];
      break;
    
    case 33:
      temp__[95].complete_and();
      temp__[281] = temp__[95] ^ temp__[280];
      y0__[30] = temp__[218] ^ temp__[281];
      y0__[41] = temp__[207] ^ temp__[270];
      temp__[97] = x1__[30] ^ temp__[281];
      temp__[96] = x0__[30] ^ temp__[281];
      temp__[98] = temp__[96] & temp__[97];
      break;
    
    case 34:
      temp__[98].complete_and();
      temp__[282] = temp__[98] ^ temp__[281];
      y0__[29] = temp__[219] ^ temp__[282]; 
      y0__[42] = temp__[206] ^ temp__[269];
      temp__[100] = x1__[29] ^ temp__[282];
      temp__[99] = x0__[29] ^ temp__[282];
      temp__[101] = temp__[99] & temp__[100];
      break;
 case 35:
  temp__[101].complete_and();
  temp__[283] = temp__[101] ^ temp__[282];
  y0__[43] = temp__[205] ^ temp__[268];
  temp__[102] = x0__[28] ^ temp__[283];
  y0__[28] = temp__[220] ^ temp__[283];
  y0__[44] = temp__[204] ^ temp__[267];
  temp__[103] = x1__[28] ^ temp__[283];
  temp__[104] = temp__[102] & temp__[103];
  break;

case 36:
  temp__[104].complete_and();
  temp__[284] = temp__[104] ^ temp__[283];
  y0__[27] = temp__[221] ^ temp__[284];
  y0__[45] = temp__[203] ^ temp__[266];
  temp__[106] = x1__[27] ^ temp__[284];
  temp__[105] = x0__[27] ^ temp__[284];
  temp__[107] = temp__[105] & temp__[106];
  break;

case 37:
  temp__[107].complete_and();
  temp__[285] = temp__[107] ^ temp__[284];
  y0__[26] = temp__[222] ^ temp__[285];
  y0__[46] = temp__[202] ^ temp__[265];
  temp__[109] = x1__[26] ^ temp__[285];
  temp__[108] = x0__[26] ^ temp__[285];
  temp__[110] = temp__[108] & temp__[109];
  break;
  
case 38:
  temp__[110].complete_and();
  temp__[286] = temp__[110] ^ temp__[285];
  y0__[25] = temp__[223] ^ temp__[286];
  y0__[47] = temp__[201] ^ temp__[264];
  temp__[112] = x1__[25] ^ temp__[286];
  temp__[111] = x0__[25] ^ temp__[286];
  temp__[113] = temp__[111] & temp__[112];
  break;
  
case 39:
  temp__[113].complete_and();
  temp__[287] = temp__[113] ^ temp__[286];
  y0__[48] = temp__[200] ^ temp__[263];
  temp__[114] = x0__[24] ^ temp__[287];
  y0__[24] = temp__[224] ^ temp__[287];
  y0__[49] = temp__[199] ^ temp__[262];
  temp__[115] = x1__[24] ^ temp__[287];
  temp__[116] = temp__[114] & temp__[115];
  break;
  
case 40:
  temp__[116].complete_and();
  temp__[288] = temp__[116] ^ temp__[287];
  y0__[23] = temp__[225] ^ temp__[288];
  y0__[50] = temp__[198] ^ temp__[261];
  temp__[118] = x1__[23] ^ temp__[288];
  temp__[117] = x0__[23] ^ temp__[288];
  temp__[119] = temp__[117] & temp__[118];
  break;

case 41:
  temp__[119].complete_and();
  temp__[289] = temp__[119] ^ temp__[288];
  y0__[22] = temp__[226] ^ temp__[289];
  y0__[51] = temp__[197] ^ temp__[260];
  temp__[121] = x1__[22] ^ temp__[289];
  temp__[120] = x0__[22] ^ temp__[289];
  temp__[122] = temp__[120] & temp__[121];
  break;
  
case 42:
  temp__[122].complete_and();
  temp__[290] = temp__[122] ^ temp__[289];
  y0__[21] = temp__[227] ^ temp__[290];
  y0__[52] = temp__[196] ^ temp__[259];
  temp__[124] = x1__[21] ^ temp__[290];
  temp__[123] = x0__[21] ^ temp__[290];
  temp__[125] = temp__[123] & temp__[124];
  break;
  
case 43:
  temp__[125].complete_and();
  temp__[291] = temp__[125] ^ temp__[290];
  y0__[53] = temp__[195] ^ temp__[258];
  temp__[126] = x0__[20] ^ temp__[291];
  y0__[20] = temp__[228] ^ temp__[291];
  y0__[54] = temp__[194] ^ temp__[257];
  temp__[127] = x1__[20] ^ temp__[291];
  temp__[128] = temp__[126] & temp__[127];
  break;
  
case 44:
  temp__[128].complete_and();
  temp__[292] = temp__[128] ^ temp__[291];
  y0__[19] = temp__[229] ^ temp__[292];
  y0__[55] = temp__[193] ^ temp__[256];
  temp__[130] = x1__[19] ^ temp__[292];
  temp__[129] = x0__[19] ^ temp__[292];
  temp__[131] = temp__[129] & temp__[130];
  break;
  
case 45:
  temp__[131].complete_and();
  temp__[293] = temp__[131] ^ temp__[292];
  y0__[18] = temp__[230] ^ temp__[293];
  y0__[56] = temp__[192] ^ temp__[255];
  temp__[133] = x1__[18] ^ temp__[293];
  temp__[132] = x0__[18] ^ temp__[293];
  temp__[134] = temp__[132] & temp__[133];
  break;
  
case 46:
  temp__[134].complete_and();
  temp__[294] = temp__[134] ^ temp__[293];
  y0__[17] = temp__[231] ^ temp__[294];
  y0__[57] = temp__[191] ^ temp__[254];
  temp__[136] = x1__[17] ^ temp__[294];
  temp__[135] = x0__[17] ^ temp__[294];
  temp__[137] = temp__[135] & temp__[136];
  break;
  
case 47:
  temp__[137].complete_and();
  temp__[295] = temp__[137] ^ temp__[294];
  y0__[58] = temp__[190] ^ temp__[253];
  temp__[138] = x0__[16] ^ temp__[295];
  y0__[16] = temp__[232] ^ temp__[295];
  y0__[59] = temp__[189] ^ temp__[252];
  temp__[139] = x1__[16] ^ temp__[295];
  temp__[140] = temp__[138] & temp__[139];
  break;
  
case 48:
  temp__[140].complete_and();
  temp__[296] = temp__[140] ^ temp__[295];
  y0__[15] = temp__[233] ^ temp__[296];
  y0__[60] = temp__[188] ^ temp__[251];
  temp__[142] = x1__[15] ^ temp__[296];
  temp__[141] = x0__[15] ^ temp__[296];
  temp__[143] = temp__[141] & temp__[142];
  break;
  
case 49:
  temp__[143].complete_and();
  temp__[297] = temp__[143] ^ temp__[296];
  y0__[14] = temp__[234] ^ temp__[297];
  y0__[61] = temp__[187] ^ temp__[250];
  temp__[145] = x1__[14] ^ temp__[297];
  temp__[144] = x0__[14] ^ temp__[297];
  temp__[146] = temp__[144] & temp__[145];
  break;
  
case 50:
  temp__[146].complete_and();
  temp__[298] = temp__[146] ^ temp__[297];
  y0__[13] = temp__[235] ^ temp__[298];
  y0__[62] = temp__[186] ^ temp__[249];
  temp__[148] = x1__[13] ^ temp__[298];
  temp__[147] = x0__[13] ^ temp__[298];
  temp__[149] = temp__[147] & temp__[148];
  break;
  
case 51:
  temp__[149].complete_and();
  temp__[299] = temp__[149] ^ temp__[298];
  temp__[151] = x1__[12] ^ temp__[299];
  temp__[150] = x0__[12] ^ temp__[299];
  temp__[152] = temp__[150] & temp__[151];
  break;
  
case 52:
  temp__[152].complete_and();
  temp__[300] = temp__[152] ^ temp__[299];
  temp__[154] = x1__[11] ^ temp__[300];
  temp__[153] = x0__[11] ^ temp__[300];
  temp__[155] = temp__[153] & temp__[154];
  break;
  
case 53:
  temp__[155].complete_and();
  temp__[301] = temp__[155] ^ temp__[300];
  temp__[157] = x1__[10] ^ temp__[301];
  temp__[156] = x0__[10] ^ temp__[301];
  temp__[158] = temp__[156] & temp__[157];
  break;
  
case 54:
  temp__[158].complete_and();
  temp__[302] = temp__[158] ^ temp__[301];
  temp__[160] = x1__[9] ^ temp__[302];
  temp__[159] = x0__[9] ^ temp__[302];
  temp__[161] = temp__[159] & temp__[160];
  break;
  
case 55:
  temp__[161].complete_and();
  temp__[303] = temp__[161] ^ temp__[302];
  temp__[163] = x1__[8] ^ temp__[303];
  temp__[162] = x0__[8] ^ temp__[303];
  temp__[164] = temp__[162] & temp__[163];
  break;
  
case 56:
  temp__[164].complete_and();
  temp__[304] = temp__[164] ^ temp__[303];
  temp__[166] = x1__[7] ^ temp__[304];
  temp__[165] = x0__[7] ^ temp__[304];
  temp__[167] = temp__[165] & temp__[166];
  break;
  
case 57:
  temp__[167].complete_and();
  temp__[305] = temp__[167] ^ temp__[304];
  y0__[6] = temp__[242] ^ temp__[305];
  temp__[169] = x1__[6] ^ temp__[305];
  temp__[168] = x0__[6] ^ temp__[305];
  temp__[170] = temp__[168] & temp__[169];
  break;
  
case 58:
  temp__[170].complete_and();
  temp__[306] = temp__[170] ^ temp__[305];
  y0__[5] = temp__[243] ^ temp__[306];
  y0__[7] = temp__[241] ^ temp__[304];
  temp__[172] = x1__[5] ^ temp__[306];
  temp__[171] = x0__[5] ^ temp__[306];
  temp__[173] = temp__[171] & temp__[172];
  break;
  
case 59:
  temp__[173].complete_and();
  temp__[307] = temp__[173] ^ temp__[306];
  y0__[4] = temp__[244] ^ temp__[307];
  y0__[8] = temp__[240] ^ temp__[303];
  temp__[175] = x1__[4] ^ temp__[307];
  temp__[174] = x0__[4] ^ temp__[307];
  temp__[176] = temp__[174] & temp__[175];
  break;
  
case 60:
  temp__[176].complete_and();
  temp__[308] = temp__[176] ^ temp__[307];
  y0__[9] = temp__[239] ^ temp__[302];
  temp__[177] = x0__[3] ^ temp__[308];
  y0__[3] = temp__[245] ^ temp__[308];
  y0__[10] = temp__[238] ^ temp__[301];
  temp__[178] = x1__[3] ^ temp__[308];
  temp__[179] = temp__[177] & temp__[178];
  break;
  
case 61:
  temp__[179].complete_and();
  temp__[309] = temp__[179] ^ temp__[308];
  y0__[2] = temp__[246] ^ temp__[309];
  y0__[11] = temp__[237] ^ temp__[300];
  temp__[181] = x1__[2] ^ temp__[309]; 
  temp__[180] = x0__[2] ^ temp__[309];
  temp__[182] = temp__[180] & temp__[181];
  break;

case 62:
  temp__[182].complete_and();
  temp__[310] = temp__[182] ^ temp__[309];
  y0__[1] = temp__[247] ^ temp__[310];
  y0__[12] = temp__[236] ^ temp__[299];
  temp__[184] = x1__[1] ^ temp__[310];
  temp__[183] = x0__[1] ^ temp__[310];
  temp__[185] = temp__[183] & temp__[184];
  break;
  
case 63:
  temp__[185].complete_and();
  temp__[311] = temp__[185] ^ temp__[310];
  y0__[0] = temp__[248] ^ temp__[311];
  break;
  
default:
  break;
}

r++;
}
/* void step() */
/* { */
/* switch(r) { */
/* case 0: */
/* temp__[248] = x0__[63] ^ x1__[63]; */
/* temp__[247] = x0__[62] ^ x1__[62]; */
/* temp__[246] = x0__[61] ^ x1__[61]; */  
/* temp__[245] = x0__[60] ^ x1__[60]; */
/* temp__[244] = x0__[59] ^ x1__[59]; */
/* temp__[243] = x0__[58] ^ x1__[58]; */
/* temp__[242] = x0__[57] ^ x1__[57]; */
/* temp__[241] = x0__[56] ^ x1__[56]; */
/* temp__[240] = x0__[55] ^ x1__[55]; */
/* temp__[239] = x0__[54] ^ x1__[54]; */
/* temp__[238] = x0__[53] ^ x1__[53]; */
/* temp__[237] = x0__[52] ^ x1__[52]; */
/* temp__[236] = x0__[51] ^ x1__[51]; */
/* temp__[235] = x0__[50] ^ x1__[50]; */
/* temp__[234] = x0__[49] ^ x1__[49]; */
/* temp__[233] = x0__[48] ^ x1__[48]; */
/* temp__[232] = x0__[47] ^ x1__[47]; */
/* temp__[231] = x0__[46] ^ x1__[46]; */
/* temp__[230] = x0__[45] ^ x1__[45]; */
/* temp__[229] = x0__[44] ^ x1__[44]; */
/* temp__[228] = x0__[43] ^ x1__[43]; */
/* temp__[227] = x0__[42] ^ x1__[42]; */
/* temp__[226] = x0__[41] ^ x1__[41]; */
/* temp__[225] = x0__[40] ^ x1__[40]; */
/* temp__[224] = x0__[39] ^ x1__[39]; */
/* temp__[223] = x0__[38] ^ x1__[38]; */
/* temp__[222] = x0__[37] ^ x1__[37]; */
/* temp__[221] = x0__[36] ^ x1__[36]; */
/* temp__[220] = x0__[35] ^ x1__[35]; */
/* temp__[219] = x0__[34] ^ x1__[34]; */
/* temp__[218] = x0__[33] ^ x1__[33]; */
/* temp__[217] = x0__[32] ^ x1__[32]; */
/* temp__[216] = x0__[31] ^ x1__[31]; */
/* temp__[215] = x0__[30] ^ x1__[30]; */
/* temp__[214] = x0__[29] ^ x1__[29]; */
/* temp__[213] = x0__[28] ^ x1__[28]; */
/* temp__[212] = x0__[27] ^ x1__[27]; */
/* temp__[211] = x0__[26] ^ x1__[26]; */
/* temp__[210] = x0__[25] ^ x1__[25]; */
/* temp__[209] = x0__[24] ^ x1__[24]; */
/* temp__[208] = x0__[23] ^ x1__[23]; */
/* temp__[207] = x0__[22] ^ x1__[22]; */
/* temp__[206] = x0__[21] ^ x1__[21]; */
/* temp__[205] = x0__[20] ^ x1__[20]; */
/* temp__[204] = x0__[19] ^ x1__[19]; */
/* temp__[203] = x0__[18] ^ x1__[18]; */
/* temp__[202] = x0__[17] ^ x1__[17]; */
/* temp__[201] = x0__[16] ^ x1__[16]; */
/* temp__[200] = x0__[15] ^ x1__[15]; */
/* temp__[199] = x0__[14] ^ x1__[14]; */
/* temp__[198] = x0__[13] ^ x1__[13]; */
/* temp__[197] = x0__[12] ^ x1__[12]; */
/* temp__[196] = x0__[11] ^ x1__[11]; */
/* temp__[195] = x0__[10] ^ x1__[10]; */
/* temp__[194] = x0__[9] ^ x1__[9]; */
/* temp__[193] = x0__[8] ^ x1__[8]; */
/* temp__[192] = x0__[7] ^ x1__[7]; */
/* temp__[191] = x0__[6] ^ x1__[6]; */
/* temp__[190] = x0__[5] ^ x1__[5]; */
/* temp__[189] = x0__[4] ^ x1__[4]; */
/* temp__[188] = x0__[3] ^ x1__[3]; */
/* temp__[187] = x0__[2] ^ x1__[2]; */
/* temp__[186] = x0__[1] ^ x1__[1]; */  
/* y0__[0] = x0__[0] ^ x1__[0]; */
/* temp__[249] = x0__[0] & x1__[0]; */
/* break; */
/* case 1: */  
/* temp__[249].complete_and(); */
/* temp__[1] = x1__[1] ^ temp__[249]; */
/* temp__[0] = x0__[1] ^ temp__[249]; */
/* temp__[2] = temp__[0] & temp__[1]; */ 
/* break; */
/* case 2: */
/* temp__[2].complete_and(); */
/* temp__[250] = temp__[2] ^ temp__[249]; */
/* temp__[4] = x1__[2] ^ temp__[250]; */
/* temp__[3] = x0__[2] ^ temp__[250]; */
/* temp__[5] = temp__[3] & temp__[4]; */
/* break; */
/* case 3: */
/* temp__[5].complete_and(); */  
/* temp__[251] = temp__[5] ^ temp__[250]; */ 
/* temp__[7] = x1__[3] ^ temp__[251]; */
/* temp__[6] = x0__[3] ^ temp__[251]; */
/* temp__[8] = temp__[6] & temp__[7]; */
/* break; */
/* case 4: */  
/* temp__[8].complete_and(); */
/* temp__[252] = temp__[8] ^ temp__[251]; */
/* temp__[10] = x1__[4] ^ temp__[252]; */
/* temp__[9] = x0__[4] ^ temp__[252]; */
/* temp__[11] = temp__[9] & temp__[10]; */
/* break; */
/* case 5: */
/* temp__[11].complete_and(); */
/* temp__[253] = temp__[11] ^ temp__[252]; */ 
/* temp__[13] = x1__[5] ^ temp__[253]; */
/* temp__[12] = x0__[5] ^ temp__[253]; */
/* temp__[14] = temp__[12] & temp__[13]; */  
/* break; */
/* case 6: */
/* temp__[14].complete_and(); */
/* temp__[254] = temp__[14] ^ temp__[253]; */
/* temp__[16] = x1__[6] ^ temp__[254]; */
/* temp__[15] = x0__[6] ^ temp__[254]; */
/* temp__[17] = temp__[15] & temp__[16]; */
/* break; */
/* case 7: */
/* temp__[17].complete_and(); */
/* temp__[255] = temp__[17] ^ temp__[254]; */
/* temp__[19] = x1__[7] ^ temp__[255]; */
/* temp__[18] = x0__[7] ^ temp__[255]; */
/* temp__[20] = temp__[18] & temp__[19]; */  
/* break; */
/* case 8: */
/* temp__[20].complete_and(); */
/* temp__[256] = temp__[20] ^ temp__[255]; */ 
/* temp__[22] = x1__[8] ^ temp__[256]; */
/* temp__[21] = x0__[8] ^ temp__[256]; */
/* temp__[23] = temp__[21] & temp__[22]; */
/* break; */
/* case 9: */
/* temp__[23].complete_and(); */
/* temp__[257] = temp__[23] ^ temp__[256]; */
/* temp__[25] = x1__[9] ^ temp__[257]; */
/* temp__[24] = x0__[9] ^ temp__[257]; */
/* temp__[26] = temp__[24] & temp__[25]; */
/* break; */
/* case 10: */  
/* temp__[26].complete_and(); */
/* temp__[258] = temp__[26] ^ temp__[257]; */
/* temp__[28] = x1__[10] ^ temp__[258]; */
/* temp__[27] = x0__[10] ^ temp__[258]; */  
/* temp__[29] = temp__[27] & temp__[28]; */
/* break; */
/* case 11: */
/* temp__[29].complete_and(); */ 
/* temp__[259] = temp__[29] ^ temp__[258]; */
/* temp__[31] = x1__[11] ^ temp__[259]; */
/* temp__[30] = x0__[11] ^ temp__[259]; */
/* temp__[32] = temp__[30] & temp__[31]; */
/* break; */
/* case 12: */
/* temp__[32].complete_and(); */
/* temp__[260] = temp__[32] ^ temp__[259]; */  
/* temp__[34] = x1__[12] ^ temp__[260]; */
/* temp__[33] = x0__[12] ^ temp__[260]; */
/* temp__[35] = temp__[33] & temp__[34]; */
/* break; */
/* case 13: */
/* temp__[35].complete_and(); */
/* temp__[261] = temp__[35] ^ temp__[260]; */
/* temp__[37] = x1__[13] ^ temp__[261]; */
/* temp__[36] = x0__[13] ^ temp__[261]; */ 
/* temp__[38] = temp__[36] & temp__[37]; */  
/* break; */
/* case 14: */  
/* temp__[38].complete_and(); */
/* temp__[262] = temp__[38] ^ temp__[261]; */
/* temp__[40] = x1__[14] ^ temp__[262]; */
/* temp__[39] = x0__[14] ^ temp__[262]; */
/* temp__[41] = temp__[39] & temp__[40]; */
/* break; */
/* case 15: */
/* temp__[41].complete_and(); */
/* temp__[263] = temp__[41] ^ temp__[262]; */ 
/* temp__[43] = x1__[15] ^ temp__[263]; */  
/* temp__[42] = x0__[15] ^ temp__[263]; */
/* temp__[44] = temp__[42] & temp__[43]; */
/* break; */
/* case 16: */
/* temp__[44].complete_and(); */
/* temp__[264] = temp__[44] ^ temp__[263]; */
/* temp__[46] = x1__[16] ^ temp__[264]; */
/* temp__[45] = x0__[16] ^ temp__[264]; */
/* temp__[47] = temp__[45] & temp__[46]; */  
/* break; */
/* case 17: */
/* temp__[47].complete_and(); */
/* temp__[265] = temp__[47] ^ temp__[264]; */
/* temp__[49] = x1__[17] ^ temp__[265]; */
/* temp__[48] = x0__[17] ^ temp__[265]; */
/* temp__[50] = temp__[48] & temp__[49]; */
/* break; */
/* case 18: */ 
/* temp__[50].complete_and(); */
/* temp__[266] = temp__[50] ^ temp__[265]; */
/* temp__[52] = x1__[18] ^ temp__[266]; */ 
/* temp__[51] = x0__[18] ^ temp__[266]; */
/* temp__[53] = temp__[51] & temp__[52]; */
/* break; */
/* case 19: */  
/* temp__[53].complete_and(); */
/* temp__[267] = temp__[53] ^ temp__[266]; */
/* temp__[55] = x1__[19] ^ temp__[267]; */
/* temp__[54] = x0__[19] ^ temp__[267]; */
/* temp__[56] = temp__[54] & temp__[55]; */ 
/* break; */
/* case 20: */
/* temp__[56].complete_and(); */  
/* temp__[268] = temp__[56] ^ temp__[267]; */
/* temp__[58] = x1__[20] ^ temp__[268]; */
/* temp__[57] = x0__[20] ^ temp__[268]; */
/* temp__[59] = temp__[57] & temp__[58]; */
/* break; */
/* case 21: */
/* temp__[59].complete_and(); */
/* temp__[269] = temp__[59] ^ temp__[268]; */
/* temp__[61] = x1__[21] ^ temp__[269]; */  
/* temp__[60] = x0__[21] ^ temp__[269]; */
/* temp__[62] = temp__[60] & temp__[61]; */
/* break; */
/* case 22: */ 
/* temp__[62].complete_and(); */
/* temp__[270] = temp__[62] ^ temp__[269]; */
/* temp__[64] = x1__[22] ^ temp__[270]; */
/* temp__[63] = x0__[22] ^ temp__[270]; */
/* temp__[65] = temp__[63] & temp__[64]; */
/* break; */
/* case 23: */
/* temp__[65].complete_and(); */  
/* temp__[271] = temp__[65] ^ temp__[270]; */
/* temp__[67] = x1__[23] ^ temp__[271]; */
/* temp__[66] = x0__[23] ^ temp__[271]; */
/* temp__[68] = temp__[66] & temp__[67]; */
/* break; */
/* case 24: */
/* temp__[68].complete_and(); */
/* temp__[272] = temp__[68] ^ temp__[271]; */ 
/* temp__[70] = x1__[24] ^ temp__[272]; */
/* temp__[69] = x0__[24] ^ temp__[272]; */
/* temp__[71] = temp__[69] & temp__[70]; */  
/* break; */
/* case 25: */  
/* temp__[71].complete_and(); */
/* temp__[273] = temp__[71] ^ temp__[272]; */
/* temp__[73] = x1__[25] ^ temp__[273]; */
/* temp__[72] = x0__[25] ^ temp__[273]; */
/* temp__[74] = temp__[72] & temp__[73]; */
/* break; */
/* case 26: */
/* temp__[74].complete_and(); */
/* temp__[274] = temp__[74] ^ temp__[273]; */  
/* temp__[76] = x1__[26] ^ temp__[274]; */
/* temp__[75] = x0__[26] ^ temp__[274]; */
/* temp__[77] = temp__[75] & temp__[76]; */
/* break; */
/* case 27: */
/* temp__[77].complete_and(); */
/* temp__[275] = temp__[77] ^ temp__[274]; */
/* temp__[79] = x1__[27] ^ temp__[275]; */ 
/* temp__[78] = x0__[27] ^ temp__[275]; */
/* temp__[80] = temp__[78] & temp__[79]; */
/* break; */
/* case 28: */  
/* temp__[80].complete_and(); */
/* temp__[276] = temp__[80] ^ temp__[275]; */
/* y0__[28] = temp__[213] ^ temp__[276]; */
/* temp__[82] = x1__[28] ^ temp__[276]; */
/* temp__[81] = x0__[28] ^ temp__[276]; */
/* temp__[83] = temp__[81] & temp__[82]; */ 
/* break; */
/* case 29: */
/* temp__[83].complete_and(); */
/* temp__[277] = temp__[83] ^ temp__[276]; */  
/* y0__[29] = temp__[214] ^ temp__[277]; */
/* y0__[27] = temp__[212] ^ temp__[275]; */ 
/* temp__[85] = x1__[29] ^ temp__[277]; */
/* temp__[84] = x0__[29] ^ temp__[277]; */
/* temp__[86] = temp__[84] & temp__[85]; */
/* break; */
/* case 30: */
/* temp__[86].complete_and(); */  
/* temp__[278] = temp__[86] ^ temp__[277]; */
/* y0__[30] = temp__[215] ^ temp__[278]; */ 
/* y0__[26] = temp__[211] ^ temp__[274]; */
/* temp__[88] = x1__[30] ^ temp__[278]; */
/* temp__[87] = x0__[30] ^ temp__[278]; */
/* temp__[89] = temp__[87] & temp__[88]; */  
/* break; */
/* case 31: */
/* temp__[89].complete_and(); */
/* temp__[279] = temp__[89] ^ temp__[278]; */
/* y0__[25] = temp__[210] ^ temp__[273]; */
/* temp__[90] = x0__[31] ^ temp__[279]; */
/* y0__[31] = temp__[216] ^ temp__[279]; */  
/* y0__[24] = temp__[209] ^ temp__[272]; */
/* temp__[91] = x1__[31] ^ temp__[279]; */ 
/* temp__[92] = temp__[90] & temp__[91]; */
/* break; */
/* case 32: */  
/* temp__[92].complete_and(); */
/* temp__[280] = temp__[92] ^ temp__[279]; */
/* y0__[32] = temp__[217] ^ temp__[280]; */
/* y0__[23] = temp__[208] ^ temp__[271]; */
/* temp__[94] = x1__[32] ^ temp__[280]; */
/* temp__[93] = x0__[32] ^ temp__[280]; */
/* temp__[95] = temp__[93] & temp__[94]; */
/* break; */
/* case 33: */
/* temp__[95].complete_and(); */ 
/* temp__[281] = temp__[95] ^ temp__[280]; */
/* y0__[33] = temp__[218] ^ temp__[281]; */  
/* y0__[22] = temp__[207] ^ temp__[270]; */
/* temp__[97] = x1__[33] ^ temp__[281]; */
/* temp__[96] = x0__[33] ^ temp__[281]; */
/* temp__[98] = temp__[96] & temp__[97]; */
/* break; */
/* case 34: */  
/* temp__[98].complete_and(); */
/* temp__[282] = temp__[98] ^ temp__[281]; */ 
/* y0__[34] = temp__[219] ^ temp__[282]; */
/* y0__[21] = temp__[206] ^ temp__[269]; */
/* temp__[100] = x1__[34] ^ temp__[282]; */  
/* temp__[99] = x0__[34] ^ temp__[282]; */
/* temp__[101] = temp__[99] & temp__[100]; */ 
/* break; */
/* case 35: */
/* temp__[101].complete_and(); */  
/* temp__[283] = temp__[101] ^ temp__[282]; */
/* y0__[20] = temp__[205] ^ temp__[268]; */
/* temp__[102] = x0__[35] ^ temp__[283]; */
/* y0__[35] = temp__[220] ^ temp__[283]; */
/* y0__[19] = temp__[204] ^ temp__[267]; */
/* temp__[103] = x1__[35] ^ temp__[283]; */
/* temp__[104] = temp__[102] & temp__[103]; */
/* break; */
/* case 36: */
/* temp__[104].complete_and(); */
/* temp__[284] = temp__[104] ^ temp__[283]; */
/* y0__[36] = temp__[221] ^ temp__[284]; */
/* y0__[18] = temp__[203] ^ temp__[266]; */  
/* temp__[106] = x1__[36] ^ temp__[284]; */
/* temp__[105] = x0__[36] ^ temp__[284]; */
/* temp__[107] = temp__[105] & temp__[106]; */
/* break; */
/* case 37: */
/* temp__[107].complete_and(); */
/* temp__[285] = temp__[107] ^ temp__[284]; */
/* y0__[37] = temp__[222] ^ temp__[285]; */
/* y0__[17] = temp__[202] ^ temp__[265]; */
/* temp__[109] = x1__[37] ^ temp__[285]; */
/* temp__[108] = x0__[37] ^ temp__[285]; */
/* temp__[110] = temp__[108] & temp__[109]; */
/* break; */
/* case 38: */
/* temp__[110].complete_and(); */  
/* temp__[286] = temp__[110] ^ temp__[285]; */
/* y0__[38] = temp__[223] ^ temp__[286]; */ 
/* y0__[16] = temp__[201] ^ temp__[264]; */
/* temp__[112] = x1__[38] ^ temp__[286]; */
/* temp__[111] = x0__[38] ^ temp__[286]; */
/* temp__[113] = temp__[111] & temp__[112]; */
/* break; */
/* case 39: */
/* temp__[113].complete_and(); */
/* temp__[287] = temp__[113] ^ temp__[286]; */
/* y0__[15] = temp__[200] ^ temp__[263]; */
/* temp__[114] = x0__[39] ^ temp__[287]; */
/* y0__[39] = temp__[224] ^ temp__[287]; */
/* y0__[14] = temp__[199] ^ temp__[262]; */  
/* temp__[115] = x1__[39] ^ temp__[287]; */
/* temp__[116] = temp__[114] & temp__[115]; */
/* break; */
/* case 40: */
/* temp__[116].complete_and(); */  
/* temp__[288] = temp__[116] ^ temp__[287]; */
/* y0__[40] = temp__[225] ^ temp__[288]; */
/* y0__[13] = temp__[198] ^ temp__[261]; */
/* temp__[118] = x1__[40] ^ temp__[288]; */
/* temp__[117] = x0__[40] ^ temp__[288]; */
/* temp__[119] = temp__[117] & temp__[118]; */
/* break; */
/* case 41: */
/* temp__[119].complete_and(); */ 
/* temp__[289] = temp__[119] ^ temp__[288]; */
/* y0__[41] = temp__[226] ^ temp__[289]; */
/* y0__[12] = temp__[197] ^ temp__[260]; */
/* temp__[121] = x1__[41] ^ temp__[289]; */
/* temp__[120] = x0__[41] ^ temp__[289]; */
/* temp__[122] = temp__[120] & temp__[121]; */
/* break; */
/* case 42: */  
/* temp__[122].complete_and(); */
/* temp__[290] = temp__[122] ^ temp__[289]; */
/* y0__[42] = temp__[227] ^ temp__[290]; */
/* y0__[11] = temp__[196] ^ temp__[259]; */
/* temp__[124] = x1__[42] ^ temp__[290]; */
/* temp__[123] = x0__[42] ^ temp__[290]; */
/* temp__[125] = temp__[123] & temp__[124]; */
/* break; */
/* case 43: */
/* temp__[125].complete_and(); */  
/* temp__[291] = temp__[125] ^ temp__[290]; */
/* y0__[10] = temp__[195] ^ temp__[258]; */
/* temp__[126] = x0__[43] ^ temp__[291]; */
/* y0__[43] = temp__[228] ^ temp__[291]; */
/* y0__[9] = temp__[194] ^ temp__[257]; */
/* temp__[127] = x1__[43] ^ temp__[291]; */
/* temp__[128] = temp__[126] & temp__[127]; */
/* break; */
/* case 44: */
/* temp__[128].complete_and(); */
/* temp__[292] = temp__[128] ^ temp__[291]; */  
/* y0__[44] = temp__[229] ^ temp__[292]; */
/* y0__[8] = temp__[193] ^ temp__[256]; */
/* temp__[130] = x1__[44] ^ temp__[292]; */
/* temp__[129] = x0__[44] ^ temp__[292]; */
/* temp__[131] = temp__[129] & temp__[130]; */
/* break; */
/* case 45: */
/* temp__[131].complete_and(); */
/* temp__[293] = temp__[131] ^ temp__[292]; */
/* y0__[45] = temp__[230] ^ temp__[293]; */
/* y0__[7] = temp__[192] ^ temp__[255]; */  
/* temp__[133] = x1__[45] ^ temp__[293]; */
/* temp__[132] = x0__[45] ^ temp__[293]; */
/* temp__[134] = temp__[132] & temp__[133]; */
/* break; */
/* case 46: */
/* temp__[134].complete_and(); */ 
/* temp__[294] = temp__[134] ^ temp__[293]; */
/* y0__[46] = temp__[231] ^ temp__[294]; */
/* y0__[6] = temp__[191] ^ temp__[254]; */
/* temp__[136] = x1__[46] ^ temp__[294]; */
/* temp__[135] = x0__[46] ^ temp__[294]; */
/* temp__[137] = temp__[135] & temp__[136]; */
/* break; */
/* case 47: */  
/* temp__[137].complete_and(); */
/* temp__[295] = temp__[137] ^ temp__[294]; */
/* y0__[5] = temp__[190] ^ temp__[253]; */
/* temp__[138] = x0__[47] ^ temp__[295]; */
/* y0__[47] = temp__[232] ^ temp__[295]; */
/* y0__[4] = temp__[189] ^ temp__[252]; */
/* temp__[139] = x1__[47] ^ temp__[295]; */
/* temp__[140] = temp__[138] & temp__[139]; */
/* break; */
/* case 48: */
/* temp__[140].complete_and(); */  
/* temp__[296] = temp__[140] ^ temp__[295]; */
/* y0__[48] = temp__[233] ^ temp__[296]; */
/* y0__[3] = temp__[188] ^ temp__[251]; */ 
/* temp__[142] = x1__[48] ^ temp__[296]; */
/* temp__[141] = x0__[48] ^ temp__[296]; */
/* temp__[143] = temp__[141] & temp__[142]; */
/* break; */
/* case 49: */
/* temp__[143].complete_and(); */
/* temp__[297] = temp__[143] ^ temp__[296]; */
/* y0__[49] = temp__[234] ^ temp__[297]; */
/* y0__[2] = temp__[187] ^ temp__[250]; */
/* temp__[145] = x1__[49] ^ temp__[297]; */
/* temp__[144] = x0__[49] ^ temp__[297]; */  
/* temp__[146] = temp__[144] & temp__[145]; */
/* break; */
/* case 50: */
/* temp__[146].complete_and(); */
/* temp__[298] = temp__[146] ^ temp__[297]; */
/* y0__[50] = temp__[235] ^ temp__[298]; */  
/* y0__[1] = temp__[186] ^ temp__[249]; */
/* temp__[148] = x1__[50] ^ temp__[298]; */
/* temp__[147] = x0__[50] ^ temp__[298]; */
/* temp__[149] = temp__[147] & temp__[148]; */
/* break; */
/* case 51: */
/* temp__[149].complete_and(); */ 
/* temp__[299] = temp__[149] ^ temp__[298]; */
/* temp__[151] = x1__[51] ^ temp__[299]; */
/* temp__[150] = x0__[51] ^ temp__[299]; */
/* temp__[152] = temp__[150] & temp__[151]; */
/* break; */
/* case 52: */  
/* temp__[152].complete_and(); */
/* temp__[300] = temp__[152] ^ temp__[299]; */
/* temp__[154] = x1__[52] ^ temp__[300]; */
/* temp__[153] = x0__[52] ^ temp__[300]; */
/* temp__[155] = temp__[153] & temp__[154]; */
/* break; */
/* case 53: */
/* temp__[155].complete_and(); */
/* temp__[301] = temp__[155] ^ temp__[300]; */  
/* temp__[157] = x1__[53] ^ temp__[301]; */ 
/* temp__[156] = x0__[53] ^ temp__[301]; */
/* temp__[158] = temp__[156] & temp__[157]; */
/* break; */
/* case 54: */
/* temp__[158].complete_and(); */
/* temp__[302] = temp__[158] ^ temp__[301]; */
/* temp__[160] = x1__[54] ^ temp__[302]; */
/* temp__[159] = x0__[54] ^ temp__[302]; */
/* temp__[161] = temp__[159] & temp__[160]; */  
/* break; */
/* case 55: */
/* temp__[161].complete_and(); */
/* temp__[303] = temp__[161] ^ temp__[302]; */
/* temp__[163] = x1__[55] ^ temp__[303]; */
/* temp__[162] = x0__[55] ^ temp__[303]; */
/* temp__[164] = temp__[162] & temp__[163]; */  
/* break; */
/* case 56: */
/* temp__[164].complete_and(); */  
/* temp__[304] = temp__[164] ^ temp__[303]; */
/* temp__[166] = x1__[56] ^ temp__[304]; */
/* temp__[165] = x0__[56] ^ temp__[304]; */
/* temp__[167] = temp__[165] & temp__[166]; */
/* break; */
/* case 57: */
/* temp__[167].complete_and(); */
/* temp__[305] = temp__[167] ^ temp__[304]; */
/* y0__[57] = temp__[242] ^ temp__[305]; */  
/* temp__[169] = x1__[57] ^ temp__[305]; */
/* temp__[168] = x0__[57] ^ temp__[305]; */
/* temp__[170] = temp__[168] & temp__[169]; */
/* break; */
/* case 58: */  
/* temp__[170].complete_and(); */
/* temp__[306] = temp__[170] ^ temp__[305]; */
/* y0__[58] = temp__[243] ^ temp__[306]; */ 
/* y0__[56] = temp__[241] ^ temp__[304]; */
/* temp__[172] = x1__[58] ^ temp__[306]; */
/* temp__[171] = x0__[58] ^ temp__[306]; */
/* temp__[173] = temp__[171] & temp__[172]; */
/* break; */
/* case 59: */
/* temp__[173].complete_and(); */
/* temp__[307] = temp__[173] ^ temp__[306]; */
/* y0__[59] = temp__[244] ^ temp__[307]; */
/* y0__[55] = temp__[240] ^ temp__[303]; */  
/* temp__[175] = x1__[59] ^ temp__[307]; */
/* temp__[174] = x0__[59] ^ temp__[307]; */
/* temp__[176] = temp__[174] & temp__[175]; */
/* break; */
/* case 60: */
/* temp__[176].complete_and(); */  
/* temp__[308] = temp__[176] ^ temp__[307]; */
/* y0__[54] = temp__[239] ^ temp__[302]; */
/* temp__[177] = x0__[60] ^ temp__[308]; */
/* y0__[60] = temp__[245] ^ temp__[308]; */
/* y0__[53] = temp__[238] ^ temp__[301]; */  
/* temp__[178] = x1__[60] ^ temp__[308]; */
/* temp__[179] = temp__[177] & temp__[178]; */
/* break; */
/* case 61: */
/* temp__[179].complete_and(); */
/* temp__[309] = temp__[179] ^ temp__[308]; */
/* y0__[61] = temp__[246] ^ temp__[309]; */ 
/* y0__[52] = temp__[237] ^ temp__[300]; */
/* temp__[181] = x1__[61] ^ temp__[309]; */
/* temp__[180] = x0__[61] ^ temp__[309]; */
/* temp__[182] = temp__[180] & temp__[181]; */
/* break; */
/* case 62: */  
/* temp__[182].complete_and(); */
/* temp__[310] = temp__[182] ^ temp__[309]; */
/* y0__[62] = temp__[247] ^ temp__[310]; */
/* y0__[51] = temp__[236] ^ temp__[299]; */
/* temp__[184] = x1__[62] ^ temp__[310]; */
/* temp__[183] = x0__[62] ^ temp__[310]; */
/* temp__[185] = temp__[183] & temp__[184]; */
/* break; */
/* case 63: */
/* temp__[185].complete_and(); */ 
/* temp__[311] = temp__[185] ^ temp__[310]; */
/* y0__[63] = temp__[248] ^ temp__[311]; */
/* break; */
/* default: */
/* break; */
/* } */
/* r++; */
/* } */
};
