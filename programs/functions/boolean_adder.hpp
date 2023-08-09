#pragma once
#include "../../protocols/Protocols.h"
#include <cstring>
#include <iostream>

template<typename Pr, typename S>
class BooleanAdder {
private:
    int r;
    Pr P;
    S x0__[BITLENGTH];
    S x1__[BITLENGTH];
    S y0__[BITLENGTH];
    S temp__[312];
   
public:
//constructor
BooleanAdder(Pr &protocol, S x0[BITLENGTH], S x1[BITLENGTH], S y0[BITLENGTH])
{
P = protocol;
r = BITLENGTH;
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

void boolean_adder()
{
switch(r) {
case 0:
temp__[248] = P.Xor(x0__[63],x1__[63],FUNC_XOR);
temp__[247] = P.Xor(x0__[62],x1__[62],FUNC_XOR);
temp__[246] = P.Xor(x0__[61],x1__[61],FUNC_XOR);
temp__[245] = P.Xor(x0__[60],x1__[60],FUNC_XOR);
temp__[244] = P.Xor(x0__[59],x1__[59],FUNC_XOR);
temp__[243] = P.Xor(x0__[58],x1__[58],FUNC_XOR);
temp__[242] = P.Xor(x0__[57],x1__[57],FUNC_XOR);
temp__[241] = P.Xor(x0__[56],x1__[56],FUNC_XOR);
temp__[240] = P.Xor(x0__[55],x1__[55],FUNC_XOR);
temp__[239] = P.Xor(x0__[54],x1__[54],FUNC_XOR);
temp__[238] = P.Xor(x0__[53],x1__[53],FUNC_XOR);
temp__[237] = P.Xor(x0__[52],x1__[52],FUNC_XOR);
temp__[236] = P.Xor(x0__[51],x1__[51],FUNC_XOR);
temp__[235] = P.Xor(x0__[50],x1__[50],FUNC_XOR);
temp__[234] = P.Xor(x0__[49],x1__[49],FUNC_XOR);
temp__[233] = P.Xor(x0__[48],x1__[48],FUNC_XOR);
temp__[232] = P.Xor(x0__[47],x1__[47],FUNC_XOR);
temp__[231] = P.Xor(x0__[46],x1__[46],FUNC_XOR);
temp__[230] = P.Xor(x0__[45],x1__[45],FUNC_XOR);
temp__[229] = P.Xor(x0__[44],x1__[44],FUNC_XOR);
temp__[228] = P.Xor(x0__[43],x1__[43],FUNC_XOR);
temp__[227] = P.Xor(x0__[42],x1__[42],FUNC_XOR);
temp__[226] = P.Xor(x0__[41],x1__[41],FUNC_XOR);
temp__[225] = P.Xor(x0__[40],x1__[40],FUNC_XOR);
temp__[224] = P.Xor(x0__[39],x1__[39],FUNC_XOR);
temp__[223] = P.Xor(x0__[38],x1__[38],FUNC_XOR);
temp__[222] = P.Xor(x0__[37],x1__[37],FUNC_XOR);
temp__[221] = P.Xor(x0__[36],x1__[36],FUNC_XOR);
temp__[220] = P.Xor(x0__[35],x1__[35],FUNC_XOR);
temp__[219] = P.Xor(x0__[34],x1__[34],FUNC_XOR);
temp__[218] = P.Xor(x0__[33],x1__[33],FUNC_XOR);
temp__[217] = P.Xor(x0__[32],x1__[32],FUNC_XOR);
temp__[216] = P.Xor(x0__[31],x1__[31],FUNC_XOR);
temp__[215] = P.Xor(x0__[30],x1__[30],FUNC_XOR);
temp__[214] = P.Xor(x0__[29],x1__[29],FUNC_XOR);
temp__[213] = P.Xor(x0__[28],x1__[28],FUNC_XOR);
temp__[212] = P.Xor(x0__[27],x1__[27],FUNC_XOR);
temp__[211] = P.Xor(x0__[26],x1__[26],FUNC_XOR);
temp__[210] = P.Xor(x0__[25],x1__[25],FUNC_XOR);
temp__[209] = P.Xor(x0__[24],x1__[24],FUNC_XOR);
temp__[208] = P.Xor(x0__[23],x1__[23],FUNC_XOR);
temp__[207] = P.Xor(x0__[22],x1__[22],FUNC_XOR);
temp__[206] = P.Xor(x0__[21],x1__[21],FUNC_XOR);
temp__[205] = P.Xor(x0__[20],x1__[20],FUNC_XOR);
temp__[204] = P.Xor(x0__[19],x1__[19],FUNC_XOR);
temp__[203] = P.Xor(x0__[18],x1__[18],FUNC_XOR);
temp__[202] = P.Xor(x0__[17],x1__[17],FUNC_XOR);
temp__[201] = P.Xor(x0__[16],x1__[16],FUNC_XOR);
temp__[200] = P.Xor(x0__[15],x1__[15],FUNC_XOR);
temp__[199] = P.Xor(x0__[14],x1__[14],FUNC_XOR);
temp__[198] = P.Xor(x0__[13],x1__[13],FUNC_XOR);
temp__[197] = P.Xor(x0__[12],x1__[12],FUNC_XOR);
temp__[196] = P.Xor(x0__[11],x1__[11],FUNC_XOR);
temp__[195] = P.Xor(x0__[10],x1__[10],FUNC_XOR);
temp__[194] = P.Xor(x0__[9],x1__[9],FUNC_XOR);
temp__[193] = P.Xor(x0__[8],x1__[8],FUNC_XOR);
temp__[192] = P.Xor(x0__[7],x1__[7],FUNC_XOR);
temp__[191] = P.Xor(x0__[6],x1__[6],FUNC_XOR);
temp__[190] = P.Xor(x0__[5],x1__[5],FUNC_XOR);
temp__[189] = P.Xor(x0__[4],x1__[4],FUNC_XOR);
temp__[188] = P.Xor(x0__[3],x1__[3],FUNC_XOR);
temp__[187] = P.Xor(x0__[2],x1__[2],FUNC_XOR);
temp__[186] = P.Xor(x0__[1],x1__[1],FUNC_XOR);
y0__[0] = P.Xor(x0__[0],x1__[0],FUNC_XOR);
P.prepare_mult(x0__[0],x1__[0],temp__[249], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 1:
P.complete_mult(temp__[249]);
temp__[1] = P.Xor(x1__[1],temp__[249],FUNC_XOR);
temp__[0] = P.Xor(x0__[1],temp__[249],FUNC_XOR);
P.prepare_mult(temp__[0],temp__[1],temp__[2], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 2:
P.complete_mult(temp__[2]);
temp__[250] = P.Xor(temp__[2],temp__[249],FUNC_XOR);
temp__[4] = P.Xor(x1__[2],temp__[250],FUNC_XOR);
temp__[3] = P.Xor(x0__[2],temp__[250],FUNC_XOR);
P.prepare_mult(temp__[3],temp__[4],temp__[5], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 3:
P.complete_mult(temp__[5]);
temp__[251] = P.Xor(temp__[5],temp__[250],FUNC_XOR);
temp__[7] = P.Xor(x1__[3],temp__[251],FUNC_XOR);
temp__[6] = P.Xor(x0__[3],temp__[251],FUNC_XOR);
P.prepare_mult(temp__[6],temp__[7],temp__[8], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 4:
P.complete_mult(temp__[8]);
temp__[252] = P.Xor(temp__[8],temp__[251],FUNC_XOR);
temp__[10] = P.Xor(x1__[4],temp__[252],FUNC_XOR);
temp__[9] = P.Xor(x0__[4],temp__[252],FUNC_XOR);
P.prepare_mult(temp__[9],temp__[10],temp__[11], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 5:
P.complete_mult(temp__[11]);
temp__[253] = P.Xor(temp__[11],temp__[252],FUNC_XOR);
temp__[13] = P.Xor(x1__[5],temp__[253],FUNC_XOR);
temp__[12] = P.Xor(x0__[5],temp__[253],FUNC_XOR);
P.prepare_mult(temp__[12],temp__[13],temp__[14], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 6:
P.complete_mult(temp__[14]);
temp__[254] = P.Xor(temp__[14],temp__[253],FUNC_XOR);
temp__[16] = P.Xor(x1__[6],temp__[254],FUNC_XOR);
temp__[15] = P.Xor(x0__[6],temp__[254],FUNC_XOR);
P.prepare_mult(temp__[15],temp__[16],temp__[17], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 7:
P.complete_mult(temp__[17]);
temp__[255] = P.Xor(temp__[17],temp__[254],FUNC_XOR);
temp__[19] = P.Xor(x1__[7],temp__[255],FUNC_XOR);
temp__[18] = P.Xor(x0__[7],temp__[255],FUNC_XOR);
P.prepare_mult(temp__[18],temp__[19],temp__[20], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 8:
P.complete_mult(temp__[20]);
temp__[256] = P.Xor(temp__[20],temp__[255],FUNC_XOR);
temp__[22] = P.Xor(x1__[8],temp__[256],FUNC_XOR);
temp__[21] = P.Xor(x0__[8],temp__[256],FUNC_XOR);
P.prepare_mult(temp__[21],temp__[22],temp__[23], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 9:
P.complete_mult(temp__[23]);
temp__[257] = P.Xor(temp__[23],temp__[256],FUNC_XOR);
temp__[25] = P.Xor(x1__[9],temp__[257],FUNC_XOR);
temp__[24] = P.Xor(x0__[9],temp__[257],FUNC_XOR);
P.prepare_mult(temp__[24],temp__[25],temp__[26], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 10:
P.complete_mult(temp__[26]);
temp__[258] = P.Xor(temp__[26],temp__[257],FUNC_XOR);
temp__[28] = P.Xor(x1__[10],temp__[258],FUNC_XOR);
temp__[27] = P.Xor(x0__[10],temp__[258],FUNC_XOR);
P.prepare_mult(temp__[27],temp__[28],temp__[29], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 11:
P.complete_mult(temp__[29]);
temp__[259] = P.Xor(temp__[29],temp__[258],FUNC_XOR);
temp__[31] = P.Xor(x1__[11],temp__[259],FUNC_XOR);
temp__[30] = P.Xor(x0__[11],temp__[259],FUNC_XOR);
P.prepare_mult(temp__[30],temp__[31],temp__[32], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 12:
P.complete_mult(temp__[32]);
temp__[260] = P.Xor(temp__[32],temp__[259],FUNC_XOR);
temp__[34] = P.Xor(x1__[12],temp__[260],FUNC_XOR);
temp__[33] = P.Xor(x0__[12],temp__[260],FUNC_XOR);
P.prepare_mult(temp__[33],temp__[34],temp__[35], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 13:
P.complete_mult(temp__[35]);
temp__[261] = P.Xor(temp__[35],temp__[260],FUNC_XOR);
temp__[37] = P.Xor(x1__[13],temp__[261],FUNC_XOR);
temp__[36] = P.Xor(x0__[13],temp__[261],FUNC_XOR);
P.prepare_mult(temp__[36],temp__[37],temp__[38], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 14:
P.complete_mult(temp__[38]);
temp__[262] = P.Xor(temp__[38],temp__[261],FUNC_XOR);
temp__[40] = P.Xor(x1__[14],temp__[262],FUNC_XOR);
temp__[39] = P.Xor(x0__[14],temp__[262],FUNC_XOR);
P.prepare_mult(temp__[39],temp__[40],temp__[41], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 15:
P.complete_mult(temp__[41]);
temp__[263] = P.Xor(temp__[41],temp__[262],FUNC_XOR);
temp__[43] = P.Xor(x1__[15],temp__[263],FUNC_XOR);
temp__[42] = P.Xor(x0__[15],temp__[263],FUNC_XOR);
P.prepare_mult(temp__[42],temp__[43],temp__[44], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 16:
P.complete_mult(temp__[44]);
temp__[264] = P.Xor(temp__[44],temp__[263],FUNC_XOR);
temp__[46] = P.Xor(x1__[16],temp__[264],FUNC_XOR);
temp__[45] = P.Xor(x0__[16],temp__[264],FUNC_XOR);
P.prepare_mult(temp__[45],temp__[46],temp__[47], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 17:
P.complete_mult(temp__[47]);
temp__[265] = P.Xor(temp__[47],temp__[264],FUNC_XOR);
temp__[49] = P.Xor(x1__[17],temp__[265],FUNC_XOR);
temp__[48] = P.Xor(x0__[17],temp__[265],FUNC_XOR);
P.prepare_mult(temp__[48],temp__[49],temp__[50], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 18:
P.complete_mult(temp__[50]);
temp__[266] = P.Xor(temp__[50],temp__[265],FUNC_XOR);
temp__[52] = P.Xor(x1__[18],temp__[266],FUNC_XOR);
temp__[51] = P.Xor(x0__[18],temp__[266],FUNC_XOR);
P.prepare_mult(temp__[51],temp__[52],temp__[53], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 19:
P.complete_mult(temp__[53]);
temp__[267] = P.Xor(temp__[53],temp__[266],FUNC_XOR);
temp__[55] = P.Xor(x1__[19],temp__[267],FUNC_XOR);
temp__[54] = P.Xor(x0__[19],temp__[267],FUNC_XOR);
P.prepare_mult(temp__[54],temp__[55],temp__[56], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 20:
P.complete_mult(temp__[56]);
temp__[268] = P.Xor(temp__[56],temp__[267],FUNC_XOR);
temp__[58] = P.Xor(x1__[20],temp__[268],FUNC_XOR);
temp__[57] = P.Xor(x0__[20],temp__[268],FUNC_XOR);
P.prepare_mult(temp__[57],temp__[58],temp__[59], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 21:
P.complete_mult(temp__[59]);
temp__[269] = P.Xor(temp__[59],temp__[268],FUNC_XOR);
temp__[61] = P.Xor(x1__[21],temp__[269],FUNC_XOR);
temp__[60] = P.Xor(x0__[21],temp__[269],FUNC_XOR);
P.prepare_mult(temp__[60],temp__[61],temp__[62], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 22:
P.complete_mult(temp__[62]);
temp__[270] = P.Xor(temp__[62],temp__[269],FUNC_XOR);
temp__[64] = P.Xor(x1__[22],temp__[270],FUNC_XOR);
temp__[63] = P.Xor(x0__[22],temp__[270],FUNC_XOR);
P.prepare_mult(temp__[63],temp__[64],temp__[65], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 23:
P.complete_mult(temp__[65]);
temp__[271] = P.Xor(temp__[65],temp__[270],FUNC_XOR);
temp__[67] = P.Xor(x1__[23],temp__[271],FUNC_XOR);
temp__[66] = P.Xor(x0__[23],temp__[271],FUNC_XOR);
P.prepare_mult(temp__[66],temp__[67],temp__[68], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 24:
P.complete_mult(temp__[68]);
temp__[272] = P.Xor(temp__[68],temp__[271],FUNC_XOR);
temp__[70] = P.Xor(x1__[24],temp__[272],FUNC_XOR);
temp__[69] = P.Xor(x0__[24],temp__[272],FUNC_XOR);
P.prepare_mult(temp__[69],temp__[70],temp__[71], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 25:
P.complete_mult(temp__[71]);
temp__[273] = P.Xor(temp__[71],temp__[272],FUNC_XOR);
temp__[73] = P.Xor(x1__[25],temp__[273],FUNC_XOR);
temp__[72] = P.Xor(x0__[25],temp__[273],FUNC_XOR);
P.prepare_mult(temp__[72],temp__[73],temp__[74], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 26:
P.complete_mult(temp__[74]);
temp__[274] = P.Xor(temp__[74],temp__[273],FUNC_XOR);
temp__[76] = P.Xor(x1__[26],temp__[274],FUNC_XOR);
temp__[75] = P.Xor(x0__[26],temp__[274],FUNC_XOR);
P.prepare_mult(temp__[75],temp__[76],temp__[77], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 27:
P.complete_mult(temp__[77]);
temp__[275] = P.Xor(temp__[77],temp__[274],FUNC_XOR);
temp__[79] = P.Xor(x1__[27],temp__[275],FUNC_XOR);
temp__[78] = P.Xor(x0__[27],temp__[275],FUNC_XOR);
P.prepare_mult(temp__[78],temp__[79],temp__[80], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 28:
P.complete_mult(temp__[80]);
temp__[276] = P.Xor(temp__[80],temp__[275],FUNC_XOR);
y0__[28] = P.Xor(temp__[213],temp__[276],FUNC_XOR);
temp__[82] = P.Xor(x1__[28],temp__[276],FUNC_XOR);
temp__[81] = P.Xor(x0__[28],temp__[276],FUNC_XOR);
P.prepare_mult(temp__[81],temp__[82],temp__[83], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 29:
P.complete_mult(temp__[83]);
temp__[277] = P.Xor(temp__[83],temp__[276],FUNC_XOR);
y0__[29] = P.Xor(temp__[214],temp__[277],FUNC_XOR);
y0__[27] = P.Xor(temp__[212],temp__[275],FUNC_XOR);
temp__[85] = P.Xor(x1__[29],temp__[277],FUNC_XOR);
temp__[84] = P.Xor(x0__[29],temp__[277],FUNC_XOR);
P.prepare_mult(temp__[84],temp__[85],temp__[86], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 30:
P.complete_mult(temp__[86]);
temp__[278] = P.Xor(temp__[86],temp__[277],FUNC_XOR);
y0__[30] = P.Xor(temp__[215],temp__[278],FUNC_XOR);
y0__[26] = P.Xor(temp__[211],temp__[274],FUNC_XOR);
temp__[88] = P.Xor(x1__[30],temp__[278],FUNC_XOR);
temp__[87] = P.Xor(x0__[30],temp__[278],FUNC_XOR);
P.prepare_mult(temp__[87],temp__[88],temp__[89], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 31:
P.complete_mult(temp__[89]);
temp__[279] = P.Xor(temp__[89],temp__[278],FUNC_XOR);
y0__[25] = P.Xor(temp__[210],temp__[273],FUNC_XOR);
temp__[90] = P.Xor(x0__[31],temp__[279],FUNC_XOR);
y0__[31] = P.Xor(temp__[216],temp__[279],FUNC_XOR);
y0__[24] = P.Xor(temp__[209],temp__[272],FUNC_XOR);
temp__[91] = P.Xor(x1__[31],temp__[279],FUNC_XOR);
P.prepare_mult(temp__[90],temp__[91],temp__[92], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 32:
P.complete_mult(temp__[92]);
temp__[280] = P.Xor(temp__[92],temp__[279],FUNC_XOR);
y0__[32] = P.Xor(temp__[217],temp__[280],FUNC_XOR);
y0__[23] = P.Xor(temp__[208],temp__[271],FUNC_XOR);
temp__[94] = P.Xor(x1__[32],temp__[280],FUNC_XOR);
temp__[93] = P.Xor(x0__[32],temp__[280],FUNC_XOR);
P.prepare_mult(temp__[93],temp__[94],temp__[95], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 33:
P.complete_mult(temp__[95]);
temp__[281] = P.Xor(temp__[95],temp__[280],FUNC_XOR);
y0__[33] = P.Xor(temp__[218],temp__[281],FUNC_XOR);
y0__[22] = P.Xor(temp__[207],temp__[270],FUNC_XOR);
temp__[97] = P.Xor(x1__[33],temp__[281],FUNC_XOR);
temp__[96] = P.Xor(x0__[33],temp__[281],FUNC_XOR);
P.prepare_mult(temp__[96],temp__[97],temp__[98], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 34:
P.complete_mult(temp__[98]);
temp__[282] = P.Xor(temp__[98],temp__[281],FUNC_XOR);
y0__[34] = P.Xor(temp__[219],temp__[282],FUNC_XOR);
y0__[21] = P.Xor(temp__[206],temp__[269],FUNC_XOR);
temp__[100] = P.Xor(x1__[34],temp__[282],FUNC_XOR);
temp__[99] = P.Xor(x0__[34],temp__[282],FUNC_XOR);
P.prepare_mult(temp__[99],temp__[100],temp__[101], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 35:
P.complete_mult(temp__[101]);
temp__[283] = P.Xor(temp__[101],temp__[282],FUNC_XOR);
y0__[20] = P.Xor(temp__[205],temp__[268],FUNC_XOR);
temp__[102] = P.Xor(x0__[35],temp__[283],FUNC_XOR);
y0__[35] = P.Xor(temp__[220],temp__[283],FUNC_XOR);
y0__[19] = P.Xor(temp__[204],temp__[267],FUNC_XOR);
temp__[103] = P.Xor(x1__[35],temp__[283],FUNC_XOR);
P.prepare_mult(temp__[102],temp__[103],temp__[104], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 36:
P.complete_mult(temp__[104]);
temp__[284] = P.Xor(temp__[104],temp__[283],FUNC_XOR);
y0__[36] = P.Xor(temp__[221],temp__[284],FUNC_XOR);
y0__[18] = P.Xor(temp__[203],temp__[266],FUNC_XOR);
temp__[106] = P.Xor(x1__[36],temp__[284],FUNC_XOR);
temp__[105] = P.Xor(x0__[36],temp__[284],FUNC_XOR);
P.prepare_mult(temp__[105],temp__[106],temp__[107], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 37:
P.complete_mult(temp__[107]);
temp__[285] = P.Xor(temp__[107],temp__[284],FUNC_XOR);
y0__[37] = P.Xor(temp__[222],temp__[285],FUNC_XOR);
y0__[17] = P.Xor(temp__[202],temp__[265],FUNC_XOR);
temp__[109] = P.Xor(x1__[37],temp__[285],FUNC_XOR);
temp__[108] = P.Xor(x0__[37],temp__[285],FUNC_XOR);
P.prepare_mult(temp__[108],temp__[109],temp__[110], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 38:
P.complete_mult(temp__[110]);
temp__[286] = P.Xor(temp__[110],temp__[285],FUNC_XOR);
y0__[38] = P.Xor(temp__[223],temp__[286],FUNC_XOR);
y0__[16] = P.Xor(temp__[201],temp__[264],FUNC_XOR);
temp__[112] = P.Xor(x1__[38],temp__[286],FUNC_XOR);
temp__[111] = P.Xor(x0__[38],temp__[286],FUNC_XOR);
P.prepare_mult(temp__[111],temp__[112],temp__[113], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 39:
P.complete_mult(temp__[113]);
temp__[287] = P.Xor(temp__[113],temp__[286],FUNC_XOR);
y0__[15] = P.Xor(temp__[200],temp__[263],FUNC_XOR);
temp__[114] = P.Xor(x0__[39],temp__[287],FUNC_XOR);
y0__[39] = P.Xor(temp__[224],temp__[287],FUNC_XOR);
y0__[14] = P.Xor(temp__[199],temp__[262],FUNC_XOR);
temp__[115] = P.Xor(x1__[39],temp__[287],FUNC_XOR);
P.prepare_mult(temp__[114],temp__[115],temp__[116], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 40:
P.complete_mult(temp__[116]);
temp__[288] = P.Xor(temp__[116],temp__[287],FUNC_XOR);
y0__[40] = P.Xor(temp__[225],temp__[288],FUNC_XOR);
y0__[13] = P.Xor(temp__[198],temp__[261],FUNC_XOR);
temp__[118] = P.Xor(x1__[40],temp__[288],FUNC_XOR);
temp__[117] = P.Xor(x0__[40],temp__[288],FUNC_XOR);
P.prepare_mult(temp__[117],temp__[118],temp__[119], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 41:
P.complete_mult(temp__[119]);
temp__[289] = P.Xor(temp__[119],temp__[288],FUNC_XOR);
y0__[41] = P.Xor(temp__[226],temp__[289],FUNC_XOR);
y0__[12] = P.Xor(temp__[197],temp__[260],FUNC_XOR);
temp__[121] = P.Xor(x1__[41],temp__[289],FUNC_XOR);
temp__[120] = P.Xor(x0__[41],temp__[289],FUNC_XOR);
P.prepare_mult(temp__[120],temp__[121],temp__[122], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 42:
P.complete_mult(temp__[122]);
temp__[290] = P.Xor(temp__[122],temp__[289],FUNC_XOR);
y0__[42] = P.Xor(temp__[227],temp__[290],FUNC_XOR);
y0__[11] = P.Xor(temp__[196],temp__[259],FUNC_XOR);
temp__[124] = P.Xor(x1__[42],temp__[290],FUNC_XOR);
temp__[123] = P.Xor(x0__[42],temp__[290],FUNC_XOR);
P.prepare_mult(temp__[123],temp__[124],temp__[125], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 43:
P.complete_mult(temp__[125]);
temp__[291] = P.Xor(temp__[125],temp__[290],FUNC_XOR);
y0__[10] = P.Xor(temp__[195],temp__[258],FUNC_XOR);
temp__[126] = P.Xor(x0__[43],temp__[291],FUNC_XOR);
y0__[43] = P.Xor(temp__[228],temp__[291],FUNC_XOR);
y0__[9] = P.Xor(temp__[194],temp__[257],FUNC_XOR);
temp__[127] = P.Xor(x1__[43],temp__[291],FUNC_XOR);
P.prepare_mult(temp__[126],temp__[127],temp__[128], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 44:
P.complete_mult(temp__[128]);
temp__[292] = P.Xor(temp__[128],temp__[291],FUNC_XOR);
y0__[44] = P.Xor(temp__[229],temp__[292],FUNC_XOR);
y0__[8] = P.Xor(temp__[193],temp__[256],FUNC_XOR);
temp__[130] = P.Xor(x1__[44],temp__[292],FUNC_XOR);
temp__[129] = P.Xor(x0__[44],temp__[292],FUNC_XOR);
P.prepare_mult(temp__[129],temp__[130],temp__[131], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 45:
P.complete_mult(temp__[131]);
temp__[293] = P.Xor(temp__[131],temp__[292],FUNC_XOR);
y0__[45] = P.Xor(temp__[230],temp__[293],FUNC_XOR);
y0__[7] = P.Xor(temp__[192],temp__[255],FUNC_XOR);
temp__[133] = P.Xor(x1__[45],temp__[293],FUNC_XOR);
temp__[132] = P.Xor(x0__[45],temp__[293],FUNC_XOR);
P.prepare_mult(temp__[132],temp__[133],temp__[134], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 46:
P.complete_mult(temp__[134]);
temp__[294] = P.Xor(temp__[134],temp__[293],FUNC_XOR);
y0__[46] = P.Xor(temp__[231],temp__[294],FUNC_XOR);
y0__[6] = P.Xor(temp__[191],temp__[254],FUNC_XOR);
temp__[136] = P.Xor(x1__[46],temp__[294],FUNC_XOR);
temp__[135] = P.Xor(x0__[46],temp__[294],FUNC_XOR);
P.prepare_mult(temp__[135],temp__[136],temp__[137], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 47:
P.complete_mult(temp__[137]);
temp__[295] = P.Xor(temp__[137],temp__[294],FUNC_XOR);
y0__[5] = P.Xor(temp__[190],temp__[253],FUNC_XOR);
temp__[138] = P.Xor(x0__[47],temp__[295],FUNC_XOR);
y0__[47] = P.Xor(temp__[232],temp__[295],FUNC_XOR);
y0__[4] = P.Xor(temp__[189],temp__[252],FUNC_XOR);
temp__[139] = P.Xor(x1__[47],temp__[295],FUNC_XOR);
P.prepare_mult(temp__[138],temp__[139],temp__[140], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 48:
P.complete_mult(temp__[140]);
temp__[296] = P.Xor(temp__[140],temp__[295],FUNC_XOR);
y0__[48] = P.Xor(temp__[233],temp__[296],FUNC_XOR);
y0__[3] = P.Xor(temp__[188],temp__[251],FUNC_XOR);
temp__[142] = P.Xor(x1__[48],temp__[296],FUNC_XOR);
temp__[141] = P.Xor(x0__[48],temp__[296],FUNC_XOR);
P.prepare_mult(temp__[141],temp__[142],temp__[143], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 49:
P.complete_mult(temp__[143]);
temp__[297] = P.Xor(temp__[143],temp__[296],FUNC_XOR);
y0__[49] = P.Xor(temp__[234],temp__[297],FUNC_XOR);
y0__[2] = P.Xor(temp__[187],temp__[250],FUNC_XOR);
temp__[145] = P.Xor(x1__[49],temp__[297],FUNC_XOR);
temp__[144] = P.Xor(x0__[49],temp__[297],FUNC_XOR);
P.prepare_mult(temp__[144],temp__[145],temp__[146], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 50:
P.complete_mult(temp__[146]);
temp__[298] = P.Xor(temp__[146],temp__[297],FUNC_XOR);
y0__[50] = P.Xor(temp__[235],temp__[298],FUNC_XOR);
y0__[1] = P.Xor(temp__[186],temp__[249],FUNC_XOR);
temp__[148] = P.Xor(x1__[50],temp__[298],FUNC_XOR);
temp__[147] = P.Xor(x0__[50],temp__[298],FUNC_XOR);
P.prepare_mult(temp__[147],temp__[148],temp__[149], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 51:
P.complete_mult(temp__[149]);
temp__[299] = P.Xor(temp__[149],temp__[298],FUNC_XOR);
temp__[151] = P.Xor(x1__[51],temp__[299],FUNC_XOR);
temp__[150] = P.Xor(x0__[51],temp__[299],FUNC_XOR);
P.prepare_mult(temp__[150],temp__[151],temp__[152], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 52:
P.complete_mult(temp__[152]);
temp__[300] = P.Xor(temp__[152],temp__[299],FUNC_XOR);
temp__[154] = P.Xor(x1__[52],temp__[300],FUNC_XOR);
temp__[153] = P.Xor(x0__[52],temp__[300],FUNC_XOR);
P.prepare_mult(temp__[153],temp__[154],temp__[155], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 53:
P.complete_mult(temp__[155]);
temp__[301] = P.Xor(temp__[155],temp__[300],FUNC_XOR);
temp__[157] = P.Xor(x1__[53],temp__[301],FUNC_XOR);
temp__[156] = P.Xor(x0__[53],temp__[301],FUNC_XOR);
P.prepare_mult(temp__[156],temp__[157],temp__[158], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 54:
P.complete_mult(temp__[158]);
temp__[302] = P.Xor(temp__[158],temp__[301],FUNC_XOR);
temp__[160] = P.Xor(x1__[54],temp__[302],FUNC_XOR);
temp__[159] = P.Xor(x0__[54],temp__[302],FUNC_XOR);
P.prepare_mult(temp__[159],temp__[160],temp__[161], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 55:
P.complete_mult(temp__[161]);
temp__[303] = P.Xor(temp__[161],temp__[302],FUNC_XOR);
temp__[163] = P.Xor(x1__[55],temp__[303],FUNC_XOR);
temp__[162] = P.Xor(x0__[55],temp__[303],FUNC_XOR);
P.prepare_mult(temp__[162],temp__[163],temp__[164], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 56:
P.complete_mult(temp__[164]);
temp__[304] = P.Xor(temp__[164],temp__[303],FUNC_XOR);
temp__[166] = P.Xor(x1__[56],temp__[304],FUNC_XOR);
temp__[165] = P.Xor(x0__[56],temp__[304],FUNC_XOR);
P.prepare_mult(temp__[165],temp__[166],temp__[167], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 57:
P.complete_mult(temp__[167]);
temp__[305] = P.Xor(temp__[167],temp__[304],FUNC_XOR);
y0__[57] = P.Xor(temp__[242],temp__[305],FUNC_XOR);
temp__[169] = P.Xor(x1__[57],temp__[305],FUNC_XOR);
temp__[168] = P.Xor(x0__[57],temp__[305],FUNC_XOR);
P.prepare_mult(temp__[168],temp__[169],temp__[170], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 58:
P.complete_mult(temp__[170]);
temp__[306] = P.Xor(temp__[170],temp__[305],FUNC_XOR);
y0__[58] = P.Xor(temp__[243],temp__[306],FUNC_XOR);
y0__[56] = P.Xor(temp__[241],temp__[304],FUNC_XOR);
temp__[172] = P.Xor(x1__[58],temp__[306],FUNC_XOR);
temp__[171] = P.Xor(x0__[58],temp__[306],FUNC_XOR);
P.prepare_mult(temp__[171],temp__[172],temp__[173], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 59:
P.complete_mult(temp__[173]);
temp__[307] = P.Xor(temp__[173],temp__[306],FUNC_XOR);
y0__[59] = P.Xor(temp__[244],temp__[307],FUNC_XOR);
y0__[55] = P.Xor(temp__[240],temp__[303],FUNC_XOR);
temp__[175] = P.Xor(x1__[59],temp__[307],FUNC_XOR);
temp__[174] = P.Xor(x0__[59],temp__[307],FUNC_XOR);
P.prepare_mult(temp__[174],temp__[175],temp__[176], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 60:
P.complete_mult(temp__[176]);
temp__[308] = P.Xor(temp__[176],temp__[307],FUNC_XOR);
y0__[54] = P.Xor(temp__[239],temp__[302],FUNC_XOR);
temp__[177] = P.Xor(x0__[60],temp__[308],FUNC_XOR);
y0__[60] = P.Xor(temp__[245],temp__[308],FUNC_XOR);
y0__[53] = P.Xor(temp__[238],temp__[301],FUNC_XOR);
temp__[178] = P.Xor(x1__[60],temp__[308],FUNC_XOR);
P.prepare_mult(temp__[177],temp__[178],temp__[179], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 61:
P.complete_mult(temp__[179]);
temp__[309] = P.Xor(temp__[179],temp__[308],FUNC_XOR);
y0__[61] = P.Xor(temp__[246],temp__[309],FUNC_XOR);
y0__[52] = P.Xor(temp__[237],temp__[300],FUNC_XOR);
temp__[181] = P.Xor(x1__[61],temp__[309],FUNC_XOR);
temp__[180] = P.Xor(x0__[61],temp__[309],FUNC_XOR);
P.prepare_mult(temp__[180],temp__[181],temp__[182], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 62:
P.complete_mult(temp__[182]);
temp__[310] = P.Xor(temp__[182],temp__[309],FUNC_XOR);
y0__[62] = P.Xor(temp__[247],temp__[310],FUNC_XOR);
y0__[51] = P.Xor(temp__[236],temp__[299],FUNC_XOR);
temp__[184] = P.Xor(x1__[62],temp__[310],FUNC_XOR);
temp__[183] = P.Xor(x0__[62],temp__[310],FUNC_XOR);
P.prepare_mult(temp__[183],temp__[184],temp__[185], FUNC_XOR, FUNC_XOR, FUNC_AND);
case 63:
P.complete_mult(temp__[185]);
temp__[311] = P.Xor(temp__[185],temp__[310],FUNC_XOR);
y0__[63] = P.Xor(temp__[248],temp__[311],FUNC_XOR);
}
r = r++;
}
};
