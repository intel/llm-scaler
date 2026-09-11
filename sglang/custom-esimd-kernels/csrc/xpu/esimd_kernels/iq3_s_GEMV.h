/* iq3_s_GEMV.h — canonical GGUF IQ3_S GEMV for Intel XPU.
 *
 * SGLang normalizes block_iq3_s to:
 *   input   [M,K]    fp16
 *   qs      [N,K/4]  uint8: low eight bits of one 9-bit grid index per 4 elems
 *   qh      [N,K/32] uint8: high grid-index bits, LSB-first
 *   signs   [N,K/8]  uint8: one sign bit per element, LSB-first
 *   scale   [N,K/32] fp16: final d * (1 + 2 * scale_nibble)
 *   output  [M,N]    fp16
 *
 * The GGML IQ3_S grid holds 512 groups of four unsigned 4-bit magnitudes.
 * weight[k] = sign[k] * scale[k/32] * grid[index[k/4]][k%4].
 */
#pragma once

namespace iq3s_esimd_detail = sycl::ext::intel::esimd::detail;

static constexpr int IQ3S_GROUP = 32;
static constexpr int IQ3S_VL = 512;
static constexpr int IQ3S_ROWS = 4;

template <int VL>
SYCL_ESIMD_FUNCTION inline simd<float, VL> iq3s_dequant_tile(
    simd<uint8_t, VL / 4> qs_data,
    simd<uint8_t, VL / 32> qh_data,
    simd<uint8_t, VL / 8> signs_data,
    simd<fp16, VL / IQ3S_GROUP> scale_h) {
    static_assert(VL % 256 == 0);

    // Each packed uint16 is four consecutive 4-bit magnitudes in the exact
    // GGML IQ3_S grid order. Keep the 512-entry table in the kernel instead
    // of materializing a [N,K] magnitude tensor in resident memory.
    simd<uint16_t, 512> lut;
        lut[0] = uint16_t(0x1111);
        lut[1] = uint16_t(0x1113);
        lut[2] = uint16_t(0x1115);
        lut[3] = uint16_t(0x111b);
        lut[4] = uint16_t(0x111f);
        lut[5] = uint16_t(0x1131);
        lut[6] = uint16_t(0x1133);
        lut[7] = uint16_t(0x1135);
        lut[8] = uint16_t(0x1139);
        lut[9] = uint16_t(0x113d);
        lut[10] = uint16_t(0x1151);
        lut[11] = uint16_t(0x1153);
        lut[12] = uint16_t(0x115b);
        lut[13] = uint16_t(0x1177);
        lut[14] = uint16_t(0x1191);
        lut[15] = uint16_t(0x1195);
        lut[16] = uint16_t(0x119b);
        lut[17] = uint16_t(0x119f);
        lut[18] = uint16_t(0x11b3);
        lut[19] = uint16_t(0x11b7);
        lut[20] = uint16_t(0x11d1);
        lut[21] = uint16_t(0x11d5);
        lut[22] = uint16_t(0x11f3);
        lut[23] = uint16_t(0x11f9);
        lut[24] = uint16_t(0x11ff);
        lut[25] = uint16_t(0x1311);
        lut[26] = uint16_t(0x1313);
        lut[27] = uint16_t(0x1315);
        lut[28] = uint16_t(0x1319);
        lut[29] = uint16_t(0x1331);
        lut[30] = uint16_t(0x1333);
        lut[31] = uint16_t(0x133b);
        lut[32] = uint16_t(0x1351);
        lut[33] = uint16_t(0x1357);
        lut[34] = uint16_t(0x135f);
        lut[35] = uint16_t(0x1373);
        lut[36] = uint16_t(0x137b);
        lut[37] = uint16_t(0x1399);
        lut[38] = uint16_t(0x13d3);
        lut[39] = uint16_t(0x13db);
        lut[40] = uint16_t(0x13f5);
        lut[41] = uint16_t(0x1511);
        lut[42] = uint16_t(0x1513);
        lut[43] = uint16_t(0x151b);
        lut[44] = uint16_t(0x151f);
        lut[45] = uint16_t(0x1531);
        lut[46] = uint16_t(0x1537);
        lut[47] = uint16_t(0x153d);
        lut[48] = uint16_t(0x1553);
        lut[49] = uint16_t(0x155b);
        lut[50] = uint16_t(0x1571);
        lut[51] = uint16_t(0x1579);
        lut[52] = uint16_t(0x1595);
        lut[53] = uint16_t(0x159b);
        lut[54] = uint16_t(0x159f);
        lut[55] = uint16_t(0x15b3);
        lut[56] = uint16_t(0x15b7);
        lut[57] = uint16_t(0x15f1);
        lut[58] = uint16_t(0x15f7);
        lut[59] = uint16_t(0x1717);
        lut[60] = uint16_t(0x1733);
        lut[61] = uint16_t(0x173b);
        lut[62] = uint16_t(0x1751);
        lut[63] = uint16_t(0x1755);
        lut[64] = uint16_t(0x1773);
        lut[65] = uint16_t(0x1777);
        lut[66] = uint16_t(0x177d);
        lut[67] = uint16_t(0x1799);
        lut[68] = uint16_t(0x17b1);
        lut[69] = uint16_t(0x17b5);
        lut[70] = uint16_t(0x17df);
        lut[71] = uint16_t(0x17f3);
        lut[72] = uint16_t(0x17fb);
        lut[73] = uint16_t(0x1911);
        lut[74] = uint16_t(0x1937);
        lut[75] = uint16_t(0x193f);
        lut[76] = uint16_t(0x1953);
        lut[77] = uint16_t(0x1959);
        lut[78] = uint16_t(0x1975);
        lut[79] = uint16_t(0x1991);
        lut[80] = uint16_t(0x1997);
        lut[81] = uint16_t(0x19b3);
        lut[82] = uint16_t(0x19f1);
        lut[83] = uint16_t(0x1b15);
        lut[84] = uint16_t(0x1b19);
        lut[85] = uint16_t(0x1b51);
        lut[86] = uint16_t(0x1b55);
        lut[87] = uint16_t(0x1b5d);
        lut[88] = uint16_t(0x1b77);
        lut[89] = uint16_t(0x1b93);
        lut[90] = uint16_t(0x1b9b);
        lut[91] = uint16_t(0x1b9f);
        lut[92] = uint16_t(0x1bdd);
        lut[93] = uint16_t(0x1bf7);
        lut[94] = uint16_t(0x1d1d);
        lut[95] = uint16_t(0x1d33);
        lut[96] = uint16_t(0x1d37);
        lut[97] = uint16_t(0x1d73);
        lut[98] = uint16_t(0x1db5);
        lut[99] = uint16_t(0x1df3);
        lut[100] = uint16_t(0x1f11);
        lut[101] = uint16_t(0x1f15);
        lut[102] = uint16_t(0x1f19);
        lut[103] = uint16_t(0x1f51);
        lut[104] = uint16_t(0x1f55);
        lut[105] = uint16_t(0x1f5d);
        lut[106] = uint16_t(0x1f77);
        lut[107] = uint16_t(0x1fb1);
        lut[108] = uint16_t(0x1fb9);
        lut[109] = uint16_t(0x3111);
        lut[110] = uint16_t(0x3113);
        lut[111] = uint16_t(0x3115);
        lut[112] = uint16_t(0x3119);
        lut[113] = uint16_t(0x3131);
        lut[114] = uint16_t(0x3133);
        lut[115] = uint16_t(0x3137);
        lut[116] = uint16_t(0x313b);
        lut[117] = uint16_t(0x313f);
        lut[118] = uint16_t(0x3151);
        lut[119] = uint16_t(0x3155);
        lut[120] = uint16_t(0x3173);
        lut[121] = uint16_t(0x3179);
        lut[122] = uint16_t(0x317d);
        lut[123] = uint16_t(0x31b9);
        lut[124] = uint16_t(0x31bd);
        lut[125] = uint16_t(0x31d3);
        lut[126] = uint16_t(0x31f5);
        lut[127] = uint16_t(0x3311);
        lut[128] = uint16_t(0x3313);
        lut[129] = uint16_t(0x3317);
        lut[130] = uint16_t(0x331d);
        lut[131] = uint16_t(0x3331);
        lut[132] = uint16_t(0x3339);
        lut[133] = uint16_t(0x3353);
        lut[134] = uint16_t(0x3371);
        lut[135] = uint16_t(0x3377);
        lut[136] = uint16_t(0x3393);
        lut[137] = uint16_t(0x33b1);
        lut[138] = uint16_t(0x33b5);
        lut[139] = uint16_t(0x33f1);
        lut[140] = uint16_t(0x33fd);
        lut[141] = uint16_t(0x3511);
        lut[142] = uint16_t(0x3535);
        lut[143] = uint16_t(0x353b);
        lut[144] = uint16_t(0x353f);
        lut[145] = uint16_t(0x3551);
        lut[146] = uint16_t(0x3559);
        lut[147] = uint16_t(0x3575);
        lut[148] = uint16_t(0x3591);
        lut[149] = uint16_t(0x3597);
        lut[150] = uint16_t(0x35bb);
        lut[151] = uint16_t(0x35d1);
        lut[152] = uint16_t(0x35f5);
        lut[153] = uint16_t(0x3713);
        lut[154] = uint16_t(0x3719);
        lut[155] = uint16_t(0x371f);
        lut[156] = uint16_t(0x3731);
        lut[157] = uint16_t(0x3737);
        lut[158] = uint16_t(0x3753);
        lut[159] = uint16_t(0x375f);
        lut[160] = uint16_t(0x3771);
        lut[161] = uint16_t(0x3779);
        lut[162] = uint16_t(0x3793);
        lut[163] = uint16_t(0x37d5);
        lut[164] = uint16_t(0x37f1);
        lut[165] = uint16_t(0x3917);
        lut[166] = uint16_t(0x391b);
        lut[167] = uint16_t(0x3935);
        lut[168] = uint16_t(0x3939);
        lut[169] = uint16_t(0x3973);
        lut[170] = uint16_t(0x3977);
        lut[171] = uint16_t(0x3995);
        lut[172] = uint16_t(0x399d);
        lut[173] = uint16_t(0x39b1);
        lut[174] = uint16_t(0x39b9);
        lut[175] = uint16_t(0x3b13);
        lut[176] = uint16_t(0x3b31);
        lut[177] = uint16_t(0x3b37);
        lut[178] = uint16_t(0x3b53);
        lut[179] = uint16_t(0x3b71);
        lut[180] = uint16_t(0x3b75);
        lut[181] = uint16_t(0x3bb3);
        lut[182] = uint16_t(0x3d51);
        lut[183] = uint16_t(0x3d59);
        lut[184] = uint16_t(0x3d5f);
        lut[185] = uint16_t(0x3d99);
        lut[186] = uint16_t(0x3d9d);
        lut[187] = uint16_t(0x3f13);
        lut[188] = uint16_t(0x3f17);
        lut[189] = uint16_t(0x3f31);
        lut[190] = uint16_t(0x3f35);
        lut[191] = uint16_t(0x3f53);
        lut[192] = uint16_t(0x3f7b);
        lut[193] = uint16_t(0x3f93);
        lut[194] = uint16_t(0x3fd5);
        lut[195] = uint16_t(0x3ff1);
        lut[196] = uint16_t(0x5111);
        lut[197] = uint16_t(0x5113);
        lut[198] = uint16_t(0x5117);
        lut[199] = uint16_t(0x511b);
        lut[200] = uint16_t(0x511f);
        lut[201] = uint16_t(0x5131);
        lut[202] = uint16_t(0x5135);
        lut[203] = uint16_t(0x5139);
        lut[204] = uint16_t(0x513d);
        lut[205] = uint16_t(0x5153);
        lut[206] = uint16_t(0x5157);
        lut[207] = uint16_t(0x515f);
        lut[208] = uint16_t(0x5171);
        lut[209] = uint16_t(0x5175);
        lut[210] = uint16_t(0x5193);
        lut[211] = uint16_t(0x5197);
        lut[212] = uint16_t(0x519b);
        lut[213] = uint16_t(0x51b1);
        lut[214] = uint16_t(0x51b5);
        lut[215] = uint16_t(0x51df);
        lut[216] = uint16_t(0x51f1);
        lut[217] = uint16_t(0x51f7);
        lut[218] = uint16_t(0x51fb);
        lut[219] = uint16_t(0x5311);
        lut[220] = uint16_t(0x5315);
        lut[221] = uint16_t(0x5331);
        lut[222] = uint16_t(0x5337);
        lut[223] = uint16_t(0x533f);
        lut[224] = uint16_t(0x5355);
        lut[225] = uint16_t(0x535b);
        lut[226] = uint16_t(0x5373);
        lut[227] = uint16_t(0x5379);
        lut[228] = uint16_t(0x5395);
        lut[229] = uint16_t(0x53b3);
        lut[230] = uint16_t(0x5513);
        lut[231] = uint16_t(0x5519);
        lut[232] = uint16_t(0x551f);
        lut[233] = uint16_t(0x5553);
        lut[234] = uint16_t(0x5557);
        lut[235] = uint16_t(0x5571);
        lut[236] = uint16_t(0x557f);
        lut[237] = uint16_t(0x5593);
        lut[238] = uint16_t(0x55b7);
        lut[239] = uint16_t(0x55bf);
        lut[240] = uint16_t(0x55f3);
        lut[241] = uint16_t(0x55f9);
        lut[242] = uint16_t(0x5711);
        lut[243] = uint16_t(0x5715);
        lut[244] = uint16_t(0x571b);
        lut[245] = uint16_t(0x5733);
        lut[246] = uint16_t(0x5755);
        lut[247] = uint16_t(0x5759);
        lut[248] = uint16_t(0x5773);
        lut[249] = uint16_t(0x5777);
        lut[250] = uint16_t(0x5795);
        lut[251] = uint16_t(0x57b1);
        lut[252] = uint16_t(0x57dd);
        lut[253] = uint16_t(0x5913);
        lut[254] = uint16_t(0x591f);
        lut[255] = uint16_t(0x5951);
        lut[256] = uint16_t(0x5957);
        lut[257] = uint16_t(0x5975);
        lut[258] = uint16_t(0x597b);
        lut[259] = uint16_t(0x5993);
        lut[260] = uint16_t(0x59f5);
        lut[261] = uint16_t(0x59fb);
        lut[262] = uint16_t(0x5b19);
        lut[263] = uint16_t(0x5b33);
        lut[264] = uint16_t(0x5b55);
        lut[265] = uint16_t(0x5b7f);
        lut[266] = uint16_t(0x5b91);
        lut[267] = uint16_t(0x5bb7);
        lut[268] = uint16_t(0x5bf1);
        lut[269] = uint16_t(0x5d11);
        lut[270] = uint16_t(0x5d15);
        lut[271] = uint16_t(0x5d1f);
        lut[272] = uint16_t(0x5d53);
        lut[273] = uint16_t(0x5dbb);
        lut[274] = uint16_t(0x5dd3);
        lut[275] = uint16_t(0x5f1b);
        lut[276] = uint16_t(0x5f33);
        lut[277] = uint16_t(0x5f5d);
        lut[278] = uint16_t(0x5f71);
        lut[279] = uint16_t(0x5f97);
        lut[280] = uint16_t(0x5fb1);
        lut[281] = uint16_t(0x7115);
        lut[282] = uint16_t(0x7133);
        lut[283] = uint16_t(0x7137);
        lut[284] = uint16_t(0x713b);
        lut[285] = uint16_t(0x713f);
        lut[286] = uint16_t(0x7155);
        lut[287] = uint16_t(0x7173);
        lut[288] = uint16_t(0x7177);
        lut[289] = uint16_t(0x717b);
        lut[290] = uint16_t(0x7195);
        lut[291] = uint16_t(0x7199);
        lut[292] = uint16_t(0x719f);
        lut[293] = uint16_t(0x71b3);
        lut[294] = uint16_t(0x71d7);
        lut[295] = uint16_t(0x71f3);
        lut[296] = uint16_t(0x7313);
        lut[297] = uint16_t(0x7317);
        lut[298] = uint16_t(0x731b);
        lut[299] = uint16_t(0x7339);
        lut[300] = uint16_t(0x7353);
        lut[301] = uint16_t(0x7357);
        lut[302] = uint16_t(0x7391);
        lut[303] = uint16_t(0x73d1);
        lut[304] = uint16_t(0x73f5);
        lut[305] = uint16_t(0x73fd);
        lut[306] = uint16_t(0x7511);
        lut[307] = uint16_t(0x7535);
        lut[308] = uint16_t(0x7551);
        lut[309] = uint16_t(0x7575);
        lut[310] = uint16_t(0x7579);
        lut[311] = uint16_t(0x75b1);
        lut[312] = uint16_t(0x7713);
        lut[313] = uint16_t(0x7731);
        lut[314] = uint16_t(0x7739);
        lut[315] = uint16_t(0x7753);
        lut[316] = uint16_t(0x7757);
        lut[317] = uint16_t(0x775f);
        lut[318] = uint16_t(0x7771);
        lut[319] = uint16_t(0x7793);
        lut[320] = uint16_t(0x7797);
        lut[321] = uint16_t(0x779f);
        lut[322] = uint16_t(0x77bb);
        lut[323] = uint16_t(0x77f7);
        lut[324] = uint16_t(0x7917);
        lut[325] = uint16_t(0x7933);
        lut[326] = uint16_t(0x793d);
        lut[327] = uint16_t(0x7955);
        lut[328] = uint16_t(0x7973);
        lut[329] = uint16_t(0x79b5);
        lut[330] = uint16_t(0x79d1);
        lut[331] = uint16_t(0x79d9);
        lut[332] = uint16_t(0x7b13);
        lut[333] = uint16_t(0x7b31);
        lut[334] = uint16_t(0x7b35);
        lut[335] = uint16_t(0x7b5b);
        lut[336] = uint16_t(0x7b75);
        lut[337] = uint16_t(0x7b99);
        lut[338] = uint16_t(0x7bbd);
        lut[339] = uint16_t(0x7bf7);
        lut[340] = uint16_t(0x7d3d);
        lut[341] = uint16_t(0x7d93);
        lut[342] = uint16_t(0x7f13);
        lut[343] = uint16_t(0x7f17);
        lut[344] = uint16_t(0x7f51);
        lut[345] = uint16_t(0x7f55);
        lut[346] = uint16_t(0x7f7b);
        lut[347] = uint16_t(0x9111);
        lut[348] = uint16_t(0x9119);
        lut[349] = uint16_t(0x9135);
        lut[350] = uint16_t(0x9151);
        lut[351] = uint16_t(0x9159);
        lut[352] = uint16_t(0x915f);
        lut[353] = uint16_t(0x9175);
        lut[354] = uint16_t(0x9193);
        lut[355] = uint16_t(0x91b1);
        lut[356] = uint16_t(0x91f1);
        lut[357] = uint16_t(0x9315);
        lut[358] = uint16_t(0x931f);
        lut[359] = uint16_t(0x9333);
        lut[360] = uint16_t(0x9337);
        lut[361] = uint16_t(0x9355);
        lut[362] = uint16_t(0x9371);
        lut[363] = uint16_t(0x937b);
        lut[364] = uint16_t(0x9397);
        lut[365] = uint16_t(0x93b3);
        lut[366] = uint16_t(0x93bb);
        lut[367] = uint16_t(0x9513);
        lut[368] = uint16_t(0x9517);
        lut[369] = uint16_t(0x9531);
        lut[370] = uint16_t(0x953b);
        lut[371] = uint16_t(0x9553);
        lut[372] = uint16_t(0x9577);
        lut[373] = uint16_t(0x9591);
        lut[374] = uint16_t(0x95bf);
        lut[375] = uint16_t(0x95d5);
        lut[376] = uint16_t(0x95f1);
        lut[377] = uint16_t(0x9719);
        lut[378] = uint16_t(0x9733);
        lut[379] = uint16_t(0x9737);
        lut[380] = uint16_t(0x9751);
        lut[381] = uint16_t(0x9755);
        lut[382] = uint16_t(0x9773);
        lut[383] = uint16_t(0x977b);
        lut[384] = uint16_t(0x9911);
        lut[385] = uint16_t(0x9915);
        lut[386] = uint16_t(0x9959);
        lut[387] = uint16_t(0x997f);
        lut[388] = uint16_t(0x9991);
        lut[389] = uint16_t(0x99f3);
        lut[390] = uint16_t(0x9b1b);
        lut[391] = uint16_t(0x9b1f);
        lut[392] = uint16_t(0x9b53);
        lut[393] = uint16_t(0x9bd5);
        lut[394] = uint16_t(0x9d37);
        lut[395] = uint16_t(0x9d79);
        lut[396] = uint16_t(0x9dd1);
        lut[397] = uint16_t(0x9f31);
        lut[398] = uint16_t(0x9f3b);
        lut[399] = uint16_t(0x9f71);
        lut[400] = uint16_t(0x9f97);
        lut[401] = uint16_t(0x9fb3);
        lut[402] = uint16_t(0xb115);
        lut[403] = uint16_t(0xb131);
        lut[404] = uint16_t(0xb139);
        lut[405] = uint16_t(0xb155);
        lut[406] = uint16_t(0xb191);
        lut[407] = uint16_t(0xb199);
        lut[408] = uint16_t(0xb19f);
        lut[409] = uint16_t(0xb1b5);
        lut[410] = uint16_t(0xb1dd);
        lut[411] = uint16_t(0xb1f9);
        lut[412] = uint16_t(0xb313);
        lut[413] = uint16_t(0xb317);
        lut[414] = uint16_t(0xb31b);
        lut[415] = uint16_t(0xb335);
        lut[416] = uint16_t(0xb353);
        lut[417] = uint16_t(0xb375);
        lut[418] = uint16_t(0xb3f5);
        lut[419] = uint16_t(0xb511);
        lut[420] = uint16_t(0xb533);
        lut[421] = uint16_t(0xb557);
        lut[422] = uint16_t(0xb571);
        lut[423] = uint16_t(0xb57d);
        lut[424] = uint16_t(0xb5b7);
        lut[425] = uint16_t(0xb715);
        lut[426] = uint16_t(0xb71f);
        lut[427] = uint16_t(0xb731);
        lut[428] = uint16_t(0xb75f);
        lut[429] = uint16_t(0xb799);
        lut[430] = uint16_t(0xb7b3);
        lut[431] = uint16_t(0xb7db);
        lut[432] = uint16_t(0xb7f7);
        lut[433] = uint16_t(0xb913);
        lut[434] = uint16_t(0xb919);
        lut[435] = uint16_t(0xb951);
        lut[436] = uint16_t(0xb975);
        lut[437] = uint16_t(0xb99d);
        lut[438] = uint16_t(0xbb35);
        lut[439] = uint16_t(0xbb5d);
        lut[440] = uint16_t(0xbbb3);
        lut[441] = uint16_t(0xbbb7);
        lut[442] = uint16_t(0xbd95);
        lut[443] = uint16_t(0xbf15);
        lut[444] = uint16_t(0xbf19);
        lut[445] = uint16_t(0xbf55);
        lut[446] = uint16_t(0xd133);
        lut[447] = uint16_t(0xd137);
        lut[448] = uint16_t(0xd13b);
        lut[449] = uint16_t(0xd173);
        lut[450] = uint16_t(0xd177);
        lut[451] = uint16_t(0xd1d1);
        lut[452] = uint16_t(0xd311);
        lut[453] = uint16_t(0xd351);
        lut[454] = uint16_t(0xd35f);
        lut[455] = uint16_t(0xd3d9);
        lut[456] = uint16_t(0xd535);
        lut[457] = uint16_t(0xd579);
        lut[458] = uint16_t(0xd595);
        lut[459] = uint16_t(0xd5bb);
        lut[460] = uint16_t(0xd5d5);
        lut[461] = uint16_t(0xd5f1);
        lut[462] = uint16_t(0xd711);
        lut[463] = uint16_t(0xd739);
        lut[464] = uint16_t(0xd753);
        lut[465] = uint16_t(0xd791);
        lut[466] = uint16_t(0xd95b);
        lut[467] = uint16_t(0xd997);
        lut[468] = uint16_t(0xd9d5);
        lut[469] = uint16_t(0xdb11);
        lut[470] = uint16_t(0xdb17);
        lut[471] = uint16_t(0xdb79);
        lut[472] = uint16_t(0xdbd1);
        lut[473] = uint16_t(0xdd1b);
        lut[474] = uint16_t(0xdd91);
        lut[475] = uint16_t(0xdf33);
        lut[476] = uint16_t(0xdf37);
        lut[477] = uint16_t(0xf111);
        lut[478] = uint16_t(0xf119);
        lut[479] = uint16_t(0xf11f);
        lut[480] = uint16_t(0xf151);
        lut[481] = uint16_t(0xf155);
        lut[482] = uint16_t(0xf17d);
        lut[483] = uint16_t(0xf191);
        lut[484] = uint16_t(0xf1b9);
        lut[485] = uint16_t(0xf1d5);
        lut[486] = uint16_t(0xf315);
        lut[487] = uint16_t(0xf333);
        lut[488] = uint16_t(0xf359);
        lut[489] = uint16_t(0xf397);
        lut[490] = uint16_t(0xf39b);
        lut[491] = uint16_t(0xf513);
        lut[492] = uint16_t(0xf519);
        lut[493] = uint16_t(0xf531);
        lut[494] = uint16_t(0xf53d);
        lut[495] = uint16_t(0xf553);
        lut[496] = uint16_t(0xf571);
        lut[497] = uint16_t(0xf5b3);
        lut[498] = uint16_t(0xf715);
        lut[499] = uint16_t(0xf775);
        lut[500] = uint16_t(0xf77b);
        lut[501] = uint16_t(0xf7b7);
        lut[502] = uint16_t(0xf913);
        lut[503] = uint16_t(0xf91b);
        lut[504] = uint16_t(0xf937);
        lut[505] = uint16_t(0xf951);
        lut[506] = uint16_t(0xf9b1);
        lut[507] = uint16_t(0xfb55);
        lut[508] = uint16_t(0xfb95);
        lut[509] = uint16_t(0xfd15);
        lut[510] = uint16_t(0xfd73);
        lut[511] = uint16_t(0xff11);

    simd<uint16_t, VL / 4> index = convert<uint16_t>(qs_data);
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {
        simd<uint8_t, VL / 32> high = (qh_data >> bit) & uint8_t(1);
        index.template select<VL / 32, 8>(bit) =
            index.template select<VL / 32, 8>(bit)
            | (convert<uint16_t>(high) << 8);
    }
    simd<uint16_t, VL / 4> packed = lut.template iselect<VL / 4>(index);

    simd<float, VL> weight_f;
    #pragma unroll
    for (int field = 0; field < 4; field++) {
        simd<uint16_t, VL / 4> magnitude =
            (packed >> (4 * field)) & uint16_t(15);
        weight_f.template select<VL / 4, 4>(field) = convert<float>(magnitude);
    }
    #pragma unroll
    for (int bit = 0; bit < 8; bit++) {
        simd<uint8_t, VL / 8> negative = (signs_data >> bit) & uint8_t(1);
        auto signed_values = weight_f.template select<VL / 8, 8>(bit);
        signed_values = signed_values *
            (1.0f - 2.0f * convert<float>(negative));
        weight_f.template select<VL / 8, 8>(bit) = signed_values;
    }

    simd<float, VL / IQ3S_GROUP> scale_f = scale_h;
    #pragma unroll
    for (int group = 0; group < VL / IQ3S_GROUP; group++) {
        auto scaled = weight_f.template select<IQ3S_GROUP, 1>(
            group * IQ3S_GROUP);
        scaled = scaled * scale_f[group];
        weight_f.template select<IQ3S_GROUP, 1>(group * IQ3S_GROUP) = scaled;
    }
    return weight_f;
}

template <int VL>
struct IQ3S_gemv_kernel {
    const fp16* input;
    const uint8_t* qs;
    const uint8_t* qh;
    const uint8_t* signs;
    const fp16* scale;
    fp16* output;
    int N, K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int row = (int)item.get_group(0) * IQ3S_ROWS
                      + (int)item.get_local_id(0);
        if (row >= N) return;

        const int qs_stride = K / 4;
        const int qh_stride = K / 32;
        const int signs_stride = K / 8;
        const int scale_stride = K / IQ3S_GROUP;
        simd<float, 8> acc(0.0f);
        int ai = 0;
        for (int k = 0; k < K; k += VL) {
            simd<fp16, VL> act = block_load<fp16, VL>(input + k);
            simd<uint8_t, VL / 4> qs_data = block_load<uint8_t, VL / 4>(
                qs + (size_t)row * qs_stride + k / 4);
            simd<uint8_t, VL / 32> qh_data = block_load<uint8_t, VL / 32>(
                qh + (size_t)row * qh_stride + k / 32);
            simd<uint8_t, VL / 8> signs_data = block_load<uint8_t, VL / 8>(
                signs + (size_t)row * signs_stride + k / 8);
            simd<fp16, VL / IQ3S_GROUP> scale_h =
                block_load<fp16, VL / IQ3S_GROUP>(
                    scale + (size_t)row * scale_stride + k / IQ3S_GROUP);
            simd<float, VL> weight_f = iq3s_dequant_tile<VL>(
                qs_data, qh_data, signs_data, scale_h);
            simd<float, VL> product = weight_f * simd<float, VL>(act);
            acc[ai] += iq3s_esimd_detail::sum<float, float, VL>(product);
            ai = (ai + 1) & 7;
        }
        output[row] = fp16(iq3s_esimd_detail::sum<float, float, 8>(acc));
    }
};

inline void iq3s_gemv_host(
    const fp16* input, const uint8_t* qs, const uint8_t* qh,
    const uint8_t* signs, const fp16* scale, fp16* output,
    uint32_t N, uint32_t K, sycl::queue& q) {
    const int nwg = ((int)N + IQ3S_ROWS - 1) / IQ3S_ROWS;
    const bool wide = (K % IQ3S_VL) == 0;
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> range((size_t)nwg * IQ3S_ROWS, IQ3S_ROWS);
        if (wide) {
            h.parallel_for(range, IQ3S_gemv_kernel<IQ3S_VL>{
                input, qs, qh, signs, scale, output, (int)N, (int)K});
        } else {
            h.parallel_for(range, IQ3S_gemv_kernel<IQ3S_VL / 2>{
                input, qs, qh, signs, scale, output, (int)N, (int)K});
        }
    });
}

template <int M, int VL>
struct IQ3S_gemv_M_kernel {
    const fp16* input;
    const uint8_t* qs;
    const uint8_t* qh;
    const uint8_t* signs;
    const fp16* scale;
    fp16* output;
    int N, K, ldo;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        const int row = (int)item.get_group(0) * IQ3S_ROWS
                      + (int)item.get_local_id(0);
        if (row >= N) return;

        constexpr int AW = 64;
        const int qs_stride = K / 4;
        const int qh_stride = K / 32;
        const int signs_stride = K / 8;
        const int scale_stride = K / IQ3S_GROUP;
        simd<float, AW> acc[M];
        #pragma unroll
        for (int m = 0; m < M; m++) acc[m] = 0.0f;

        for (int k = 0; k < K; k += VL) {
            simd<uint8_t, VL / 4> qs_data = block_load<uint8_t, VL / 4>(
                qs + (size_t)row * qs_stride + k / 4);
            simd<uint8_t, VL / 32> qh_data = block_load<uint8_t, VL / 32>(
                qh + (size_t)row * qh_stride + k / 32);
            simd<uint8_t, VL / 8> signs_data = block_load<uint8_t, VL / 8>(
                signs + (size_t)row * signs_stride + k / 8);
            simd<fp16, VL / IQ3S_GROUP> scale_h =
                block_load<fp16, VL / IQ3S_GROUP>(
                    scale + (size_t)row * scale_stride + k / IQ3S_GROUP);
            simd<float, VL> weight_f = iq3s_dequant_tile<VL>(
                qs_data, qh_data, signs_data, scale_h);
            #pragma unroll
            for (int m = 0; m < M; m++) {
                simd<fp16, VL> act = block_load<fp16, VL>(
                    input + (size_t)m * K + k);
                #pragma unroll
                for (int c = 0; c < VL / AW; c++) {
                    acc[m] += weight_f.template select<AW, 1>(c * AW)
                            * simd<float, AW>(
                                act.template select<AW, 1>(c * AW));
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < M; m++) {
            output[(size_t)m * ldo + row] = fp16(
                iq3s_esimd_detail::sum<float, float, AW>(acc[m]));
        }
    }
};

template <int M>
inline void iq3s_gemv_M_launch(
    const fp16* input, const uint8_t* qs, const uint8_t* qh,
    const uint8_t* signs, const fp16* scale, fp16* output,
    uint32_t N, uint32_t K, uint32_t ldo, sycl::queue& q) {
    const int nwg = ((int)N + IQ3S_ROWS - 1) / IQ3S_ROWS;
    const bool wide = (K % IQ3S_VL) == 0;
    q.submit([&](sycl::handler& h) {
        sycl::nd_range<1> range((size_t)nwg * IQ3S_ROWS, IQ3S_ROWS);
        if (wide) {
            h.parallel_for(range, IQ3S_gemv_M_kernel<M, IQ3S_VL>{
                input, qs, qh, signs, scale, output,
                (int)N, (int)K, (int)ldo});
        } else {
            h.parallel_for(range, IQ3S_gemv_M_kernel<M, IQ3S_VL / 2>{
                input, qs, qh, signs, scale, output,
                (int)N, (int)K, (int)ldo});
        }
    });
}

inline void iq3s_gemv_M_host(
    const fp16* input, const uint8_t* qs, const uint8_t* qh,
    const uint8_t* signs, const fp16* scale, fp16* output,
    uint32_t M, uint32_t N, uint32_t K, uint32_t ldo, sycl::queue& q) {
    uint32_t m0 = 0;
    while (m0 < M) {
        const uint32_t remaining = M - m0;
        const fp16* in = input + (size_t)m0 * K;
        fp16* out = output + (size_t)m0 * ldo;
        if (remaining >= 16) {
            iq3s_gemv_M_launch<16>(in, qs, qh, signs, scale, out,
                                   N, K, ldo, q);
            m0 += 16;
        } else if (remaining >= 8) {
            iq3s_gemv_M_launch<8>(in, qs, qh, signs, scale, out,
                                  N, K, ldo, q);
            m0 += 8;
        } else if (remaining >= 4) {
            iq3s_gemv_M_launch<4>(in, qs, qh, signs, scale, out,
                                  N, K, ldo, q);
            m0 += 4;
        } else if (remaining >= 2) {
            iq3s_gemv_M_launch<2>(in, qs, qh, signs, scale, out,
                                  N, K, ldo, q);
            m0 += 2;
        } else {
            iq3s_gemv_host(in, qs, qh, signs, scale, out, N, K, q);
            m0 += 1;
        }
    }
}
