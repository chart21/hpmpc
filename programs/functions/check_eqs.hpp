#pragma once
#if PARTY == 0
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_0_init_template.hpp"
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_0_template.hpp"
#define CV_INIT OEC_MAL0_NO_CV_init<Datatype>
#define CV_LIVE OEC_MAL0_NO_CV_Share<Datatype>
#elif PARTY == 1
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_1_init_template.hpp"
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_1_template.hpp"
#define CV_INIT OEC_MAL1_NO_CV_init<Datatype>
#define CV_LIVE OEC_MAL1_NO_CV_Share<Datatype>
#elif PARTY == 2
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_2_init_template.hpp"
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_2_template.hpp"
#define CV_INIT OEC_MAL2_NO_CV_init<Datatype>
#define CV_LIVE OEC_MAL2_NO_CV_Share<Datatype>
#elif PARTY == 3
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_3_init_template.hpp"
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_3_template.hpp"
#define CV_INIT OEC_MAL3_NO_CV_init<Datatype>
#if PRE == 1
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_3-post_template.hpp"
#define CV_PRE OEC_MAL3_NO_CV_Share<Datatype>
#define CV_LIVE OEC_MAL3_POST_NO_CV_Share<Datatype>  
#else
#define CV_LIVE OEC_MAL3_NO_CV_PRE_Share<Datatype>
#endif
#endif

#include "../../datatypes/XOR_Share.hpp"
#include "../../protocols/live_protocol_base.hpp"

const int sha_length = 256;

template <typename Share, typename Datatype, int k>
void check_eqs(char hash[k][sha_length/8], int ph1[], int ph2[])
{
    const int len = sha_length / 8;
    using S = XOR_Share<Datatype, Share>;

    auto chash[k][len];
    for(int i = 0; i < k; i++)
    {
        auto shash1[len];
        auto shash2[len];
        auto chash[len];
        if(ph1[i] == PARTY)
        {
            Datatype dhash[len];
            for (int j = 0; j < len; j++)
            {
            dhash[j] = (hash1[i][j] >> 7) & 1 ? ZERO : ONES;
            shash1[i][j].template prepare_receive_from<PARTY>(dhash[j]);
            }
        }
        else
        {
        switch (ph1[i])
        {
        case 0:
            for (int j = 0; j < len; j++)
                shash1[i][j].template prepare_receive_from<0>();
            break;
        case 1:
            for (int j = 0; j < len; j++)
                shash1[i][j].template prepare_receive_from<1>();
            break;
        case 2:
            for (int j = 0; j < len; j++)
                shash1[i][j].template prepare_receive_from<2>();
            break;
        case 3:
            for (int j = 0; j < len; j++)
                shash1[i][j].template prepare_receive_from<3>();
            break;
        }
        }
        
        if(ph2[i] == PARTY)
        {
            Datatype dhash[len];
            for (int j = 0; j < len; j++)
            {
            dhash[i] = (hash2[i][j] >> 7) & 1 ? ZERO : ONES;
            shash2[i][j].template prepare_receive_from<PARTY>(dhash[i]);
        }
        }
        else
        {
        switch(ph2[i])
        {
        case 0:
            for (int j = 0; j < len; j++)
                shash2[i][j].template prepare_receive_from<0>();
            break;
        case 1:
            for (int j = 0; j < len; j++)
                shash2[i][j].template prepare_receive_from<1>();
            break;
        case 2:
            for (int j = 0; j < len; j++)
                shash2[i][j].template prepare_receive_from<2>();
            break;
        case 3:
            for (int j = 0; j < len; j++)
                shash2[i][j].template prepare_receive_from<3>();
            break;
        }
        }
    }
        Share::communicate();
        for (int i = 0; i < k; i++)
        {
            switch(ph1[i])
            {
            case 0:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from<0>();
                break;
            case 1:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from<1>();
                break;
            case 2:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from<2>();
                break;
            case 3:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from<3>();
                break;
            }
            switch(ph2[i])
            {
            case 0:               for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from<0>();
                break;
            case 1:               for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from<1>();
                break;
            case 2:               for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from<2>();
                break;
            case 3:               for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from<3>();
                break;  
            }
        }
        Share::communicate();
        for (int i = 0; i < k; i++)
        {
            switch(ph1[i])
            {
            case 0:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from2<0>();
                break;
            case 1:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from2<1>();
                break;
            case 2:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from2<2>();
                break;
            case 3: 
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from2<3>();
                break;
            }   
            switch (ph2[i])
            {            case 0:
                for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from2<0>();
                break;
            case 1:
                for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from2<1>();
                break;
            case 2:
                for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from2<2>();
                break;
            case 3:               
                for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from2<3>();
                break;  
            }   
        }
        for (int i = 0; i < k; i++)
            for (int j = 0; j < len; j++)
                chash[i][j] = ! (shash1[i][j] ^ shash2[i][j]);
    
    for(int i = 0; i < (int)std::log2(len); i++)
    {
        for(int j = 0; j < k; j++)
        {
            for(int l = 0; l < len; l += 1 << (i+1))
            {
               chash[j][l] = chash[j][l].prepare_and(chash[j][l + (1 << i)]);
            }
        }
        Share::communicate();
        for(int j = 0; j < k; j++)
        {
            for(int l = 0; l < len; l += 1 << (i+1))
            {
               chash[j][l].complete_and();
            }
        }
        Share::communicate();
        for(int j = 0; j < k; j++)
        {
            for(int l = 0; l < len; l += 1 << (i+1))
            {
               chash[j][l].complete_and2();
            }
        }
    }
        
    for(int i = 0; i < (int)std::log2(k); i++)
    {
        for(int j = 0; j < k; j += 1 << (i+1))
        {
        chash[j][0] = chash[j][0].prepare_and(chash[j + (1 << i)][0]);
        }
        Share::communicate();
        for(int j = 0; j < k; j += 1 << (i+1))
        {        
            chash[j][0].complete_and();
        }
        Share::communicate();
        for(int j = 0; j < k; j += 1 << (i+1))
        {        
            chash[j][0].complete_and2();       
        }
    }

    chash[0][0].prepare_reveal_to_all();
    Share::communicate();
    Datatype result = chash[0][0].complete_reveal_to_all();
    if(dat_equal(result, ZERO))
    {
        print("Checkeqs passed!");
    }
    else
    {
        print("Checkeqs failed!");
    }
}

constexpr int ph1_quad[] = {0, 0};
constexpr int ph2_quad[] = {1, 2};
const int num_comparisons = 2;

template <typename Share, typename Datatype>
void compare_view_check_eqs_quad(int ph1[], int ph2[], int sha_length, int num_comparisons)
{
    //perform final hashes
    for (int player_id = 0; player_id < num_players * player_multiplier; player_id++)
    {
        if (elements_to_compare[player_id] > 0)
        {
            perform_compare_view(player_id);
        }
    }
   
    //optimization: merge P012 verification into P0 <-> P1 and P0 <-> P2 verification
    if(PARTY == 0 && elements_to_compare[P_012] > 0)
    {
        verify_buffer[P_1][0] = verify_buffer[P_012][0];
        verify_buffer[P_2][0] = verify_buffer[P_012][0];
        verify_buffer_index[P_2][0] = 1;
        verify_buffer_index[P_1][0] = 1;
    }
    else if(PARTY == 1 && elements_to_compare[P_012] > 0)
    {
        verify_buffer[P_0][0] = verify_buffer[P_012][0];
        verify_buffer_index[P_0][0] = 1;
    }
    else if(PARTY == 2 && elements_to_compare[P_012] > 0)
    {
        verify_buffer[P_0][0] = verify_buffer[P_012][0];
        verify_buffer_index[P_0][0] = 1;
    }
    perform_compare_view(P_012);

    // perform compare view check eqs

    char hashes[num_comparisons][sha_length / 8];
#if PARTY == 0
    std::memcpy(hashes[0], hash_val[P_1], sha_length / 8);
    std::memcpy(hashes[1], hash_val[P_2], sha_length / 8);
#elif PARTY == 1
    std::memcpy(hashes[0], hash_val[P_0], sha_length / 8);
#elif PARTY == 2
    std::memcpy(hashes[1], hash_val[P_0], sha_length / 8);
#endif
    check_eqs<XOR_Share<DATATYPE, Share>, Datatype, num_comparisons>(hashes, sha_length / 8, ph1_quad, ph2_quad);
}

void check_eqs_quad_init()
{
    const char hash[num_comparisons][sha_length/8];
    check_eqs<XOR_Share<DATATYPE, CV_INIT>, DATATYPE>(hash, ph1_quad, ph2_quad);
}

void check_eqs_quad_pre()
{
    #ifdef CV_PRE
    const char hash[num_comparisons][sha_length/8];
    check_eqs<XOR_Share<DATATYPE, CV_PRE>, DATATYPE>(hash, ph1_quad, ph2_quad);
    #endif
}

void check_eqs_quad_live()
{
  compare_view_check_eqs_quad<XOR_Share<DATATYPE, CV_LIVE>, DATATYPE>(ph1_quad, ph2_quad, sha_length, num_comparisons);   
}

