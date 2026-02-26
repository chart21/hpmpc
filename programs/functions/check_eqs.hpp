#pragma once
#if PARTY == 0
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_0_init_template.hpp"
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_0_template.hpp"
#define CV_INIT OEC_MAL0_NO_CV_init<DATATYPE>
#define CV_LIVE OEC_MAL0_NO_CV_Share<DATATYPE>
#elif PARTY == 1
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_1_init_template.hpp"
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_1_template.hpp"
#define CV_INIT OEC_MAL1_NO_CV_init<DATATYPE>
#define CV_LIVE OEC_MAL1_NO_CV_Share<DATATYPE>
#elif PARTY == 2
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_2_init_template.hpp"
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_2_template.hpp"
#define CV_INIT OEC_MAL2_NO_CV_init<DATATYPE>
#define CV_LIVE OEC_MAL2_NO_CV_Share<DATATYPE>
#elif PARTY == 3
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_3_init_template.hpp"
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_3_template.hpp"
#define CV_INIT OEC_MAL3_NO_CV_init<DATATYPE>
#if PRE == 1
#include "../../protocols/4-PC/quad_no_cv/oec-mal-P_3-post_template.hpp"
#define CV_PRE OEC_MAL3_NO_CV_Share<DATATYPE>
#define CV_LIVE OEC_MAL3_POST_NO_CV_Share<DATATYPE>  
#else
#define CV_LIVE OEC_MAL3_NO_CV_Share<DATATYPE>
#endif
#endif

#include "../../datatypes/XOR_Share.hpp"
#include "../../protocols/live_protocol_base.hpp"

constexpr int sha_length = 256;
constexpr int sha_bytes = sha_length / 8;

template <typename Datatype, typename Share, int k>
void check_eqs(const char hash[k][sha_bytes], const int ph1[], const int ph2[])
{
    constexpr int len = sha_bytes;
    using S = XOR_Share<Datatype, Share>;

    S chash[k][len];
    S shash1[k][len];
    S shash2[k][len];
    for(int i = 0; i < k; i++)
    {
        if(ph1[i] == PARTY)
        {
            Datatype dhash[len];
            for (int j = 0; j < len; j++)
            {
            dhash[j] = (hash[i][j] >> 7) & 1 ? ZERO : ONES;
            shash1[i][j].template prepare_receive_from<PSELF>(dhash[j]);
            }
        }
        else
        {
        switch (ph1[i])
        {
        case 0:
            for (int j = 0; j < len; j++)
                shash1[i][j].template prepare_receive_from<P_0>(SET_ALL_ZERO());
            break;
        case 1:
            for (int j = 0; j < len; j++)
                shash1[i][j].template prepare_receive_from<P_1>(SET_ALL_ZERO());
            break;
        case 2:
            for (int j = 0; j < len; j++)
                shash1[i][j].template prepare_receive_from<P_2>(SET_ALL_ZERO());
            break;
        case 3:
            for (int j = 0; j < len; j++)
                shash1[i][j].template prepare_receive_from<P_3>(SET_ALL_ZERO());
            break;
        }
        }
        
        if(ph2[i] == PARTY)
        {
            Datatype dhash[len];
            for (int j = 0; j < len; j++)
            {
            dhash[i] = (hash[i][j] >> 7) & 1 ? ZERO : ONES;
            shash2[i][j].template prepare_receive_from<PSELF>(dhash[i]);
        }
        }
        else
        {
        switch(ph2[i])
        {
        case 0:
            for (int j = 0; j < len; j++)
                shash2[i][j].template prepare_receive_from<P_0>(SET_ALL_ZERO());
            break;
        case 1:
            for (int j = 0; j < len; j++)
                shash2[i][j].template prepare_receive_from<P_1>(SET_ALL_ZERO());
            break;
        case 2:
            for (int j = 0; j < len; j++)
                shash2[i][j].template prepare_receive_from<P_2>(SET_ALL_ZERO());
            break;
        case 3:
            for (int j = 0; j < len; j++)
                shash2[i][j].template prepare_receive_from<P_3>(SET_ALL_ZERO());
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
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from<P_0>();
                break;
            case 1:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from<P_1>();
                break;
            case 2:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from<P_2>();
                break;
            case 3:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from<P_3>();
                break;
            }
            switch(ph2[i])
            {
            case 0:               for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from<P_0>();
                break;
            case 1:               for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from<P_1>();
                break;
            case 2:               for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from<P_2>();
                break;
            case 3:               for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from<P_3>();
                break;  
            }
        }
        Share::communicate();
        for (int i = 0; i < k; i++)
        {
            switch(ph1[i])
            {
            case 0:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from2<P_0>();
                break;
            case 1:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from2<P_1>();
                break;
            case 2:
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from2<P_2>();
                break;
            case 3: 
                for (int j = 0; j < len; j++)                    shash1[i][j].template complete_receive_from2<P_3>();
                break;
            }   
            switch (ph2[i])
            {            case 0:
                for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from2<P_0>();
                break;
            case 1:
                for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from2<P_1>();
                break;
            case 2:
                for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from2<P_2>();
                break;
            case 3:               
                for (int j = 0; j < len; j++)                    shash2[i][j].template complete_receive_from2<P_3>();
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
    std::cout << "Checkeqs result: " << result << std::endl;
    if(current_phase == PHASE_LIVE)
    {
    if(dat_equal(result, ONES))
    {
        print("Checkeqs passed! \n");
    }
    else
    {
        print("Checkeqs failed! \n");
    }
    }
}

constexpr int ph1_quad[] = {0, 0};
constexpr int ph2_quad[] = {1, 2};
constexpr int num_comparisons_quad = 2;

template <typename Datatype, typename Share>
void compare_view_check_eqs_quad(const int ph1[], const int ph2[])
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
        verify_buffer_index[P_2] = 1;
        verify_buffer_index[P_1] = 1;
    }
    else if(PARTY == 1 && elements_to_compare[P_012] > 0)
    {
        verify_buffer[P_0][0] = verify_buffer[P_012][0];
        verify_buffer_index[P_0] = 1;
    }
    else if(PARTY == 2 && elements_to_compare[P_012] > 0)
    {
        verify_buffer[P_0][0] = verify_buffer[P_012][0];
        verify_buffer_index[P_0] = 1;
    }
    perform_compare_view(P_012);

    // perform compare view check eqs

    constexpr int len = sha_bytes;
    char hashes[num_comparisons_quad][len];
#if PARTY == 0
    std::memcpy(hashes[0], hash_val[P_1], len);
    std::memcpy(hashes[1], hash_val[P_2], len);
#elif PARTY == 1
    std::memcpy(hashes[0], hash_val[P_0], len);
#elif PARTY == 2
    std::memcpy(hashes[1], hash_val[P_0], len);
#endif

#if PARTY == 0
    std::fill(hashes[0], hashes[0] + len, 0);
    std::fill(hashes[1], hashes[1] + len, 1);
#elif PARTY == 1
    std::fill(hashes[0], hashes[0] + len, 0);
#elif PARTY == 2
    std::fill(hashes[1], hashes[1] + len, 1);
#endif
    check_eqs<Datatype, Share, num_comparisons_quad>(hashes, ph1_quad, ph2_quad);
}

void check_eqs_quad_init()
{
    char hash[num_comparisons_quad][sha_bytes];
    check_eqs<DATATYPE, CV_INIT, num_comparisons_quad>(hash, ph1_quad, ph2_quad);
}

void check_eqs_quad_pre()
{
    #ifdef CV_PRE
    const char hash[num_comparisons_quad][sha_bytes];
    check_eqs<DATATYPE, CV_PRE, num_comparisons_quad>(hash, ph1_quad, ph2_quad);
    #endif
}

void check_eqs_quad_live()
{
    compare_view_check_eqs_quad<DATATYPE, CV_LIVE>(ph1_quad, ph2_quad);
}

