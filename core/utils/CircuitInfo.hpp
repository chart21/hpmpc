#pragma once
#if NO_INI == 1
#include "../../config.h"
#include "../networking/buffers.h"
#include "CircuitDetails.h"
#include <fstream>
#include <iostream>
void init_from_file()
{
    RR SR ETR ETS

        for (int t = 0; t < (num_players - 1); t++)
    {
        receiving_args[t].rec_rounds = rec_rounds[t];
        sending_args[t].send_rounds = send_rounds[t];
        for (int i = 0; i < rec_rounds[t]; i++)
        {
            receiving_args[t].elements_to_rec[i] = elements_to_rec[t][i];
        }
        for (int i = 0; i < send_rounds[t]; i++)
        {
            sending_args[t].elements_to_send[i] = elements_to_send[t][i];
        }
    }
}

void export_Details_to_file()
{
    std::string line1 = "#define RR int send_rounds[";
    line1 += std::to_string(num_players - 1);
    line1 += "] = {";
    for (int t = 0; t < (num_players - 1); t++)
    {
        line1 += std::to_string(receiving_args[t].rec_rounds);
        line1 += ",";
    }
    line1 += "};";

    std::string line2 = "#define SR int rec_rounds[";
    line2 += std::to_string(num_players - 1);
    line2 += "] = {";
    for (int t = 0; t < (num_players - 1); t++)
    {
        line2 += std::to_string(sending_args[t].send_rounds);
        line2 += ",";
    }
    line2 += "};";

    std::string line3 = "#define ETR int elements_to_rec[";
    line3 += std::to_string(num_players - 1);
    line3 += "][";
    line3 += std::to_string(receiving_args[0].rec_rounds);  // currently rec_rounds of all threads are the same
    line3 += "] = {";
    std::string line4 = "#define ETS int elements_to_send[";
    line4 += std::to_string(num_players - 1);
    line4 += "][";
    line4 += std::to_string(sending_args[0].send_rounds);  // currently send_rounds of all threads are the same
    line4 += "] = {";

    for (int t = 0; t < (num_players - 1); t++)
    {
        line3 += "{";
        line4 += "{";
        for (int i = 0; i < receiving_args[t].rec_rounds; i++)
        {
            line3 += std::to_string(receiving_args[t].elements_to_rec[i]);
            line3 += ",";
        }
        for (int i = 0; i < sending_args[t].send_rounds; i++)
        {
            line4 += std::to_string(sending_args[t].elements_to_send[i]);
            line4 += ",";
        }
        line3 += "},";
        line4 += "},";
    }
    line3 += "};";
    line4 += "};";

    std::ofstream myfile;
    myfile.open("protocols/CircuitDetails.h");
    myfile << line1 << "\n";
    myfile << line2 << "\n";
    myfile << line3 << "\n";
    myfile << line4;
    myfile.close();
    std::cout << "Exported Circuit Details"
              << "\n";
}

void finalize(std::string* ips)
{
    for (int t = 0; t < (num_players - 1); t++)
    {
        int offset = 0;
        if (t >= player_id)
            offset = 1;  // player should not receive from itself
        receiving_args[t].player_count = num_players;
        receiving_args[t].received_elements =
            new DATATYPE*[receiving_args[t].rec_rounds];  // every thread gets its own pointer array for receiving
                                                          // elements

        receiving_args[t].player_id = player_id;
        receiving_args[t].connected_to = t + offset;
        receiving_args[t].ip = ips[t];
        receiving_args[t].hostname = (char*)"hostname";
        receiving_args[t].port =
            (int)base_port + player_id * (num_players - 1) +
            t;  // e.g. P_0 receives on base port from P_1, P_2 on base port + num_players from P_0 6000,6002
    }
    for (int t = 0; t < (num_players - 1); t++)
    {
        int offset = 0;
        if (t >= player_id)
            offset = 1;  // player should not send to itself
        sending_args[t].sent_elements = new DATATYPE*[sending_args[t].send_rounds];
        /* sending_args[t].elements_to_send[0] = 0; //input sharing with SRNGs */
        sending_args[t].player_id = player_id;
        sending_args[t].player_count = num_players;
        sending_args[t].connected_to = t + offset;
        sending_args[t].port = (int)base_port + (t + offset) * (num_players - 1) + player_id - 1 +
                               offset;  // e.g. P_0 sends on base port + num_players  for P_1, P_2 on base port +
                                        // num_players for P_0 (6001,6000)
        sending_args[t].sent_elements[0] =
            NEW(DATATYPE[sending_args[t].elements_to_send[0]]);  // Allocate memory for first round
    }
    rounds = 0;
    sending_rounds = 0;
    rb = 0;
    sb = 0;
}
#endif
