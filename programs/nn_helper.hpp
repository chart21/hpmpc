#pragma once
#include "../core/include/pch.h"
#include "headers/config.h"
void load_config(Config& cfg, std::string filename)
{
// Open the configuration file
    FILE* file = fopen(filename.c_str(), "r");
    if (file == NULL) {
        std::cerr << "Error opening file" << std::endl;
        exit(1);
    }

    // Buffer to hold each line from the file
    char line[256];

    // Read the file line by line
    while (fgets(line, sizeof(line), file)) {
        // Get the key part of the line
        char* key = strtok(line, "=");
        // Get the value part of the line
        char* value = strtok(NULL, "\n");

        if (key && value) {
            // Compare the key and assign the value to the appropriate field in cfg
            /* std::cout << " Key: " << key << " Value: " << value << std::endl; */
            if (strcmp(key, "mode") == 0) {
                cfg.mode = value;
            }
            else if (strcmp(key, "save_dir") == 0) {
                cfg.save_dir = value;
            }
            else if (strcmp(key, "data_dir") == 0) {
                cfg.data_dir = value;
            }
            else if (strcmp(key, "pretrained") == 0) {
                cfg.pretrained = value;
            }
            else if (strcmp(key, "image_file") == 0) {
                cfg.image_file = value;
            }
            else if (strcmp(key, "label_file") == 0) {
                cfg.label_file = value;
            }
        }
    }

    // Close the file after reading
    fclose(file);
}
