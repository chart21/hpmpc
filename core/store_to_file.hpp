#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <cstdint> // For uint64_t

template <typename T>
void save_triple_file(T* arithmetic_triple_a, uint64_t size_a, 
                      T* arithmetic_triple_b, uint64_t size_b, 
                      T* boolean_triple_a, uint64_t size_bool_a,
                      T* boolean_triple_b, uint64_t size_bool_b,
                      const std::string& pself, 
                      const std::string& ending) {
    // Construct the filenames
    std::string filename = "triples_" + pself + "." + ending;
    std::string lock_filename = "triples_" + pself + "_lock." + ending;

    // Create a lock file
    std::ofstream lockfile(lock_filename);
    if (!lockfile) {
        std::cerr << "Error creating lock file: " << lock_filename << std::endl;
        return;
    }
    lockfile.close();

    // Open the file for writing
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        std::filesystem::remove(lock_filename); // Remove lock file if writing fails
        return;
    }

    // Write arithmetic triples
    outfile.write(reinterpret_cast<const char*>(&size_a), sizeof(size_a));
    outfile.write(reinterpret_cast<const char*>(arithmetic_triple_a), size_a * sizeof(T));
    
    outfile.write(reinterpret_cast<const char*>(&size_b), sizeof(size_b));
    outfile.write(reinterpret_cast<const char*>(arithmetic_triple_b), size_b * sizeof(T));
    
    // Write boolean triples
    outfile.write(reinterpret_cast<const char*>(&size_bool_a), sizeof(size_bool_a));
    outfile.write(reinterpret_cast<const char*>(boolean_triple_a), size_bool_a * sizeof(T));
    
    outfile.write(reinterpret_cast<const char*>(&size_bool_b), sizeof(size_bool_b));
    outfile.write(reinterpret_cast<const char*>(boolean_triple_b), size_bool_b * sizeof(T));

    outfile.close();

    // Remove the lock file after writing
    std::filesystem::remove(lock_filename);
    std::cout << "Data saved to " << filename << std::endl;
}

template <typename T>
void load_triple_file(T*& arithmetic_triple_a, uint64_t& size_a, 
                      T*& arithmetic_triple_b, uint64_t& size_b, 
                      T*& boolean_triple_a, uint64_t& size_bool_a, 
                      T*& boolean_triple_b, uint64_t& size_bool_b, 
                      const std::string& pother, 
                      const std::string& ending) {
    // Construct the filenames
    std::string lock_filename = "triples_" + pother + "_lock." + ending;
    std::string data_filename = "triples_" + pother + "." + ending;

    // Wait until the lock file is deleted and the data file exists
    while (std::filesystem::exists(lock_filename) || !std::filesystem::exists(data_filename)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Sleep for a short duration
    }

    // Open the file for reading
    std::ifstream infile(data_filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file for reading: " << data_filename << std::endl;
        return;
    }

    // Read arithmetic triples
    infile.read(reinterpret_cast<char*>(&size_a), sizeof(size_a));
    arithmetic_triple_a = new T[size_a];
    infile.read(reinterpret_cast<char*>(arithmetic_triple_a), size_a * sizeof(T));
    
    infile.read(reinterpret_cast<char*>(&size_b), sizeof(size_b));
    arithmetic_triple_b = new T[size_b];
    infile.read(reinterpret_cast<char*>(arithmetic_triple_b), size_b * sizeof(T));
    
    // Read boolean triples
    infile.read(reinterpret_cast<char*>(&size_bool_a), sizeof(size_bool_a));
    boolean_triple_a = new T[size_bool_a];
    infile.read(reinterpret_cast<char*>(boolean_triple_a), size_bool_a * sizeof(T));
    
    infile.read(reinterpret_cast<char*>(&size_bool_b), sizeof(size_bool_b));
    boolean_triple_b = new T[size_bool_b];
    infile.read(reinterpret_cast<char*>(boolean_triple_b), size_bool_b * sizeof(T));

    infile.close();
    std::cout << "Data loaded from " << data_filename << std::endl;
}

void delete_triple_file(const std::string& pself, const std::string& ending) {
    std::string filename = "triples_" + pself + "." + ending;
    if (std::filesystem::exists(filename)) {
        std::filesystem::remove(filename);
        std::cout << "Deleted file: " << filename << std::endl;
    } else {
        std::cout << "File not found for deletion: " << filename << std::endl;
    }
}
