#include <fstream>
#include <iostream>
#include <vector>


template <typename T>
void read_bin_file(std::vector<T>& vec, std::string fpath) {
    std::ifstream binFileIn(fpath, std::ios::in | std::ios::binary);
    if (!binFileIn.is_open()) {
        std::cerr << "Error: read file " << fpath << std::endl;
        std::terminate();
    }

    // Read the binary data
    binFileIn.read((char*)vec.data(), vec.size()*sizeof(T));
}


template <typename T>
void write_bin_file(std::vector<T>& vec, std::string fpath) {
    std::ofstream binFileOut(fpath, std::ios::out | std::ios::binary);
    if (!binFileOut.is_open()) {
        std::cerr << "Error: write file " << fpath << std::endl;
        std::terminate();
    }
    
    binFileOut.write((char*)vec.data(), vec.size()*sizeof(T));
    binFileOut.close();
}
