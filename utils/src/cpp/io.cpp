#include <fstream>
#include <iostream>
#include <string>
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


std::string replace_string(std::string org_string, std::string old_string, std::string new_string) {
    std::string mod_string = org_string;
    size_t pos = mod_string.find(old_string);

    if (pos != std::string::npos)
        mod_string.replace(pos, old_string.length(), new_string);
    else
        std::cerr << "Error: \"" << old_string << "\" not found in \"" << org_string << "\"" << std::endl;
    
    return mod_string;
}
