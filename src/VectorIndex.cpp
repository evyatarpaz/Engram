#include "../include/VectorIndex.h"

#include <cmath>        
#include <stdexcept>    
#include <fstream>      
#include <algorithm>    
#include <iostream>  


// Constructor using "Initializer List" style (Preferred in C++)
VectorIndex::VectorIndex(size_t dimension) : _dimension(dimension), _count(0) {
    _data.reserve(1000 * dimension); // Pre-allocate space for 1000 vectors
}


// Private Method: Calculate Squared Euclidean Distance
float VectorIndex::calculate_distance(const float* vec_a, const float* vec_b) const{
    float dist = 0.0f;
    for (size_t i = 0; i < _dimension; ++i){
        float diff = vec_a[i] - vec_b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// Public Method: Search Nearest Neighbors
std::vector<std::pair<size_t, float>> VectorIndex::search(const std::vector<float>& query, int k){
    // Validate query dimension
    if(query.size() != _dimension){
        throw std::invalid_argument("Query vector dimension does not match index dimension." + std::to_string(query.size()) + " != " + std::to_string(_dimension));
    }
    // Compute distances to all vectors
    std::vector<std::pair<size_t,float>> results;
    results.reserve(_count);

    // Calculate distances
    for(size_t i = 0; i < _count; i++){
        float distance = calculate_distance(&_data[i*_dimension],query.data());
        results.emplace_back(i,distance);
    }
    // Sort results by distance
    std::sort(results.begin(),results.end(),[](const auto& a, const auto& b){
        return a.second < b.second;
    });

    // Return top-k results
    if(results.size() > k){
        results.resize(k);
    }
    // Return the results
    return results;
}

// Public Method: Add Vector
void VectorIndex::add_vector(const std::vector<float>& vec){
    if (vec.size() != get_dimension()){
        throw std::invalid_argument("Vector dimension does not match index dimension.");
    }
    _count += 1;
    _data.insert(_data.end(), vec.begin(), vec.end()); 
}


// Public Method: Save Index to Disk
void VectorIndex::save_index(const std::string& filepath) const{

    // Open file in binary write mode
    std::ofstream output_file(filepath, std::ios::binary);
    // Check if file opened successfully
    if(!output_file.is_open()) {
        throw std::runtime_error("Error: Could not open file for writing: " + filepath); 
    }

    // Write dimension and count
    output_file.write(reinterpret_cast<const char*>(&_dimension), sizeof(_dimension));
    output_file.write(reinterpret_cast<const char*>(&_count), sizeof(_count));

    // Write vector data
    if(_data.size() > 0) {
        output_file.write(reinterpret_cast<const char*>(_data.data()), _data.size() * sizeof(float));
    }
    output_file.close();


}
// Public Method: Load Index from Disk
void VectorIndex::load_index(const std::string& filepath){
    std::ifstream input_file(filepath, std::ios::binary);

    if(!input_file.is_open()) {
        throw std::runtime_error("Error: Could not open file for reading: " + filepath);
    }
    size_t file_dimension = 0;
    size_t file_count = 0;
    // Read dimension and count
    input_file.read(reinterpret_cast<char*>(&file_dimension), sizeof(file_dimension));
    input_file.read(reinterpret_cast<char*>(&file_count), sizeof(file_count));

    if(file_dimension != _dimension) {
        throw std::runtime_error("Error: Dimension mismatch when loading index from file.");
    }

    // Read vector data
    _data.resize(file_count * file_dimension);
    input_file.read(reinterpret_cast<char*>(_data.data()), _data.size() * sizeof(float));

    if (!input_file) {
        throw std::runtime_error("Error: Index file is corrupted or truncated.");
    }
    _count = file_count;
    input_file.close();
}
// Getter: Get Count
size_t VectorIndex::get_count() const{
    return _count;
}
// Getter: Get Dimension
size_t VectorIndex::get_dimension() const{
    return _dimension;
}
