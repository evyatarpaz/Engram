#include "../include/VectorIndex.h"
#include <cmath>        
#include <fstream>      
#include <algorithm>    
#include <iostream>
#include <queue>

VectorIndex::VectorIndex(size_t dimension) : _dimension(dimension), _count(0) {
    // Pad dimension to nearest multiple of 8 to maintain alignment across vectors
    _padded_dimension = (_dimension + 7) & ~7;
    _data.reserve(1000 * _padded_dimension);
}

float VectorIndex::calculate_squared_distance(const float* vec_a, const float* vec_b) const {
    __m256 sum_vec = _mm256_setzero_ps();
    
    // Process strictly using aligned loads
    for (size_t i = 0; i < _padded_dimension; i += 8) {
        __m256 a = _mm256_load_ps(&vec_a[i]); 
        __m256 b = _mm256_load_ps(&vec_b[i]);
        __m256 diff = _mm256_sub_ps(a, b);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }

    float temp[8];
    _mm256_store_ps(temp, sum_vec); 
    return temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
}

void VectorIndex::add_vector(const std::vector<float>& vec) {
    if (vec.size() != _dimension) throw std::invalid_argument("Vector dimension does not match index.");
    
    size_t start_idx = _data.size();
    // Resize automatically populates padding space with 0.0f
    _data.resize(start_idx + _padded_dimension, 0.0f); 
    std::copy(vec.begin(), vec.end(), _data.begin() + start_idx);
    _count++;
}

void VectorIndex::delete_vector(size_t index) {
    if (index >= _count) throw std::out_of_range("Index out of bounds.");
    
    size_t last_index = _count - 1;
    if (index != last_index) {
        std::copy(_data.begin() + last_index * _padded_dimension, 
                  _data.begin() + (last_index + 1) * _padded_dimension, 
                  _data.begin() + index * _padded_dimension);
    }
    _data.resize(last_index * _padded_dimension);
    _count--;
}

std::vector<std::pair<size_t, float>> VectorIndex::search(const std::vector<float>& query, int k) {
    if (query.size() != _dimension) throw std::invalid_argument("Query dimension mismatch.");
    
    // Create an aligned, padded copy of the query to safely execute SIMD comparisons
    std::vector<float, AlignedAllocator<float, 32>> padded_query(_padded_dimension, 0.0f);
    std::copy(query.begin(), query.end(), padded_query.begin());

    // Max-heap for O(N log k) top-k extraction
    auto cmp = [](const std::pair<size_t, float>& left, const std::pair<size_t, float>& right) {
        return left.second < right.second; 
    };
    std::priority_queue<std::pair<size_t, float>, std::vector<std::pair<size_t, float>>, decltype(cmp)> max_heap(cmp);

    for (size_t i = 0; i < _count; ++i) {
        float dist_sq = calculate_squared_distance(&_data[i * _padded_dimension], padded_query.data());
        
        if (max_heap.size() < static_cast<size_t>(k)) {
            max_heap.emplace(i, dist_sq);
        } else if (dist_sq < max_heap.top().second) {
            max_heap.pop();
            max_heap.emplace(i, dist_sq);
        }
    }

    std::vector<std::pair<size_t, float>> results;
    results.reserve(max_heap.size());
    while (!max_heap.empty()) {
        results.push_back(max_heap.top());
        max_heap.pop();
    }
    std::reverse(results.begin(), results.end());
    return results;
}

void VectorIndex::save_index(const std::string& filepath) const {
    std::ofstream output_file(filepath, std::ios::binary);
    if(!output_file.is_open()) throw std::runtime_error("Could not open file for writing: " + filepath); 

    output_file.write(reinterpret_cast<const char*>(&_dimension), sizeof(_dimension));
    output_file.write(reinterpret_cast<const char*>(&_padded_dimension), sizeof(_padded_dimension));
    output_file.write(reinterpret_cast<const char*>(&_count), sizeof(_count));

    if(!_data.empty()) {
        output_file.write(reinterpret_cast<const char*>(_data.data()), _data.size() * sizeof(float));
    }
    output_file.close();
}

void VectorIndex::load_index(const std::string& filepath) {
    std::ifstream input_file(filepath, std::ios::binary);
    if(!input_file.is_open()) throw std::runtime_error("Could not open file for reading: " + filepath);
    
    size_t file_dimension = 0;
    size_t file_padded_dimension = 0;
    size_t file_count = 0;

    input_file.read(reinterpret_cast<char*>(&file_dimension), sizeof(file_dimension));
    input_file.read(reinterpret_cast<char*>(&file_padded_dimension), sizeof(file_padded_dimension));
    input_file.read(reinterpret_cast<char*>(&file_count), sizeof(file_count));

    if(file_dimension != _dimension) throw std::runtime_error("Dimension mismatch when loading.");

    _padded_dimension = file_padded_dimension;
    _data.resize(file_count * _padded_dimension);
    
    if (file_count > 0) {
        input_file.read(reinterpret_cast<char*>(_data.data()), _data.size() * sizeof(float));
    }
    
    if (!input_file) throw std::runtime_error("Index file is corrupted or truncated.");
    
    _count = file_count;
    input_file.close();
}

size_t VectorIndex::get_count() const { return _count; }
size_t VectorIndex::get_dimension() const { return _dimension; }