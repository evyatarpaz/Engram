#include <iostream>
#include <vector>
#include <cassert>  // For assertions (crashes program if condition is false)
#include <cmath>
#include <filesystem> // To clean up temp files
#include "../include/VectorIndex.h"

// Helper function for floating point comparison (due to small precision errors)
bool is_close(float a, float b, float epsilon = 1e-4) {
    return std::abs(a - b) < epsilon;
}

void test_initialization() {
    std::cout << "[Running] test_initialization..." << std::endl;
    VectorIndex idx(128);
    
    // Validate initial state
    assert(idx.get_dimension() == 128);
    assert(idx.get_count() == 0);
    
    std::cout << "[Passed] test_initialization" << std::endl;
}

void test_add_and_search() {
    std::cout << "[Running] test_add_and_search..." << std::endl;
    
    // Create an index with 3 dimensions for easy mental calculation
    VectorIndex idx(3);
    
    // Vector 0: [1, 0, 0]
    idx.add_vector({1.0f, 0.0f, 0.0f}); 
    // Vector 1: [0, 1, 0]
    idx.add_vector({0.0f, 1.0f, 0.0f}); 
    
    assert(idx.get_count() == 2);

    // Search for a vector close to ID 0: [0.9, 0.1, 0]
    // Expected distance: sqrt((1-0.9)^2 + (0-0.1)^2) = sqrt(0.01 + 0.01) ~= 0.1414
    std::vector<float> query = {0.9f, 0.1f, 0.0f};
    auto results = idx.search(query, 1); // Request top 1 result
    
    assert(!results.empty());
    assert(results[0].first == 0); // Must find ID 0
    assert(is_close(results[0].second, 0.1414f)); // Validate math accuracy
    
    std::cout << "[Passed] test_add_and_search" << std::endl;
}

void test_persistence() {
    std::cout << "[Running] test_persistence..." << std::endl;
    
    std::string filename = "test_index.bin";
    
    // Phase 1: Create and Save
    {
        VectorIndex idx(2);
        idx.add_vector({10.0f, 20.0f});
        idx.add_vector({30.0f, 40.0f});
        idx.save_index(filename);
    } // 'idx' goes out of scope here and is destroyed

    // Phase 2: Load and Verify
    {
        VectorIndex idx_loaded(2); // Dimension must match the file
        idx_loaded.load_index(filename);
        
        assert(idx_loaded.get_count() == 2);
        
        // Check if search still works on loaded data
        // Search for vector close to {10, 20}
        auto res = idx_loaded.search({10.1f, 20.1f}, 1);
        assert(res[0].first == 0);
    }
    
    std::cout << "[Passed] test_persistence" << std::endl;
}

void test_invalid_input() {
    std::cout << "[Running] test_invalid_input..." << std::endl;
    VectorIndex idx(5);
    
    try {
        idx.add_vector({1.0f, 2.0f}); // Error: Vector too short (2 dims instead of 5)
        assert(false); // Should not reach here, exception expected
    } catch (const std::invalid_argument& e) {
        // Success: Exception caught
    }
    std::cout << "[Passed] test_invalid_input" << std::endl;
}

int main() {
    std::cout << "=== Starting Unit Tests for Engram Engine ===" << std::endl;
    
    test_initialization();
    test_add_and_search();
    test_persistence();
    test_invalid_input();
    
    std::cout << "=== All Tests Passed Successfully ===" << std::endl;
    return 0;
}