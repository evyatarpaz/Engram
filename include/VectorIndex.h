#ifndef VECTOR_INDEX_H
#define VECTOR_INDEX_H

#include <vector>
#include <string>
#include <utility>

class VectorIndex {

private:
    // -- Members Variables --

    // The main data store. 
    // We use a "Flat Buffer" approach: a single 1D vector storing all data sequentially.
    // Layout: [vec1_dim1, vec1_dim2, ..., vec2_dim1, vec2_dim2, ...]
    // Why? It's much faster (CPU Cache friendly) than vector<vector<float>>.
    std::vector<float> _data;

    // the fixed number of dimensions for each vector(1536 in openai embeddings)
    const size_t _dimension;

    // How many vectors are currently stored.
    size_t _count;

    // -- Private Methods --

    // Calculates the squared Euclidean distance between two vectors.
    // We pass pointers (float*) to avoid data copying for maximum performance.

    float calculate_distance(const float* vec_a, const float* vec_b) const;



public:
    // --- Public API ---

    // Constructor: Initializes the DB with a fixed dimension size.
    VectorIndex(size_t dimension);

    // Adds a new vector to the index.
    // Throws std::invalid_argument if vec.size() != _dimension.
    void add_vector(const std::vector<float>& vec);

    // Finds the 'k' nearest neighbors to the query vector.
    // Returns: A vector of pairs {Vector_ID, Distance}, sorted by best match.
    std::vector<std::pair<size_t, float>> search(const std::vector<float>& query, int k = 1);

    // Saves the entire index to a binary file on disk.
    void save_index(const std::string& filepath) const;

    // Loads the index from a binary file (overwrites current memory).
    void load_index(const std::string& filepath);
    
    // Getters
    size_t get_count() const;
    size_t get_dimension() const;
};

#endif // VECTOR_INDEX_H