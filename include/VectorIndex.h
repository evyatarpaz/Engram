#ifndef VECTOR_INDEX_H
#define VECTOR_INDEX_H

#include <immintrin.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Standard-compliant Custom Aligned Allocator
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  AlignedAllocator() noexcept = default;
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

  // Required by GCC 15 to rebind the allocator to different types
  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    void* ptr = _mm_malloc(n * sizeof(T), Alignment);
    if (!ptr) throw std::bad_alloc();
    return static_cast<T*>(ptr);
  }

  void deallocate(T* p, std::size_t) noexcept { _mm_free(p); }

  // Required by GCC 15 for allocator comparison
  bool operator==(const AlignedAllocator&) const noexcept { return true; }
  bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

class VectorIndex {
 private:
  const size_t _dimension;
  size_t _padded_dimension;
  size_t _count;

  std::vector<float, AlignedAllocator<float, 32>> _data;

  float calculate_squared_distance(const float* vec_a,
                                   const float* vec_b) const;

 public:
  VectorIndex(size_t dimension);

  void add_vector(const std::vector<float>& vec);
  void delete_vector(size_t index);

  std::vector<std::pair<size_t, float>> search(const std::vector<float>& query,
                                               int k = 1);

  void save_index(const std::string& filepath) const;
  void load_index(const std::string& filepath);

  size_t get_count() const;
  size_t get_dimension() const;
};

#endif  // VECTOR_INDEX_H