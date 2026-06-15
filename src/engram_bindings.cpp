#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/VectorIndex.h"

namespace py = pybind11;

PYBIND11_MODULE(engram, m) {
  m.doc() = "Engram: A fast C++ Vector Database with AVX2 SIMD optimizations";

  py::class_<VectorIndex>(m, "VectorIndex")
      .def(py::init<size_t>(), py::arg("dimension"))
      .def("add_vector", &VectorIndex::add_vector, "Add a vector to the index")
      .def("delete_vector", &VectorIndex::delete_vector, py::arg("index"),
           "Delete a vector in O(1) time without shifting memory")
      .def("search", &VectorIndex::search, py::arg("query"), py::arg("k") = 1,
           "Search for k-nearest neighbors using Max-Heap")
      .def("save_index", &VectorIndex::save_index, "Save padded index to disk")
      .def("load_index", &VectorIndex::load_index,
           "Load padded index from disk")
      .def_property_readonly("count", &VectorIndex::get_count)
      .def_property_readonly("dimension", &VectorIndex::get_dimension);
}