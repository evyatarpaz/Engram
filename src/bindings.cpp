#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Magic header that automatically converts std::vector to Python List
#include "../include/VectorIndex.h"

namespace py = pybind11;

// PYBIND11_MODULE(module_name, variable_name)
// This macro creates the entry point for the Python extension.
PYBIND11_MODULE(engram, m) {
    m.doc() = "Engram: A fast C++ Vector Database"; // Module docstring

    // Expose the VectorIndex class to Python
    py::class_<VectorIndex>(m, "VectorIndex")
        // Expose the Constructor: __init__(dimension)
        .def(py::init<size_t>(), py::arg("dimension"))
        
        // Expose methods
        .def("add_vector", &VectorIndex::add_vector, "Add a vector to the index")
        
        // Expose search with default arguments (k=1)
        .def("search", &VectorIndex::search, py::arg("query"), py::arg("k") = 1, "Search for k-nearest neighbors")
        
        // Expose persistence methods
        .def("save_index", &VectorIndex::save_index, "Save index to disk")
        .def("load_index", &VectorIndex::load_index, "Load index from disk")
        
        // Expose getters as Python properties (allows usage like: db.count instead of db.get_count())
        .def_property_readonly("count", &VectorIndex::get_count)
        .def_property_readonly("dimension", &VectorIndex::get_dimension);
}