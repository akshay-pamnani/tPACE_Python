#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>

namespace py = pybind11;

// Function to sort a numpy array
py::array_t<double> pybind_sort(py::array_t<double> arr) {
    // Request a buffer descriptor from NumPy array
    py::buffer_info buf = arr.request();

    // Pointer to the data as a double array
    double *ptr = static_cast<double *>(buf.ptr);
    size_t size = buf.size;

    // Create a vector and copy data from NumPy array
    std::vector<double> vec(ptr, ptr + size);

    // Sort the vector
    std::sort(vec.begin(), vec.end());

    // Return a new NumPy array with sorted values
    return py::array_t<double>(vec.size(), vec.data());
}

// Binding the C++ function to Python
PYBIND11_MODULE(Rcppsort, m) {
    m.def("pybind_sort", &pybind_sort, "Sort a numpy array using C++");
}
