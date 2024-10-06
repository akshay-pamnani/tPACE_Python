#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversions of STL types
#include <limits> // To get NaN
#include <vector>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

// Function to check if a range is sorted (renamed to avoid conflict)
template <class iter>
bool custom_is_sorted(iter begin, iter end) {
    if (begin == end) return true;
    iter next = begin;
    while (++next != end) {
        if (*next < *begin)
            std::cout << "Comparing *next: " << *next << " and *begin: " << *begin << std::endl;
            return false;
        ++begin;
    }
    return true;
}

// Trapezoid Rule Numerical Integration
double trapz(const std::vector<double> &X, const std::vector<double> &Y) {
    if (Y.size() != X.size()) {
        throw std::invalid_argument("The input Y-grid does not have the same number of points as input X-grid.");
    }
    if (custom_is_sorted(X.begin(), X.end())) {
        double trapzsum = 0;
        for (unsigned int ind = 0; ind != X.size() - 1; ++ind) {
            trapzsum += 0.5 * (X[ind + 1] - X[ind]) * (Y[ind] + Y[ind + 1]);
        }
        return trapzsum;
    } else {
        std::cout << X.begin() << X.end() << std::endl;
        throw std::invalid_argument("The input X-grid is not sorted.");
    }
}

// Create Python module
PYBIND11_MODULE(trapzRcpp, m) {
    m.def("trapz", &trapz, "Trapezoid Rule Numerical Integration",
          py::arg("X"), py::arg("Y"));
}
