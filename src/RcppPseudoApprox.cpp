#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

float LinearInterpolation(const Eigen::VectorXd &X, const Eigen::VectorXd &Y, float X_PointOfInterest) {
    // Ensure the sizes of X and Y match
    if (X.size() != Y.size()) {
        throw std::runtime_error("Unequal vector sizes for linear interpolation.");
    }

    // Check if the point of interest is within bounds
    if (X_PointOfInterest < X(0) || X_PointOfInterest > X(Y.size() - 1)) {
        throw std::runtime_error("Point of interest is outside curve boundaries.");
    }

    float xk, xkp1, yk, ykp1 = 0;

    // Find the points around the point of interest
    for (int i = 1; i < X.size(); i++) {
        if (X(i) >= X_PointOfInterest) {
            xkp1 = X(i);
            xk = X(i - 1);
            ykp1 = Y(i);
            yk = Y(i - 1);
            break;
        }
    }

    // Linear interpolation using point-slope form
    float t = (X_PointOfInterest - xk) / (xkp1 - xk);
    float yPOI = (1 - t) * yk + t * ykp1;
    return yPOI;
}

Eigen::VectorXd PseudoApprox(const Eigen::VectorXd &X, const Eigen::VectorXd &Y, const Eigen::VectorXd &X_target) {
    int N = X_target.size();
    Eigen::VectorXd rr(N);

    for (int i = 0; i < N; i++) {
        rr(i) = LinearInterpolation(X, Y, X_target(i));
    }
    
    return rr;
}

PYBIND11_MODULE(interpolate, m) {
    m.def("linear_interpolation", &LinearInterpolation, "Linear interpolation function");
    m.def("pseudo_approx", &PseudoApprox, "Pseudo Approximation using Linear Interpolation");
}
