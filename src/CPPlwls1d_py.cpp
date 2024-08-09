#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

namespace py = pybind11;

Eigen::VectorXd CPPlwls1d(const double & bw, const std::string kernel_type, const Eigen::VectorXd & win, const Eigen::VectorXd & xin, const Eigen::VectorXd & yin, const Eigen::VectorXd & xout, const unsigned int & npoly = 1, const unsigned int & nder = 0) {
    const double invSqrt2pi= 1./(sqrt(2.*M_PI));
    const double factorials[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800};

    const unsigned int nXGrid = xin.size();
    const unsigned int nUnknownPoints = xout.size();
    Eigen::VectorXd result(nUnknownPoints);

    if(nXGrid == 0) {
        throw std::invalid_argument("The input X-grid has length zero.");
    }

    if(nXGrid != yin.size()){
        throw std::invalid_argument("The input Y-grid does not have the same number of points as input X-grid.");
    }

    if(nXGrid != win.size()){
        throw std::invalid_argument("The input weight vector does not have the same number of points as input X-grid.");
    }

    if(bw <= 0.) {
        throw std::invalid_argument("The bandwidth supplied for 1-D smoothing is not positive.");
    }

    if(npoly < nder) {
        throw std::invalid_argument("The degree of polynomial supplied for 1-D smoothing is less than the order of derivative");
    }

    std::map<std::string, int> possibleKernels;
    possibleKernels["epan"]    = 1;
    possibleKernels["rect"]    = 2;
    possibleKernels["gauss"]   = 3;
    possibleKernels["gausvar"] = 4;
    possibleKernels["quar"]    = 5;

    int KernelName = 0;
    if(possibleKernels.count(kernel_type) != 0) {
        KernelName = possibleKernels[kernel_type];
    } else {
        KernelName = possibleKernels["epan"];
    }

    if(!(win.all())) {
        throw std::invalid_argument("Cases with zero-valued windows may not be safe.");
    }

    if(!(std::is_sorted(xin.data(), xin.data() + nXGrid))) {
        throw std::invalid_argument("The X-grid used is not sorted.");
    }

    for(unsigned int i = 0; i != nUnknownPoints; ++i) {
        std::vector<unsigned int> indx;
        const double* lower;
        const double* upper;

        if(KernelName != 3 && KernelName != 4) {
            lower = std::lower_bound(xin.data(), xin.data() + nXGrid, xout(i) - bw);
            upper = std::lower_bound(xin.data(), xin.data() + nXGrid, xout(i) + bw);
        } else {
            lower = xin.data();
            upper = xin.data() + nXGrid;
        }

        const unsigned int firstElement = lower - &xin[0];
        for(unsigned int xx1 = 0; xx1 != upper - lower; ++xx1) {
            indx.push_back(firstElement + xx1);
        }

        unsigned int indxSize = indx.size();
        Eigen::VectorXd temp(indxSize);
        Eigen::VectorXd lw(indxSize);
        Eigen::VectorXd ly(indxSize);
        Eigen::VectorXd lx(indxSize);

        for(unsigned int y = 0; y != indxSize; ++y) {
            lx(y) = xin(indx[y]);
            lw(y) = win(indx[y]);
            ly(y) = yin(indx[y]);
        }

        Eigen::VectorXd llx = (lx.array() - xout(i)) * (1./bw);

        switch(KernelName) {
            case 1:
                temp = (1 - llx.array().pow(2)) * 0.75 * (lw.array());
                break;
            case 2:
                temp = lw;
                break;
            case 3:
                temp = ((-.5 * (llx.array().pow(2))).exp()) * invSqrt2pi * (lw.array());
                break;
            case 4:
                temp = (lw.array()) * ((-0.5 * llx.array().pow(2)).array().exp() * invSqrt2pi).array() * (1.25 - (0.25 * (llx.array().pow(2))).array());
                break;
            case 5:
                temp = (lw.array()) * ((1 - llx.array().pow(2)).array().pow(2)) * (15./16.);
                break;
        }

        if(nder >= indxSize) {
            result(i) = std::numeric_limits<double>::quiet_NaN();
        } else {
            Eigen::MatrixXd X(indxSize, npoly + 1);
            X.setOnes();
            for(unsigned int y = 1; y <= npoly; ++y) {
                X.col(y) = (xout(i) - lx.array()).array().pow(y);
            }

            Eigen::LDLT<Eigen::MatrixXd> ldlt_XTWX(X.transpose() * temp.asDiagonal() * X);
            Eigen::VectorXd beta = ldlt_XTWX.solve(X.transpose() * temp.asDiagonal() * ly);
            result(i) = beta(nder + 0) * factorials[nder] * std::pow(-1.0, int(nder));
        }
    }

    return result;
}

PYBIND11_MODULE(CPPlwls1d_py, m) {
    m.def("CPPlwls1d", &CPPlwls1d, "Local Weighted Least Squares (LWLS) smoothing function",
          py::arg("bw"), py::arg("kernel_type"), py::arg("win"), py::arg("xin"), py::arg("yin"), py::arg("xout"), py::arg("npoly") = 1, py::arg("nder") = 0);
}
