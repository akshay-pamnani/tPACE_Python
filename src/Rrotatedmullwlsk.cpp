#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

Eigen::VectorXd rotatedmullwlsk(const Eigen::VectorXd& bw,
                                 const std::string& kernel_type,
                                 const Eigen::MatrixXd& tPairs,
                                 const Eigen::MatrixXd& cxxn,
                                 const Eigen::VectorXd& win,
                                 const Eigen::MatrixXd& xygrid,
                                 const unsigned int npoly,
                                 const bool bwCheck) {

    const double invSqrt2pi = 1. / std::sqrt(2. * M_PI);

    std::map<std::string, int> possibleKernels = {
        {"epan", 1}, {"rect", 2}, {"gauss", 3}, {"gausvar", 4}, {"quar", 5}
    };

    int KernelName = possibleKernels.count(kernel_type) ? possibleKernels[kernel_type] : 1;

    if ((win.array() == 0).any()) {
        throw std::runtime_error("Cases with zero-valued windows are not yet implemented");
    }

    Eigen::Matrix2d RC;
    RC << 1, -1, 1, 1;
    RC *= std::sqrt(2.) / 2.;

    Eigen::MatrixXd rtPairs = RC * tPairs;
    Eigen::MatrixXd rxygrid = RC * xygrid;

    unsigned int xygridN = rxygrid.cols();
    Eigen::VectorXd mu(xygridN);
    mu.setZero();

    for (unsigned int i = 0; i < xygridN; ++i) {
        std::vector<unsigned int> indx;

        if (KernelName != 3 && KernelName != 4) {
            std::vector<unsigned int> list1, list2;

            for (unsigned int y = 0; y < tPairs.cols(); ++y) {
                if (std::abs(rtPairs(0, y) - rxygrid(0, i)) <= bw(0)) list1.push_back(y);
                if (std::abs(rtPairs(1, y) - rxygrid(1, i)) <= bw(1)) list2.push_back(y);
            }

            std::sort(list1.begin(), list1.end());
            std::sort(list2.begin(), list2.end());
            std::set_intersection(list1.begin(), list1.end(), list2.begin(), list2.end(),
                                  std::back_inserter(indx));
        } else {
            for (unsigned int y = 0; y < tPairs.cols(); ++y) {
                indx.push_back(y);
            }
        }

        unsigned int indxSize = indx.size();
        Eigen::VectorXd lw(indxSize), ly(indxSize);
        Eigen::MatrixXd lx(2, indxSize);

        for (unsigned int u = 0; u < indxSize; ++u) {
            lx.col(u) = rtPairs.col(indx[u]);
            lw(u) = win(indx[u]);
            ly(u) = cxxn(indx[u]);
        }

        if (ly.size() >= npoly + 1 && !bwCheck) {
            Eigen::VectorXd temp(indxSize);
            Eigen::MatrixXd llx(2, indxSize);
            llx.row(0) = (lx.row(0).array() - rxygrid(0, i)) / bw(0);
            llx.row(1) = (lx.row(1).array() - rxygrid(1, i)) / bw(1);

            switch (KernelName) {
                case 1: // epan
                    temp = ((1 - llx.row(0).array().square()) * (1 - llx.row(1).array().square())).array()
                           * ((9. / 16) * lw.array());
                    break;
                case 2: // rect
                    temp = 0.25 * lw.array();
                    break;
                case 3: // gauss
                    temp = ((-0.5 * llx.row(1).array().square()).exp() * invSqrt2pi *
                            (-0.5 * llx.row(0).array().square()).exp() * invSqrt2pi *
                            lw.array());
                    break;
                case 4: // gausvar
                    temp = lw.array()
                        * (-0.5 * llx.row(0).array().square()).exp() * invSqrt2pi
                        * (-0.5 * llx.row(1).array().square()).exp() * invSqrt2pi
                        * (1.25 - 0.25 * llx.row(0).array().square())
                        * (1.50 - 0.50 * llx.row(1).array().square());
                    break;
                case 5: // quar
                    temp = lw.array()
                        * (1 - llx.row(0).array().square()).pow(2)
                        * (1 - llx.row(1).array().square()).pow(2)
                        * (225. / 256.);
                    break;
            }

            Eigen::MatrixXd X(indxSize, 3);
            X.setOnes();
            X.col(1) = (lx.row(0).array() - rxygrid(0, i)).square().transpose();
            X.col(2) = (lx.row(1).array() - rxygrid(1, i)).transpose();

            Eigen::LDLT<Eigen::MatrixXd> ldlt_XTWX(X.transpose() * temp.asDiagonal() * X);
            Eigen::VectorXd beta = ldlt_XTWX.solve(X.transpose() * temp.asDiagonal() * ly);
            mu(i) = beta(0);

        } else if (ly.size() == 1 && !bwCheck) {
            mu(i) = ly(0);
        } else if (ly.size() != 1 && (ly.size() < npoly + 1)) {
            if (bwCheck) {
                Eigen::VectorXd checker(1);
                checker(0) = 0.;
                return checker;
            } else {
                throw std::runtime_error("Not enough points in local window, increase bandwidth.");
            }
        }
    }

    if (bwCheck) {
        Eigen::VectorXd checker(1);
        checker(0) = 1.;
        return checker;
    }

    return mu;
}

PYBIND11_MODULE(Rrotatedmullwlsk, m) {
    m.doc() = "Local weighted kernel smoother with rotated coordinates";
    m.def("rotatedmullwlsk", &rotatedmullwlsk, "Kernel smoother",
          py::arg("bw"), py::arg("kernel_type"), py::arg("tPairs"),
          py::arg("cxxn"), py::arg("win"), py::arg("xygrid"),
          py::arg("npoly"), py::arg("bwCheck"));
}
