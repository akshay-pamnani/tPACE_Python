#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <limits>

namespace py = pybind11;

Eigen::MatrixXd pinv(const Eigen::MatrixXd& mat) {
    const double pinvtol = 1.e-9;
    Eigen::JacobiSVD<Eigen::MatrixXd> svdMat(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd S = svdMat.singularValues();
    Eigen::VectorXd Sinv = S;

    for (int i = 0; i < mat.cols(); ++i) {
        if (S(i) > pinvtol) {
            Sinv(i) = 1.0 / S(i);
        } else {
            Sinv(i) = 0;
        }
    }
    return svdMat.matrixV() * Sinv.asDiagonal() * svdMat.matrixU().transpose();
}

py::dict GetIndCEScoresCPPnewInd(
    const Eigen::VectorXd& yVec,
    const Eigen::VectorXd& muVec,
    const Eigen::VectorXd& lamVec,
    const Eigen::MatrixXd& phiMat,
    const Eigen::MatrixXd& SigmaYi,
    const Eigen::MatrixXd& newPhi,
    const Eigen::VectorXd& newMu
) {
    const unsigned int lenlamVec = lamVec.size();

    Eigen::MatrixXd xiEst = Eigen::MatrixXd::Constant(lenlamVec, 1, std::numeric_limits<double>::quiet_NaN());
    Eigen::MatrixXd xiVar = Eigen::MatrixXd::Constant(lenlamVec, lenlamVec, std::numeric_limits<double>::quiet_NaN());
    Eigen::MatrixXd fittedY = Eigen::MatrixXd::Constant(newPhi.rows(), 1, std::numeric_limits<double>::quiet_NaN());

    // LDLT decomposition is preferred over pinv for stability
    Eigen::LDLT<Eigen::MatrixXd> ldlt_SigmaYi(SigmaYi);
    Eigen::MatrixXd LamPhi = lamVec.asDiagonal() * phiMat.transpose();

    xiEst = LamPhi * ldlt_SigmaYi.solve(yVec - muVec);
    xiVar = -LamPhi * ldlt_SigmaYi.solve(LamPhi.transpose());
    xiVar.diagonal() += lamVec;

    fittedY = newMu + newPhi * xiEst;

    return py::dict(
        "xiEst"_a=xiEst,
        "xiVar"_a=xiVar,
        "fittedY"_a=fittedY
    );
}

PYBIND11_MODULE(GetIndCEScoresCPPnewInd, m) {
    m.doc() = "Conditional Expectation Score Calculation (C++/Eigen, Pybind11)";
    m.def("get_ind_ce_scores_cpp_new_ind", &GetIndCEScoresCPPnewInd, "Compute CE scores with new index");
}
