#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <limits>

namespace py = pybind11;

// This function corresponds to GetIndCEScoresCPP
py::dict GetIndCEScoresCPP(
    const Eigen::VectorXd& yVec,
    const Eigen::VectorXd& muVec,
    const Eigen::VectorXd& lamVec,
    const Eigen::MatrixXd& phiMat,
    const Eigen::MatrixXd& SigmaYi
) {
    const unsigned int lenlamVec = lamVec.size();

    Eigen::MatrixXd xiEst = Eigen::MatrixXd::Constant(lenlamVec, 1, std::numeric_limits<double>::quiet_NaN());
    Eigen::MatrixXd xiVar = Eigen::MatrixXd::Constant(lenlamVec, lenlamVec, std::numeric_limits<double>::quiet_NaN());
    Eigen::MatrixXd fittedY = Eigen::MatrixXd::Constant(muVec.size(), 1, std::numeric_limits<double>::quiet_NaN());

    Eigen::MatrixXd LamPhi = lamVec.asDiagonal() * phiMat.transpose();
    Eigen::LDLT<Eigen::MatrixXd> ldlt_SigmaYi(SigmaYi);

    xiEst = LamPhi * ldlt_SigmaYi.solve(yVec - muVec);
    xiVar = -LamPhi * ldlt_SigmaYi.solve(LamPhi.transpose());
    xiVar.diagonal() += lamVec;

    fittedY = muVec + phiMat * xiEst;

    return py::dict(
        "xiEst"_a = xiEst,
        "xiVar"_a = xiVar,
        "fittedY"_a = fittedY
    );
}

PYBIND11_MODULE(GetIndCEScoresCPP, m) {
    m.doc() = "Basic Conditional Expectation Score Calculation (Pybind11)";
    m.def("get_ind_ce_scores_cpp", &GetIndCEScoresCPP, "Compute CE scores");
}
