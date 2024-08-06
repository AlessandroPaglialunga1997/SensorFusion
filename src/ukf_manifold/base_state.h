#pragma once
#include <Eigen/Core>

using Vector9d = Eigen::Matrix<double, 9, 1>;
using Matrix9d = Eigen::Matrix<double, 9, 9>;

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

namespace ukf_manifold {

  template <int StateDim_, int InputDim_>
  class BaseState_ {
  public:
    static constexpr int StateDim = StateDim_;
    static constexpr int InputDim = InputDim_;
    using StateVectorType         = Eigen::Matrix<double, StateDim, 1>;
    using CovMatrixType           = Eigen::Matrix<double, StateDim, StateDim>;
    using InputVectorType         = Eigen::Matrix<double, InputDim, 1>;

    BaseState_() {
    }
    virtual void transition(const InputVectorType& input, const InputVectorType& noise, const double& dt = 0.0);
    virtual BaseState_ boxplus(const StateVectorType& xi);
    virtual StateVectorType boxminus(const BaseState_& state_hat);

  protected:
    CovMatrixType covariance_ = CovMatrixType::Zero();
  };

} // namespace ukf_manifold
