#pragma once
#include "base_state.h"
#include "lie_algebra.h"
#include <iostream>
#include <memory>

namespace ukf_manifold {

  class OdomState {
  public:
    OdomState();
    virtual ~OdomState() = default;

    // given noisy input, update prev state with transition dunction
    // madre
    void transition(const Vector6d& input, const Vector6d& noise, const double& dt = 0.0);
    std::unique_ptr<OdomState> boxplus(const Eigen::VectorXd& xi);
    std::unique_ptr<Eigen::VectorXd> boxminus(const OdomState& state_hat);

    // accessors
    inline const Eigen::Matrix3d& rotation() const {
      return R_;
    }
    inline const Eigen::Vector3d& position() const {
      return p_;
    }
    inline const Matrix6d& covariance() const {
      return covariance_;
    }

    // setters
    inline Eigen::Matrix3d& rotation() {
      return R_;
    }
    inline Eigen::Vector3d& position() {
      return p_;
    }
    inline Matrix6d& covariance() {
      return covariance_;
    }

    // dim state for dynamic shit
    inline const int stateDim() const {
      return state_dim_;
    }
    inline const int inputDim() const {
      return input_dim_;
    }

    friend std::ostream& operator<<(std::ostream& os, const OdomState& state_);

  protected:
    Eigen::Matrix3d R_   = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_   = Eigen::Vector3d::Zero();
    Matrix6d covariance_ = Matrix6d::Zero();
    int state_dim_       = 6;
    int input_dim_       = 6;
  };

} // namespace ukf_manifold