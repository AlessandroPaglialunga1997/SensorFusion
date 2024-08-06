#pragma once
#include "base_state.h"
#include "lie_algebra.h"
#include <iostream>
#include <memory>

namespace ukf_manifold {

  class NavState {
  public:
    // static constexpr int StateDim = 9;
    // static constexpr int InputDim = 6;

    NavState();
    virtual ~NavState() = default;

    // given noisy input, update prev state with transition dunction
    // madre
    void transition(const Vector6d& input, const Vector6d& noise, const double& dt);
    std::unique_ptr<NavState> boxplus(const Eigen::VectorXd& xi);
    std::unique_ptr<Eigen::VectorXd> boxminus(const NavState& state_hat);

    // accessors
    inline const Eigen::Matrix3d& rotation() const {
      return R_;
    }
    inline const Eigen::Vector3d& position() const {
      return p_;
    }
    inline const Eigen::Vector3d& velocity() const {
      return v_;
    }
    inline const Matrix9d& covariance() const {
      return covariance_;
    }

    // setters
    inline Eigen::Matrix3d& rotation() {
      return R_;
    }
    inline Eigen::Vector3d& position() {
      return p_;
    }
    inline Eigen::Vector3d& velocity() {
      return v_;
    }
    inline Matrix9d& covariance() {
      return covariance_;
    }

    // dim state for dynamic shit
    inline const int stateDim() const {
      return state_dim_;
    }
    inline const int inputDim() const {
      return input_dim_;
    }

    friend std::ostream& operator<<(std::ostream& os, const NavState& state);

  protected:
    Eigen::Matrix3d R_   = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p_   = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_   = Eigen::Vector3d::Zero();
    Matrix9d covariance_ = Matrix9d::Zero();
    Eigen::Vector3d g_   = Eigen::Vector3d(0.d, 0.d, -9.82);
    size_t state_dim_    = Matrix9d::RowsAtCompileTime;
    size_t input_dim_    = 6;
  };

  using Vector15d = Eigen::Matrix<double, 15, 1>;
  using Matrix15d = Eigen::Matrix<double, 15, 15>;

  class NavStateExtended : public NavState {
  public:
    // static constexpr int StateDim = 15;
    // static constexpr int InputDim = 6;

    NavStateExtended();
    virtual ~NavStateExtended() = default;

    void transition(const Vector6d& input, const Vector6d& noise, const double& dt);
    std::unique_ptr<NavStateExtended> boxplus(const Eigen::VectorXd& xi);
    std::unique_ptr<Eigen::VectorXd> boxminus(const NavStateExtended& state_hat);

    inline const Eigen::Vector3d& biasAcc() const {
      return bias_acc_;
    } // getters
    inline const Eigen::Vector3d& biasGyro() const {
      return bias_gyro_;
    }
    inline const Matrix15d& covariance() const {
      return covariance_;
    }

    inline Eigen::Vector3d& biasAcc() {
      return bias_acc_;
    } // setters
    inline Eigen::Vector3d& biasGyro() {
      return bias_gyro_;
    }
    inline Matrix15d& covariance() {
      return covariance_;
    }

    inline const int stateDim() const {
      return state_dim_;
    }

    friend std::ostream& operator<<(std::ostream& os, const NavStateExtended& state);

  protected:
    Eigen::Vector3d bias_acc_  = Eigen::Vector3d::Zero();
    Eigen::Vector3d bias_gyro_ = Eigen::Vector3d::Zero();
    Matrix15d covariance_      = Matrix15d::Zero();
    int state_dim_             = Matrix15d::RowsAtCompileTime;
  };

} // namespace ukf_manifold