#include "odometry_state.h"

namespace ukf_manifold {

  /** nav state without bias begin **/
  OdomState::OdomState() {
  }

  void OdomState::transition(const Vector6d& input, const Vector6d& noise, const double& dt) {
    const Eigen::Vector3d& position_noise    = noise.head(3);
    const Eigen::Vector3d& orientation_noise = noise.tail(3);
    const Eigen::Vector3d& position_input    = input.head(3);
    const Eigen::Vector3d& orientation_input = input.tail(3);

    p_ = p_ + R_ * (position_input + position_noise);
    R_ = R_ * lie_algebra::SO3Exp((Eigen::Vector3d) orientation_input + orientation_noise);
    fixRotation(R_);
  }

  std::unique_ptr<OdomState> OdomState::boxplus(const Eigen::VectorXd& xi) {
    // copy whole state
    std::unique_ptr<OdomState> odom_state = std::make_unique<OdomState>(*this);
    const Eigen::Vector3d& p_pert         = xi.head(3);
    const Eigen::Vector3d& r_pert         = xi.tail(3); // starting at position 3 get 3 elems
    odom_state->position() += p_pert;
    odom_state->rotation() = R_ * lie_algebra::SO3Exp(r_pert);
    return odom_state;
  }

  std::unique_ptr<Eigen::VectorXd> OdomState::boxminus(const OdomState& state_hat) {
    std::unique_ptr<Eigen::VectorXd> xi = std::make_unique<Eigen::VectorXd>(state_dim_);
    const Eigen::Matrix3d R_pert        = state_hat.rotation() * R_.transpose();
    xi->head(3)                         = state_hat.position() - p_;
    xi->tail(3)                         = lie_algebra::SO3Log(R_pert);
    return xi;
  }

  std::ostream& operator<<(std::ostream& os, const OdomState& state) {
    os << "p: " << state.p_.transpose() << "\n"
       << "R:\n"
       << state.R_;
    return os;
  }
} // namespace ukf_manifold