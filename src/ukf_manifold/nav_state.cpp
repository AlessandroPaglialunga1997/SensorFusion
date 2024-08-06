#include "nav_state.h"

namespace ukf_manifold {

  /** nav state without bias begin **/
  NavState::NavState() {
  }

  void NavState::transition(const Vector6d& input, const Vector6d& noise, const double& dt) {
    const Eigen::Vector3d& acc_noise  = noise.head(3);
    const Eigen::Vector3d& gyro_noise = noise.tail(3);
    const Eigen::Vector3d& acc_input  = input.head(3);
    const Eigen::Vector3d& gyro_input = input.tail(3);

    const Eigen::Vector3d acc    = R_ * (acc_input + acc_noise) + g_; // transdorm acc vec
    const Eigen::Vector3d r_incr = (gyro_input + gyro_noise) * dt;

    // update directly on state variables
    p_ = p_ + v_ * dt + 0.5 * acc * dt * dt;
    v_ = v_ + acc * dt;
    R_ = R_ * lie_algebra::SO3Exp(r_incr);
    fixRotation(R_);
  }

  std::unique_ptr<NavState> NavState::boxplus(const Eigen::VectorXd& xi) {
    // copy whole state
    std::unique_ptr<NavState> nav_state = std::make_unique<NavState>(*this);
    const Eigen::Vector3d& r_pert       = xi.head(3);
    const Eigen::Vector3d& v_pert       = xi.segment(3, 3); // starting at position 3 get 3 elems
    const Eigen::Vector3d& p_pert       = xi.tail(3);
    nav_state->rotation()               = lie_algebra::SO3Exp(r_pert) * R_;
    nav_state->velocity() += v_pert;
    nav_state->position() += p_pert;
    return nav_state;
  }

  std::unique_ptr<Eigen::VectorXd> NavState::boxminus(const NavState& state_hat) {
    std::unique_ptr<Eigen::VectorXd> xi = std::make_unique<Eigen::VectorXd>(state_dim_);
    const Eigen::Matrix3d R_inv_pert    = R_.transpose() * state_hat.rotation();
    xi->head(3)                         = lie_algebra::SO3Log(R_inv_pert);
    xi->segment(3, 3)                   = state_hat.velocity() - v_;
    xi->tail(3)                         = state_hat.position() - p_;
    return xi;
  }

  std::ostream& operator<<(std::ostream& os, const NavState& state) {
    os << "p: " << state.p_.transpose() << "\n"
       << "v: " << state.v_.transpose() << "\n"
       << "R:\n"
       << state.R_;
    return os;
  }

  /** nav state without bias end **/

  /** nav state with bias start **/

  NavStateExtended::NavStateExtended() {
  }

  void NavStateExtended::transition(const Vector6d& input, const Vector6d& noise, const double& dt) {
    const Eigen::Vector3d& gyro_noise = noise.head(3);
    const Eigen::Vector3d& acc_noise  = noise.tail(3);
    const Eigen::Vector3d& gyro_input = input.head(3);
    const Eigen::Vector3d& acc_input  = input.tail(3);

    // unbias, add noise and transform measurements
    const Eigen::Vector3d gyro = gyro_input - bias_gyro_ + gyro_noise;
    const Eigen::Vector3d acc  = R_ * (acc_input - bias_acc_ + acc_noise) + g_;

    // update directly on state variables
    p_ = p_ + v_ * dt + 0.5 * acc * dt * dt;
    v_ = v_ + acc * dt;
    R_ = R_ * lie_algebra::SO3Exp((Eigen::Vector3d) gyro * dt);
    fixRotation(R_);
    // TODO add noise to bias?
  }

  std::unique_ptr<NavStateExtended> NavStateExtended::boxplus(const Eigen::VectorXd& xi) {
    // copy whole state
    std::unique_ptr<NavStateExtended> nav_state = std::make_unique<NavStateExtended>(*this);

    // slice some quantities
    const Eigen::Vector3d& r_pert       = xi.head(3);
    const Eigen::Vector3d& v_pert       = xi.segment(3, 3); // starting at position 3 get 3 elems
    const Eigen::Vector3d& p_pert       = xi.segment(6, 3);
    const Eigen::Vector3d& delta_b_gyro = xi.segment(9, 3);
    const Eigen::Vector3d& delta_b_acc  = xi.tail(3);
    // boxplus
    nav_state->rotation() = lie_algebra::SO3Exp(r_pert) * R_;
    nav_state->velocity() += v_pert;
    nav_state->position() += p_pert;
    nav_state->biasGyro() += delta_b_gyro;
    nav_state->biasAcc() += delta_b_acc;

    return nav_state;
  }

  std::unique_ptr<Eigen::VectorXd> NavStateExtended::boxminus(const NavStateExtended& state_hat) {
    std::unique_ptr<Eigen::VectorXd> xi = std::make_unique<Eigen::VectorXd>(state_dim_);
    const Eigen::Matrix3d R_inv_pert    = state_hat.rotation() * R_.transpose();
    xi->head(3)                         = lie_algebra::SO3Log(R_inv_pert);
    xi->segment(3, 3)                   = state_hat.velocity() - v_;
    xi->segment(6, 3)                   = state_hat.position() - p_;
    xi->segment(9, 3)                   = state_hat.biasGyro() - bias_gyro_;
    xi->tail(3)                         = state_hat.biasAcc() - bias_acc_;
    return xi;
  }

  std::ostream& operator<<(std::ostream& os, const NavStateExtended& state) {
    os << "p: " << state.p_.transpose() << "\n"
       << "v: " << state.v_.transpose() << "\n"
       << "R:\n"
       << state.R_ << "\nbias acc: " << state.bias_acc_.transpose() << "\nbias gyro: " << state.bias_gyro_.transpose();
    return os;
  }

} // namespace ukf_manifold