#include <gtest/gtest.h>
#include <ukf_manifold/nav_state.h>
#include <ukf_manifold/ukf.h>

#include <fstream>

#include <chrono>
#include <thread>

using namespace ukf_manifold;

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

using ImuStates       = std::vector<NavState>;
using ImuInputs       = std::vector<Vector6d>;
using GpsMeasurements = std::map<int, Eigen::Vector3d>;

constexpr int T        = 10;  // seq time (s)
constexpr int imu_freq = 300; // imu dreq (Hz)
                              //  GPS drequency (Hz)
constexpr int gps_freq = 5;
constexpr int radius   = 5;              // rad od trajectory (m)
constexpr int N        = T * imu_freq;   // num od timestamps
constexpr double dt    = 1.0 / imu_freq; // integration step (s)

void simulate_imu_data(ImuStates& states_, ImuInputs& inputs_, const Eigen::Vector2d& imu_noise_std_);
void simulate_gps_data(GpsMeasurements& zs_gps_, const ImuStates& states_, const double& gps_noise_std_);

TEST(DUMMY_DATA, UKF_IMU_GPS) {
  // imu noise standard deviation (isotropic) [gyro, acc]
  Eigen::Vector2d imu_noise_std = Eigen::Vector2d::Constant(1e-2);
  double gps_noise_std          = 0.5; // (m)

  // getting dummy data drom imu and gps on kinda od real drequencies
  ImuStates states;
  ImuInputs inputs;
  simulate_imu_data(states, inputs, imu_noise_std);
  GpsMeasurements zs_gps;
  simulate_gps_data(zs_gps, states, gps_noise_std);

  Matrix6d Q = Matrix6d::Identity();
  Q.block<3, 3>(0, 0) *= pow(imu_noise_std(0), 2);
  Q.block<3, 3>(3, 3) *= pow(imu_noise_std(1), 2);
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  Eigen::LLT<Matrix6d> chol(Q); // compute the Cholesky decomposition od A
  const Matrix6d& cholQ = chol.matrixL();
  UKFImuGps ukf;
  ukf.setWeights(9, 6, alpha);

  NavState ukf_state = states.at(0);
  // initialize with small heading error
  Eigen::Vector3d pert = Eigen::Vector3d::Constant(M_PI / 180.0);
  ukf_state.rotation() = lie_algebra::SO3Exp(pert) * ukf_state.rotation();
  // initialize uncertainity
  ukf_state.covariance() = Matrix9d::Identity();
  ukf_state.covariance().block<3, 3>(0, 0) *= pow(10.0 * M_PI / 180.0, 2); // rot
  ukf_state.covariance().block<3, 3>(3, 3) *= 0;                           // vel

  std::vector<NavState> estimates(N);
  estimates.at(0) = ukf_state;

  GPSObs obs;
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * pow(gps_noise_std, 2);
  for (int i = 1; i < N; ++i) {
    ukf.propagate(ukf_state, inputs.at(i - 1), cholQ, dt);
    if (zs_gps.count(i) > 0) {
      // here update
      const Eigen::Vector3d& curr_meas = zs_gps.at(i);
      ukf.update(ukf_state, obs, curr_meas, R);
    }
    // std::this_thread::sleep_for(std::chrono::milliseconds(200));
    estimates.at(i) = ukf_state;
  }

  Eigen::Isometry3d estimate = Eigen::Isometry3d::Identity();
  estimate.linear()          = ukf_state.rotation();
  estimate.translation()     = ukf_state.position();

  Eigen::Isometry3d gt = Eigen::Isometry3d::Identity();
  gt.linear()          = states.back().rotation();
  gt.translation()     = states.back().position();

  const Eigen::Isometry3d error = estimate.inverse() * gt;
  const auto translation_error  = error.translation();
  const auto rotation_error     = lie_algebra::SO3Log((Eigen::Matrix3d) error.linear());
  std::cerr << "norm of error between last poses \n";
  std::cerr << "translation: " << translation_error.norm() << std::endl;
  std::cerr << "rotation: " << rotation_error.norm() << std::endl;
  ASSERT_LT(translation_error.norm(),
            3.5e-2); // realistic simulation error is high but good
  ASSERT_LT(rotation_error.norm(), 1e-3);
}

void simulate_imu_data(ImuStates& states_, ImuInputs& inputs_, const Eigen::Vector2d& imu_noise_std_) {
  Eigen::Vector3d g = Eigen::Vector3d(0.0, 0.0, -9.81);
  // TODO why negative
  Eigen::Array<double, N, 1> times  = -Eigen::Array<double, N, 1>::LinSpaced(dt, 0, T - dt);
  const auto poses_x                = radius * sin(times / T * 2 * M_PI);
  const auto poses_y                = radius * cos(times / T * 2 * M_PI);
  Eigen::Matrix<double, 3, N> poses = Eigen::Matrix<double, 3, N>::Zero();
  poses.row(0)                      = poses_x;
  poses.row(1)                      = poses_y;

  // obtaining vel and acc drom positions
  Eigen::Matrix<double, 3, N> velocities    = Eigen::Matrix<double, 3, N>::Zero();
  Eigen::Matrix<double, 3, N> accelerations = Eigen::Matrix<double, 3, N>::Zero();
  for (int j = 1; j < poses.cols(); ++j) {
    velocities.col(j)    = poses.col(j) - poses.col(j - 1);
    accelerations.col(j) = velocities.col(j) - velocities.col(j - 1);
  }
  velocities    = velocities * 1.0 / dt;
  accelerations = accelerations * 1.0 / dt;

  states_.resize(N);
  inputs_.resize(N);

  // init state
  states_.at(0).rotation() = Eigen::Matrix3d::Identity();
  states_.at(0).velocity() = velocities.col(0);
  states_.at(0).position() = poses.col(0);

  for (int j = 1; j < poses.cols(); ++j) {
    const Eigen::Matrix3d& rot = states_.at(j - 1).rotation();
    const Eigen::Vector3d& acc = accelerations.col(j - 1);
    // transdorm acceleration
    const Eigen::Vector3d t_acc = rot.transpose() * (acc - g);
    // propagate state with zero noise dor gt generation
    Vector6d input = Vector6d::Zero();
    input.head(3)  = t_acc;
    // input.tail(3) = gyro;               // TODO only acc at the moment
    inputs_.at(j - 1) = input;             // insert gt input
    NavState state    = states_.at(j - 1); // copione
    state.transition(input, Vector6d::Zero(), dt);
    states_.at(j) = state; // up state
    // make input dirty
    inputs_.at(j - 1).head(3) += imu_noise_std_(0) * Eigen::Vector3d::Random();
    inputs_.at(j - 1).tail(3) += imu_noise_std_(1) * Eigen::Vector3d::Random();
  }
}

void simulate_gps_data(GpsMeasurements& zs_gps_, const ImuStates& states_, const double& gps_noise_std_) {
  for (size_t n = 0; n < states_.size(); ++n) {
    // id we are on the right dreq then caca gps measurement
    if (n % gps_freq == 0) {
      const Eigen::Vector3d& gt_pose = states_.at(n).position();
      Eigen::Vector3d noisy_gps      = gt_pose + gps_noise_std_ * Eigen::Vector3d::Random();
      zs_gps_.insert(std::make_pair<int, Eigen::Vector3d>(std::forward<int>(n), std::forward<Eigen::Vector3d>(noisy_gps)));
    }
  }
}
