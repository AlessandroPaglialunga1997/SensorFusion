#include <gtest/gtest.h>
#include <ukf_manifold/base_state.h>
#include <ukf_manifold/lie_algebra.h>
#include <ukf_manifold/odometry_state.h>
#include <ukf_manifold/ukf.h>

#include <fstream>
#include <istream>

#include <chrono>
#include <thread>

using namespace ukf_manifold;

using OdomStates      = std::vector<OdomState>;
using OdomInputs      = std::vector<Vector6d>;
using GpsMeasurements = std::map<int, Eigen::Vector3d>;

constexpr int T        = 20;             // seq time (s)
constexpr int odo_freq = 100;            // imu dreq (Hz)
constexpr int gps_freq = 5;              //  GPS drequency (Hz)
constexpr int radius   = 5;              // rad od trajectory (m)
constexpr int N        = T * odo_freq;   // num od timestamps
constexpr double dt    = 1.0 / odo_freq; // integration step (s)

const std::string data_folder(UKF_DATA_FOLDER);

void simulate_gps_data(GpsMeasurements& zs_gps, const OdomStates& states, const double& gps_noise_std);

void read_input(OdomInputs& inputs_, const std::string filename) {
  std::ifstream is(filename.c_str());
  if (!is.good())
    return;
  while (is) {
    double px, py, pz, ax, ay, az;
    is >> px >> py >> pz >> ax >> ay >> az;
    Vector6d input;
    input << px, py, pz, ax, ay, az;
    inputs_.push_back(input);
  }
  inputs_.shrink_to_fit();
}

void read_states(OdomStates& states, const std::string filename) {
  std::ifstream is(filename.c_str());
  if (!is.good())
    return;
  while (is) {
    double px, py, pz, ax, ay, az;
    is >> px >> py >> pz >> ax >> ay >> az;
    const Eigen::Vector3d rot_vec = Eigen::Vector3d(ax, ay, az);
    const Eigen::Vector3d p       = Eigen::Vector3d(px, py, pz);
    const Eigen::Matrix3d R       = lie_algebra::SO3Exp(rot_vec);
    OdomState state;
    state.position() = p;
    state.rotation() = R;
    states.push_back(state);
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: this-executable path-to-output" << std::endl;
    return 1;
  }

  Vector6d odom_noise_std;
  odom_noise_std.head(3) = Eigen::Vector3d::Constant(0.01 * dt);               // positional speed noise (m)
  odom_noise_std.tail(3) = Eigen::Vector3d::Constant(dt * 1.0 / 180.0 * M_PI); // orientational speed noise (rad/s)
  double gps_noise_std   = 0.4;                                                // (m)

  // getting dummy data from imu and gps on kinda of real frequencies
  OdomStates states;
  OdomInputs inputs;

  std::string inputs_data_path = data_folder + "/odom_data/odom_inputs.txt";
  std::string inputs_gt_path   = data_folder + "/odom_data/odom_states.txt";

  read_input(inputs, inputs_data_path);
  read_states(states, inputs_gt_path);

  if (inputs.size() != states.size())
    throw std::runtime_error("input and gt file size wrong!");

  GpsMeasurements zs_gps;
  simulate_gps_data(zs_gps, states, gps_noise_std);

  Matrix6d Q = Matrix6d::Identity();
  Q.block<3, 3>(0, 0) *= pow(odom_noise_std(0), 2);
  Q.block<3, 3>(3, 3) *= pow(odom_noise_std(3), 2);
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  Eigen::LLT<Matrix6d> chol(Q); // compute the Cholesky decomposition of A
  const Matrix6d& cholQ = chol.matrixL();
  UKFOdomGps ukf;
  ukf.setWeights(6, 6, alpha);

  OdomState ukf_state = states.at(0);
  // initialize with small heading error
  Eigen::AngleAxisd aa = Eigen::AngleAxisd(3.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ());
  ukf_state.rotation() = aa.toRotationMatrix() * ukf_state.rotation();
  ukf_state.position() += Eigen::Vector3d::Constant(0.3);

  // initialize uncertainty
  ukf_state.covariance() = Matrix6d::Identity();
  ukf_state.covariance().block<3, 3>(3, 3) *= pow(5.0 * M_PI / 180.0, 2);

  // start from init gt state
  OdomStates estimates(N);
  estimates.at(0) = ukf_state;

  GPSObs obs;
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * pow(gps_noise_std, 2);
  for (int i = 1; i < N; ++i) {
    ukf.propagate(ukf_state, inputs.at(i - 1), cholQ);
    if (zs_gps.count(i) > 0) {
      // here update
      const Eigen::Vector3d& curr_meas = zs_gps.at(i);
      ukf.update(ukf_state, obs, curr_meas, R);
    }
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
  std::cerr << "error\n";
  std::cerr << "t: " << translation_error.norm() << std::endl;
  std::cerr << "rot: " << rotation_error.norm() << std::endl;

  {
    std::ofstream file(std::string(argv[1]) + "/traj_gt.txt");
    if (file.is_open()) {
      for (const auto& s : states) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }

  {
    std::ofstream file(std::string(argv[1]) + "/traj_est.txt");
    if (file.is_open()) {
      for (const auto& s : estimates) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }
}

void simulate_gps_data(GpsMeasurements& zs_gps, const OdomStates& states, const double& gps_noise_std) {
  for (size_t n = 0; n < states.size(); ++n) {
    // id we are on the right freq then caca gps measurement
    if (n % gps_freq == 0) {
      const Eigen::Vector3d& gt_pose = states.at(n).position();
      Eigen::Vector3d noisy_gps      = gt_pose + gps_noise_std * Eigen::Vector3d::Random();
      zs_gps.insert(std::make_pair<int, Eigen::Vector3d>(std::forward<int>(n), std::forward<Eigen::Vector3d>(noisy_gps)));
    }
  }
}
