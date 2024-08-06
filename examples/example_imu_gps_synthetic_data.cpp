#include <ukf_manifold/base_state.h>
#include <ukf_manifold/lie_algebra.h>
#include <ukf_manifold/nav_state.h>
#include <ukf_manifold/ukf.h>

#include <fstream>
#include <istream>

#include <chrono>
#include <thread>

#include <map>
#include <vector>

using namespace ukf_manifold;

using ImuStates        = std::vector<NavState>;
using ImuInputs        = std::vector<Vector6d>;
using GpsMeasurements  = std::map<int, Eigen::Vector3d>;
using IntegrationSteps = std::vector<double>;

constexpr int T        = 10;  // seq time (s)
constexpr int imu_freq = 100; // imu dreq (Hz)
                              //  GPS drequency (Hz)
constexpr int gps_freq = 5;
constexpr int radius   = 5;              // rad od trajectory (m)
constexpr int N        = T * imu_freq;   // num od timestamps
constexpr double dt    = 1.0 / imu_freq; // integration step (s)

const std::string data_folder(UKF_DATA_FOLDER);

void simulate_gps_data(GpsMeasurements& zs_gps, const ImuStates& states, const double& gps_noise_std);

void read_input(ImuInputs& inputs_, IntegrationSteps& dts_, const std::string filename) {
  std::ifstream is(filename.c_str());
  if (!is.good())
    return;
  while (is) {
    double dt, ax, ay, az, wx, wy, wz;
    is >> dt >> ax >> ay >> az >> wx >> wy >> wz;
    Vector6d input;
    input << ax, ay, az, wx, wy, wz;
    dts_.push_back(dt);
    inputs_.push_back(input);
  }
  inputs_.shrink_to_fit();
  dts_.shrink_to_fit();
}

void read_states(ImuStates& states, const std::string filename) {
  std::ifstream is(filename.c_str());
  if (!is.good())
    return;
  while (is) {
    double dt, rx, ry, rz, vx, vy, vz, px, py, pz;
    is >> dt >> rx >> ry >> rz >> vx >> vy >> vz >> px >> py >> pz;
    const Eigen::Vector3d rot_vec = Eigen::Vector3d(rx, ry, rz);
    const Eigen::Matrix3d R       = lie_algebra::SO3Exp(rot_vec);
    const Eigen::Vector3d v       = Eigen::Vector3d(vx, vy, vz);
    const Eigen::Vector3d p       = Eigen::Vector3d(px, py, pz);
    NavState state;
    state.rotation() = R;
    state.velocity() = v;
    state.position() = p;
    states.push_back(state);
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: this-executable path-to-output" << std::endl;
    return 1;
  }

  // imu noise standard deviation (isotropic) [gyro, acc]
  Eigen::Vector2d imu_noise_std = Eigen::Vector2d::Constant(1e-2);
  double gps_noise_std          = 0.4; // (m)

  // getting dummy data from imu and gps on kinda of real frequencies
  ImuStates states;
  ImuInputs inputs;
  IntegrationSteps dts;

  std::string inputs_data_path = data_folder + "/imu_data/imu_inputs.txt";
  std::string inputs_gt_path   = data_folder + "/imu_data/imu_states.txt";

  read_input(inputs, dts, inputs_data_path);
  read_states(states, inputs_gt_path);

  if (inputs.size() != states.size())
    throw std::runtime_error("input and gt file size wrong!");

  GpsMeasurements zs_gps;
  simulate_gps_data(zs_gps, states, gps_noise_std);

  Matrix6d Q = Matrix6d::Identity();
  Q.block<3, 3>(0, 0) *= pow(imu_noise_std(0), 2);
  Q.block<3, 3>(3, 3) *= pow(imu_noise_std(1), 2);
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  Eigen::LLT<Matrix6d> chol(Q); // compute the Cholesky decomposition of A
  const Matrix6d& cholQ = chol.matrixL();
  UKFImuGps ukf;
  ukf.setWeights(9, 6, alpha);

  NavState ukf_state = states.at(0);
  // initialize with small heading error
  Eigen::AngleAxisd aa = Eigen::AngleAxisd(3.0 * M_PI / 180.0, Eigen::Vector3d::UnitZ());
  ukf_state.rotation() = aa.toRotationMatrix() * ukf_state.rotation();
  ukf_state.position() += Eigen::Vector3d::Constant(0.3);

  // initialize uncertainity
  ukf_state.covariance() = Matrix9d::Identity();
  ukf_state.covariance().block<3, 3>(0, 0) *= pow(5.0 * M_PI / 180.0, 2);
  ukf_state.covariance().block<3, 3>(3, 3) *= 0;

  // start from init gt state
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

void simulate_gps_data(GpsMeasurements& zs_gps, const ImuStates& states, const double& gps_noise_std) {
  for (size_t n = 0; n < states.size(); ++n) {
    // id we are on the right freq then caca gps measurement
    if (n % gps_freq == 0) {
      const Eigen::Vector3d& gt_pose = states.at(n).position();
      Eigen::Vector3d noisy_gps      = gt_pose + gps_noise_std * Eigen::Vector3d::Random();
      zs_gps.insert(std::make_pair<int, Eigen::Vector3d>(std::forward<int>(n), std::forward<Eigen::Vector3d>(noisy_gps)));
    }
  }
}
