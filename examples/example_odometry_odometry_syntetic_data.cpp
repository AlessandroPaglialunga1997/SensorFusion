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

using OdomStates       = std::vector<OdomState>;
using OdomInputs       = std::vector<Vector6d>;
using OdomMeasurements = std::map<int, Vector6d>;

constexpr int T        = 20;             // seq time (s)
constexpr int odo_freq = 100;            // imu dreq (Hz)
constexpr int N        = T * odo_freq;   // num od timestamps
constexpr double dt    = 1.0 / odo_freq; // integration step (s)

const std::string data_folder(UKF_DATA_FOLDER);

void simulate_odom_correction_data(OdomMeasurements& zs_odom,
                                   const OdomStates& states,
                                   const Vector6d& odom_noise_std,
                                   const Eigen::Isometry3d& offset);

void read_input(OdomInputs& inputs, const std::string filename) {
  std::ifstream is(filename.c_str());
  if (!is.good())
    return;
  while (is) {
    double px, py, pz, ax, ay, az;
    is >> px >> py >> pz >> ax >> ay >> az;
    Vector6d input;
    input << px, py, pz, ax, ay, az;
    inputs.push_back(input);
  }
  inputs.shrink_to_fit();
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
  states.shrink_to_fit();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: this-executable path-to-output" << std::endl;
    return 1;
  }

  Vector6d odom_noise_std;
  odom_noise_std.head(3) = Eigen::Vector3d::Constant(0.01 * dt);               // positional speed noise (m/s)
  odom_noise_std.tail(3) = Eigen::Vector3d::Constant(dt * 1.0 / 180.0 * M_PI); // orientational speed noise (rad/s)

  // getting dummy data from imu and gps on kinda of real frequencies
  OdomStates states;
  OdomInputs inputs;

  std::string inputs_data_path = data_folder + "/odom_data/odom_inputs.txt";
  std::string inputs_gt_path   = data_folder + "/odom_data/odom_states.txt";

  read_input(inputs, inputs_data_path);
  read_states(states, inputs_gt_path);

  if (inputs.size() != states.size())
    throw std::runtime_error("input and gt file size wrong!");

  OdomMeasurements zs_odom;
  Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
  Eigen::Quaterniond q(0.3, 0.3, 0.3, 0.3);
  q.normalize();
  offset.linear() = q.toRotationMatrix();
  offset.translation() << 0.5, -0.5, 0.3;
  simulate_odom_correction_data(zs_odom, states, odom_noise_std, offset);

  Matrix6d Q = Matrix6d::Identity();
  Q.block<3, 3>(0, 0) *= pow(odom_noise_std(0), 2);
  Q.block<3, 3>(3, 3) *= pow(odom_noise_std(3), 2);
  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  Eigen::LLT<Matrix6d> chol(Q); // compute the Cholesky decomposition of A
  const Matrix6d& cholQ = chol.matrixL();
  UKFOdomOdom ukf;
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

  OdomObs obs;
  obs.setOffset(offset);
  const Matrix6d R = Q; // assuming noise model is the same, this should be
                        // obtained directly from covariance output from T265
  for (int i = 1; i < N; ++i) {
    ukf.propagate(ukf_state, inputs.at(i - 1), cholQ);
    if (zs_odom.count(i) > 0) {
      // here update
      const Vector6d& curr_meas = zs_odom.at(i);
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

void simulate_odom_correction_data(OdomMeasurements& zs_odom,
                                   const OdomStates& states,
                                   const Vector6d& odom_noise_std,
                                   const Eigen::Isometry3d& offset) {
  for (size_t n = 0; n < states.size(); ++n) {
    Eigen::Isometry3d full_pose              = Eigen::Isometry3d::Identity();
    full_pose.linear()                       = states.at(n).rotation();
    full_pose.translation()                  = states.at(n).position();
    const Eigen::Isometry3d transformed_pose = offset * full_pose;
    const Eigen::Vector3d& gt_pose           = transformed_pose.translation();
    const Eigen::Vector3d& gt_orientation    = lie_algebra::SO3Log(transformed_pose.linear());
    Vector6d noisy_odom;
    noisy_odom.head(3) = gt_pose + odom_noise_std.head(3).cwiseProduct(Eigen::Vector3d::Random());
    noisy_odom.tail(3) = gt_orientation + odom_noise_std.tail(3).cwiseProduct(Eigen::Vector3d::Random());
    // useless a map here but who cares
    zs_odom.insert(std::make_pair<int, Vector6d>(std::forward<int>(n), std::forward<Vector6d>(noisy_odom)));
  }
}
