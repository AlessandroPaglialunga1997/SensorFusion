#include <gtest/gtest.h>
#include <ukf_manifold/base_state.h>
#include <ukf_manifold/lie_algebra.h>
#include <ukf_manifold/nav_state.h>
#include <ukf_manifold/ukf.h>

#include <fstream>
#include <istream>

#include <chrono>
#include <thread>

using namespace ukf_manifold;

struct KittiCalibration {
  double body_ptx;
  double body_pty;
  double body_ptz;
  double body_prx;
  double body_pry;
  double body_prz;
  double accelerometer_sigma;
  double gyroscope_sigma;
  double integration_sigma;
  double accelerometer_bias_sigma;
  double gyroscope_bias_sigma;
  double average_delta_t;
};

struct ImuMeasurement {
  double time;
  Eigen::Vector3d accelerometer;
  Eigen::Vector3d gyroscope; // omega
};

struct GpsMeasurement {
  double time;
  Eigen::Vector3d position; // x,y,z
};

using GpsMeasurements = std::map<size_t, GpsMeasurement>;

using Vector4d    = Eigen::Matrix<double, 4, 1>;
using Matrix12d   = Eigen::Matrix<double, 12, 12>;
using Matrix3_6d  = Eigen::Matrix<double, 3, 6>;
using Matrix3_15d = Eigen::Matrix<double, 3, 15>;

const std::string data_folder(UKF_DATA_FOLDER);

void loadKittiData(KittiCalibration& kitti_calibration,
                   std::vector<ImuMeasurement>& imu_measurements,
                   GpsMeasurements& gps_measurements);

void ukfJacobianUpdate(Eigen::Vector3d& error,
                       Eigen::MatrixXd& J,
                       const NavStateExtended& state,
                       const GPSObs& z,
                       const Eigen::Vector3d& meas,
                       const Weights& weights_);
void extendedKalmanFilterUpdate(NavStateExtended& state,
                                const Eigen::Vector3d& error,
                                const Eigen::MatrixXd& J,
                                const Eigen::Matrix3d& R);
int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: this-executable path-to-output" << std::endl;
    return 1;
  }

  std::vector<ImuMeasurement> imu_measurements;
  GpsMeasurements gps_measurements;
  KittiCalibration metadata;
  loadKittiData(metadata, imu_measurements, gps_measurements);

  Vector4d imu_noise_std;
  // imu_noise_std << metadata.gyroscope_sigma, metadata.accelerometer_sigma,
  //     metadata.gyroscope_bias_sigma, metadata.accelerometer_bias_sigma;
  imu_noise_std << 0.01, 0.05, 0.000001, 0.0001;
  const double gps_noise_std = 0.2; // (m)

  // propagation noise covariance
  Matrix12d Q = Matrix12d::Identity();
  Q.block<3, 3>(0, 0) *= pow(imu_noise_std(0), 2);
  Q.block<3, 3>(3, 3) *= pow(imu_noise_std(1), 2);
  Q.block<3, 3>(6, 6) *= pow(imu_noise_std(2), 2);
  Q.block<3, 3>(9, 9) *= pow(imu_noise_std(3), 2);

  const Eigen::Vector3d alpha = Eigen::Vector3d(1e-3, 1e-3, 1e-3);

  UKFImuExtendedGps ukf;
  NavStateExtended ukf_state;

  // reduced weights during prediction statedim 15
  ukf.setWeights(ukf_state.stateDim(), 3, alpha);
  // Weights red_weights = ukf.weights();
  // weights during update for ekf
  // ukf.setWeights(3, 3, alpha);
  // Weights weights = ukf.weights();

  // ukf.setWeights(15, 3, alpha);

  std::cerr << "extended nav state -> dim state: " << ukf_state.stateDim() << "| dim input: " << ukf_state.inputDim()
            << std::endl;

  // initialize uncertainity
  ukf_state.covariance() = Matrix15d::Identity();      // pose and velocity
  ukf_state.covariance().block<3, 3>(0, 0) *= 0.01;    // rot
  ukf_state.covariance().block<3, 3>(9, 9) *= 0.001;   // bias gyro
  ukf_state.covariance().block<3, 3>(12, 12) *= 0.001; // bias acc

  // initialize state with first gps measurement
  ukf_state.position() = gps_measurements.at(0).position;

  // getting gps h shit and covariance
  GPSObs obs;
  const Eigen::Matrix3d R = Eigen::Matrix3d::Identity() * pow(gps_noise_std, 2);

  // loop for sensor syncronization
  std::vector<NavStateExtended> estimates;

  for (size_t i = 1; i < imu_measurements.size(); ++i) {
    // predict
    Vector6d input = Vector6d::Zero(); // [gyro, acc]
    input.head(3)  = imu_measurements.at(i - 1).gyroscope;
    input.tail(3)  = imu_measurements.at(i - 1).accelerometer;
    // calculating dt
    const double prev_time = imu_measurements.at(i - 1).time;
    const double curr_time = imu_measurements.at(i).time;
    const double dt        = curr_time - prev_time;

    const Matrix6d Q_inputs = Q.block<6, 6>(0, 0);
    // ukf.setWeights(red_weights); // set reduced weights for propagation
    ukf.propagate(ukf_state, input, Q_inputs, dt);
    // add bias covariance
    const Matrix6d Q_bias = Q.block<6, 6>(6, 6);
    ukf_state.covariance().block<6, 6>(9, 9) += Q_bias * dt * dt;

    if (gps_measurements.count(i) > 0) {
      // getting measurement from gps
      const Eigen::Vector3d& curr_meas = gps_measurements.at(i).position;
      // ekf update TODO not working
      // MatrixXd J;
      // Eigen::Vector3d error =Eigen::Vector3d::Zero();
      // ukfJacobianUpdate(error, J, ukf_state, obs, curr_meas, weights);
      // extendedKalmanFilterUpdate(ukf_state, error, J, R);
      // std::this_thread::sleep_for(std::chrono::seconds(1));

      // ukf update
      ukf.update(ukf_state, obs, curr_meas, R);
    }
    estimates.push_back(ukf_state);
  }

  {
    const std::string filepath = std::string(argv[1]) + "/kitti_estimate.txt";
    std::cout << "writing estimate to: " << filepath << std::endl;
    std::ofstream file(filepath);
    if (file.is_open()) {
      for (const auto& s : estimates) {
        file << s.position().x() << " " << s.position().y() << " " << s.position().z() << "\n";
      }
      file.close();
    }
  }
}

void ukfJacobianUpdate(Eigen::Vector3d& error,
                       Eigen::MatrixXd& J,
                       const NavStateExtended& state,
                       const GPSObs& z,
                       const Eigen::Vector3d& meas,
                       const Weights& ws) {
  // get some quantities at runtime
  const int dim_state            = state.stateDim(); // dim state
  const int dim_obs              = z.obsDim();
  const int dim_restricted_state = z.obsDim();

  //  resize dynamic shit
  J = Eigen::MatrixXd::Zero(dim_obs, dim_state);

  Eigen::Matrix3d tol_mat               = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d pose_covariance = state.covariance().block<3, 3>(6, 6) + 1e-9 * tol_mat;

  // compute the Cholesky decomposition od A
  Eigen::LLT<Eigen::Matrix3d> chol(pose_covariance);
  const Eigen::Matrix3d& L = chol.matrixL();
  // set sigma points
  const Eigen::Matrix3d xis = ws.w_u.sqrt_lambda_ * L;

  Matrix3_6d zs         = Matrix3_6d::Zero();
  Eigen::Vector3d z_hat = z.observation(state);

  for (int j = 0; j < dim_restricted_state; ++j) {
    // apply perturbation to sigma points
    NavStateExtended chi_j_plus, chi_j_minus;
    chi_j_plus = chi_j_minus = state;
    // hand crafted boxplus for pose only
    chi_j_plus.position()  = state.position() + xis.col(j);
    chi_j_minus.position() = state.position() - xis.col(j);
    std::cerr << chi_j_plus << std::endl << std::endl;
    std::cerr << chi_j_minus << std::endl << std::endl << std::endl;
    // propagate through observation
    zs.col(j)                        = z.observation(chi_j_plus);
    zs.col(dim_restricted_state + j) = z.observation(chi_j_minus);
  }

  // dim (dim_obs, 1)
  Eigen::Vector3d z_bar = ws.w_u.wm0_ * z_hat + ws.w_u.wj_ * zs.rowwise().sum();
  // prune mean
  zs.colwise() -= z_bar;

  Matrix3_6d xis_stacked                      = Matrix3_6d::Zero();
  xis_stacked.leftCols(dim_restricted_state)  = xis;
  xis_stacked.rightCols(dim_restricted_state) = -xis;
  Eigen::Matrix3d Z                           = ws.w_u.wm0_ * zs * xis_stacked.transpose(); // dim (3, 6)(6, 3)
  Eigen::Matrix3d Jpose                       = Z * pose_covariance.inverse();
  // update error and jacobian
  error = meas - z_bar;

  J.block<3, 3>(0, 6) = Jpose;
}

void extendedKalmanFilterUpdate(NavStateExtended& state,
                                const Eigen::Vector3d& error,
                                const Eigen::MatrixXd& J,
                                const Eigen::Matrix3d& R) {
  const int dim_state     = state.stateDim(); // dim state
  const Eigen::MatrixXd S = J * state.covariance() * J.transpose() + R;
  const Eigen::MatrixXd K = state.covariance() * J.transpose() * S.inverse();
  // update state
  const Eigen::VectorXd werror = K * error;
  std::cerr << werror.transpose() << std::endl;
  // std::cerr << "state\n" << state << std::endl;
  std::unique_ptr<NavStateExtended> new_state(std::move(state.boxplus(werror)));
  // std::cerr << "new state\n" << *new_state << std::endl;
  // update covariance
  const Eigen::MatrixXd IKJ = Eigen::MatrixXd::Identity(dim_state, dim_state) - K * J;
  new_state->covariance()   = IKJ * state.covariance() * IKJ.transpose() + K * R * K.transpose();
  state                     = *new_state;
}

void loadKittiData(KittiCalibration& kitti_calibration,
                   std::vector<ImuMeasurement>& imu_measurements,
                   GpsMeasurements& gps_measurements) {
  std::string line;

  // init stuff for time calculation
  bool init        = true;
  double init_time = 0.0;

  // Read IMU metadata and compute relative sensor pose transforms
  // BodyPtx BodyPty BodyPtz BodyPrx BodyPry BodyPrz AccelerometerSigma
  // GyroscopeSigma IntegrationSigma AccelerometerBiasSigma GyroscopeBiasSigma
  // AverageDelta
  std::ifstream imu_metadata(data_folder + "/kitti_data/KittiImuBiasedMetadata.txt");
  printf("-- Reading sensor metadata\n");

  getline(imu_metadata, line, '\n'); // ignore the first line
  // Load Kitti calibration
  getline(imu_metadata, line, '\n');
  sscanf(line.c_str(),
         "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
         &kitti_calibration.body_ptx,
         &kitti_calibration.body_pty,
         &kitti_calibration.body_ptz,
         &kitti_calibration.body_prx,
         &kitti_calibration.body_pry,
         &kitti_calibration.body_prz,
         &kitti_calibration.accelerometer_sigma,
         &kitti_calibration.gyroscope_sigma,
         &kitti_calibration.integration_sigma,
         &kitti_calibration.accelerometer_bias_sigma,
         &kitti_calibration.gyroscope_bias_sigma,
         &kitti_calibration.average_delta_t);
  printf("IMU metadata: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
         kitti_calibration.body_ptx,
         kitti_calibration.body_pty,
         kitti_calibration.body_ptz,
         kitti_calibration.body_prx,
         kitti_calibration.body_pry,
         kitti_calibration.body_prz,
         kitti_calibration.accelerometer_sigma,
         kitti_calibration.gyroscope_sigma,
         kitti_calibration.integration_sigma,
         kitti_calibration.accelerometer_bias_sigma,
         kitti_calibration.gyroscope_bias_sigma,
         kitti_calibration.average_delta_t);

  // Read IMU data
  // Time dt accelX accelY accelZ omegaX omegaY omegaZ
  printf("-- Reading IMU measurements from file");
  {
    std::ifstream imu_data(data_folder + "/kitti_data/KittiImuBiased.txt");
    getline(imu_data, line, '\n'); // ignore the first line

    double time = 0, dt = 0, acc_x = 0, acc_y = 0, acc_z = 0, gyro_x = 0, gyro_y = 0, gyro_z = 0;
    while (!imu_data.eof()) {
      getline(imu_data, line, '\n');
      sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf", &time, &dt, &acc_x, &acc_y, &acc_z, &gyro_x, &gyro_y, &gyro_z);
      if (init) {
        init_time = time;
        init      = false;
      }
      ImuMeasurement measurement;
      const double correct_time = time - init_time;
      measurement.time          = correct_time;
      measurement.accelerometer = Eigen::Vector3d(acc_x, acc_y, acc_z);
      measurement.gyroscope     = Eigen::Vector3d(gyro_x, gyro_y, gyro_z);
      imu_measurements.push_back(measurement);
    }
  }
  printf(" | number of measurements: %li\n", imu_measurements.size());

  // Read GPS data
  // Time,X,Y,Z
  std::vector<GpsMeasurement> gps_measvec;
  printf("-- Reading GPS measurements from file");
  {
    std::ifstream gps_data(data_folder + "/kitti_data/KittiGps.txt");
    getline(gps_data, line, '\n'); // ignore the first line

    double time = 0, gps_x = 0, gps_y = 0, gps_z = 0;
    while (!gps_data.eof()) {
      getline(gps_data, line, '\n');
      sscanf(line.c_str(), "%lf,%lf,%lf,%lf", &time, &gps_x, &gps_y, &gps_z);

      GpsMeasurement measurement;
      const double correct_time = time - init_time;
      measurement.time          = correct_time;
      measurement.position      = Eigen::Vector3d(gps_x, gps_y, gps_z);
      gps_measvec.push_back(measurement);
    }
  }

  size_t count_gps = 0;
  for (size_t i = 0; i < imu_measurements.size(); ++i) {
    if (gps_measvec.at(count_gps).time <= imu_measurements.at(i).time) {
      gps_measurements.insert(
        std::make_pair<size_t, GpsMeasurement>(std::forward<size_t>(i), std::forward<GpsMeasurement>(gps_measvec.at(count_gps))));
      count_gps++;
    }
    if (count_gps > gps_measvec.size() - 1) {
      break;
    }
  }

  printf(" | number of measurements: %li\n", gps_measvec.size());
}