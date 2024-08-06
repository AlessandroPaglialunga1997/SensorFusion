#pragma once
#include "nav_state.h"
#include "observations.h"
#include "odometry_state.h"
#include <Eigen/Cholesky>
#include <memory>

namespace ukf_manifold {

  struct Weight {
    double lambda_      = 0.0;
    double sqrt_lambda_ = 0.0;
    double wj_          = 0.0;
    double wm0_         = 0.0;
    double wc0_         = 0.0;
  };

  struct Weights {
    Weight w_d;
    Weight w_q;
    Weight w_u;
  };

  template <class StateType, class ObservationType>
  class UKF_ {
  public:
    void propagate(StateType& state, const Eigen::VectorXd& input, const Eigen::MatrixXd& CholQ, const double& dt = 0.0);
    Eigen::VectorXd update(StateType& state, ObservationType& z, const Eigen::VectorXd& meas, const Eigen::MatrixXd& R);

    inline void setWeights(const Weights& ws) {
      ws_ = ws;
    }

    inline void setWeights(const int& d, const int& q, const Eigen::Vector3d& alpha) {
      // params dor state propagation wrt uncertainty
      ws_.w_d = setWeight(d, alpha(0));
      // params dor state propagation wrt noise
      ws_.w_q = setWeight(q, alpha(1));
      // params dor state update
      ws_.w_u = setWeight(d, alpha(2));
    }

    inline const Weights& weights() const {
      return ws_;
    }

  protected:
    Weight setWeight(const int n, const double alpha);
    Weights ws_;
    const double tol_ = 1e-9;
  };

  using UKFImuGps         = UKF_<NavState, GPSObs>;
  using UKFImuExtendedGps = UKF_<NavStateExtended, GPSObs>;
  using UKFOdomGps        = UKF_<OdomState, GPSObs>;
  using UKFOdomOdom       = UKF_<OdomState, OdomObs>;

} // namespace ukf_manifold

#include "ukf.hpp"