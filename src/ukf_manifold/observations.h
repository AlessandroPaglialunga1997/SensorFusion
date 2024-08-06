#pragma once
#include "nav_state.h"
#include <Eigen/Dense>

namespace ukf_manifold {
  // gps observation model
  struct GPSObs {
    template <typename StateType>
    Eigen::Vector3d const observation(const StateType& state) const {
      return state.position();
    }

    inline const size_t obsDim() const {
      return 3;
    }
  };

  // odom observation model with offset
  struct OdomObs {
    inline void setOffset(const Eigen::Isometry3d& offset) {
      offset_ = offset;
    }
    template <typename StateType>
    Vector6d const observation(const StateType& state) const {
      const Eigen::Matrix3d orientation = offset_.linear() * state.rotation();
      const Eigen::Vector3d pose        = offset_.linear() * state.position() + offset_.translation();

      Vector6d obs;
      obs.head(3) = pose;
      obs.tail(3) = lie_algebra::SO3Log(orientation);
      return obs;
    }

    inline const size_t obsDim() const {
      return 6;
    }

    Eigen::Isometry3d offset_ = Eigen::Isometry3d::Identity();
  };

} // namespace ukf_manifold