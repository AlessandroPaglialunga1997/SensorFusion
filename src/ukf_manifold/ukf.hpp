namespace ukf_manifold {
  // d dim state
  // q dim input
  template <class StateType, class ObservationType>
  void UKF_<StateType, ObservationType>::propagate(StateType& state,
                                                   const Eigen::VectorXd& input,
                                                   const Eigen::MatrixXd& CholQ,
                                                   const double& dt) {
    const int dim_state = state.stateDim(); // dim state
    const int dim_input = state.inputDim(); // dim state

    Eigen::MatrixXd tol_mat(dim_state, dim_state);
    tol_mat.setIdentity();
    state.covariance().noalias() += tol_ * tol_mat;

    // apply transition current state omitting noise
    StateType prev_state = state;
    Eigen::VectorXd zero_noise(dim_input);
    zero_noise.setZero();
    state.transition(input, zero_noise, dt);

    // keep cov dimension
    Eigen::LLT<Eigen::MatrixXd> chol(state.covariance()); // compute the Cholesky decomposition od A
    const Eigen::MatrixXd& L  = chol.matrixL();
    const Eigen::MatrixXd xis = ws_.w_d.sqrt_lambda_ * L;

    Eigen::MatrixXd xis_new(dim_state, 2 * dim_state);
    xis_new.setZero();

    for (int j = 0; j < dim_state; ++j) {
      // apply perturbation to sigma points
      std::unique_ptr<StateType> s_j_p(std::move(prev_state.boxplus(xis.col(j))));
      std::unique_ptr<StateType> s_j_m(std::move(prev_state.boxplus(-xis.col(j))));
      // propagate with input
      s_j_p->transition(input, zero_noise, dt);
      s_j_m->transition(input, zero_noise, dt);
      // go back in R^d to calculate covariance
      xis_new.col(j)             = *std::move(state.boxminus(*s_j_p));
      xis_new.col(dim_state + j) = *std::move(state.boxminus(*s_j_m));
    }

    // compute covariance, xi_mean has size (dim_state, 1)
    const Eigen::VectorXd xi_mean = ws_.w_d.wj_ * xis_new.rowwise().sum();

    xis_new.colwise() -= xi_mean;
    state.covariance() = ws_.w_d.wj_ * (xis_new * xis_new.transpose()) + ws_.w_d.wc0_ * (xi_mean * xi_mean.transpose());

    // compute covariance wrt to noise (dim_state, 2*dim_input)
    Eigen::MatrixXd xis_new_noise(dim_state, 2 * dim_input);
    xis_new_noise.setZero();
    for (int i = 0; i < dim_input; ++i) {
      const Eigen::VectorXd w_p = ws_.w_q.sqrt_lambda_ * CholQ.col(i);
      const Eigen::VectorXd w_m = -w_p;
      StateType s_j_p_new       = prev_state;
      StateType s_j_m_new       = prev_state;
      s_j_p_new.transition(input, w_p, dt);
      s_j_m_new.transition(input, w_m, dt);
      xis_new_noise.col(i)             = *std::move(state.boxminus(s_j_p_new));
      xis_new_noise.col(i + dim_input) = *std::move(state.boxminus(s_j_m_new));
    }

    // compute covariance wrt to noise
    const Eigen::VectorXd xi_mean_noise = ws_.w_q.wj_ * xis_new_noise.rowwise().sum();
    xis_new_noise.colwise() -= xi_mean_noise;
    const Eigen::MatrixXd Q =
      ws_.w_q.wj_ * (xis_new_noise * xis_new_noise.transpose()) + ws_.w_q.wc0_ * (xi_mean_noise * xi_mean_noise.transpose());
    // add contribution of noise covariance to overall covariance
    state.covariance().noalias() += Q;
  }

  template <class StateType, class ObservationType>
  Eigen::VectorXd UKF_<StateType, ObservationType>::update(StateType& state,
                                                           ObservationType& z,
                                                           const Eigen::VectorXd& meas,
                                                           const Eigen::MatrixXd& R) {
    const int dim_state = state.stateDim(); // dim state
    const int dim_obs   = z.obsDim();

    Eigen::MatrixXd tol_mat(dim_state, dim_state);
    tol_mat.setIdentity();
    state.covariance().noalias() += tol_ * tol_mat;

    //  TODO move R covariance measurement out this shit

    // compute the Cholesky decomposition od A
    Eigen::LLT<Eigen::MatrixXd> chol(state.covariance());
    const Eigen::MatrixXd& L = chol.matrixL();
    // set sigma points
    const Eigen::MatrixXd xis = ws_.w_u.sqrt_lambda_ * L;

    Eigen::MatrixXd zs(dim_obs, 2 * dim_state);
    zs.setZero();
    Eigen::VectorXd z_hat = z.observation(state);
    for (int j = 0; j < dim_state; ++j) {
      // apply perturbation to sigma points
      std::unique_ptr<StateType> chi_j_plus(std::move(state.boxplus(xis.col(j))));
      std::unique_ptr<StateType> chi_j_minus(std::move(state.boxplus(-xis.col(j))));
      // propagate through observation
      zs.col(j)             = z.observation(*chi_j_plus);
      zs.col(dim_state + j) = z.observation(*chi_j_minus);
    }

    // dim (dim_obs, 1)
    Eigen::VectorXd z_bar = ws_.w_u.wm0_ * z_hat + ws_.w_u.wj_ * zs.rowwise().sum();
    // prune mean before computing covariance TODO colwise
    zs.colwise() -= z_bar;
    z_hat.noalias() -= z_bar;

    // compute covariance and cross covariance matrices, dim (dim_obs, dim_obs)
    Eigen::MatrixXd P_zz = ws_.w_u.wc0_ * (z_hat * z_hat.transpose()) + ws_.w_u.wj_ * (zs * zs.transpose());
    P_zz.noalias() += R;

    Eigen::MatrixXd xis_stacked(dim_state, 2 * dim_state);
    xis_stacked.setZero();
    xis_stacked.leftCols(dim_state)  = xis;
    xis_stacked.rightCols(dim_state) = -xis;

    // dim (dim_state, dim_obs)
    Eigen::MatrixXd P_xz = ws_.w_u.wj_ * xis_stacked * zs.transpose();
    // solve system and get Kalman gain dim (dim_state, dim_obs)
    const Eigen::MatrixXd K = P_xz * P_zz.inverse();
    // update state
    const Eigen::VectorXd error   = meas - z_bar;
    const Eigen::VectorXd xi_plus = K * error;
    StateType old_state           = state;
    state                         = *old_state.boxplus(xi_plus);
    state.covariance().noalias() -= (K * P_zz * K.transpose());
    state.covariance() = 0.5 * (state.covariance() + state.covariance().transpose());
    return error;
  }

  template <class StateType, class ObservationType>
  Weight UKF_<StateType, ObservationType>::setWeight(const int n, const double alpha) {
    const double alpha2 = alpha * alpha;
    const double lambda = (alpha2 - 1.0) * n;
    Weight w;
    w.lambda_      = lambda;
    w.sqrt_lambda_ = sqrt(n + lambda);
    w.wj_          = 1.0 / (2.0 * (n + lambda));
    w.wm0_         = lambda / (lambda + n);
    w.wc0_         = lambda / (lambda + n) + 3.0 - alpha2;
    return w;
  }

} // namespace ukf_manifold