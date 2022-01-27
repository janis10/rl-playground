import numpy as np
import scipy.linalg as LA


def lqr_gain(A, B, Q, R):
    """
    Arguments:
      State transition matrices (A, B)
      LQR Costs (Q, R)
    Outputs:
      K: optimal infinite-horizon LQR gain matrix given
    """
    # Solve Discrete Algebraic Ricatti Equation
    M = LA.solve_discrete_are(A, B, Q, R)
    # K = (B' M B + R)^(-1)*(B' M A)
    return np.dot(LA.inv(np.dot(np.dot(B.T, M), B) + R), (np.dot(np.dot(B.T, M), A)))


def cost_inf_K(A, B, Q, R, K):
    """
    Arguments:
      System (A, B)
      LQR Costs (Q, R)
      Control Gain K
    Outputs:
      cost: Infinite time horizon LQR cost of static gain K
    """
    cl_sys = A + B.dot(K)
    # Check stability of closed-loop system:
    epsilon = 1.0e-6
    stable = np.amax(np.abs(LA.eigvals(cl_sys))) < 1.0 - epsilon
    if stable:
        cost = np.trace(LA.solve_discrete_lyapunov(cl_sys.T, Q + np.dot(K.T, R.dot(K))))
    else:
        cost = float("inf")

    return cost


def cost_finite_model(A_true, B_true, Q, R, x0, T, A_dat, B_dat):
    """
    Arguments:
      True model (A_true, B_true)
      LQR Costs (Q, R)
      Initial State x0
      Time Horizon T
      Estimated model (A_dat, B_dat)
    Outputs:
      cost: finite time horizon LQR cost when control is computed using
      (A_dat,B_dat) but executed on system (A_true, B_true)
    """
    d = B_true.shape[0]

    # Ricatti recursion
    M = np.zeros((d, d, T))
    M[:, :, -1] = Q
    for k in range(T - 2, -1, -1):
        AMA = np.dot(A_dat.T, M[:, :, k + 1].dot(A_dat))
        AMB = np.dot(A_dat.T, M[:, :, k + 1].dot(B_dat))
        BMB = np.dot(B_dat.T, M[:, :, k + 1].dot(B_dat))
        M[:, :, k] = Q + AMA - np.dot(AMB, LA.inv(R + BMB).dot(AMB.T))

    # Controls and costs using the Ricatti iterates
    cost = 0
    x = x0
    for k in range(T):
        AMB = np.dot(A_dat.T, M[:, :, k].dot(B_dat))
        BMB = np.dot(B_dat.T, M[:, :, k].dot(B_dat))
        u = -np.dot(LA.inv(R + BMB), np.dot(AMB.T, x))
        x = A_true.dot(x) + B_true.dot(u)
        cost += np.dot(x.T, Q.dot(x)) + np.dot(u.T, R.dot(u))

    return cost.flatten()[0]


def cost_finite_K(A_true, B_true, Q, R, x0, T, K):
    """
    Arguments:
      True model (A_true, B_true)
      LQR Costs (Q, R)
      Initial State x0
      Time Horizon T
      Static Control Gain K
    Outputs:
      cost: finite time horizon LQR cost when control is static gain K on
      system (A_true, B_true)
    """

    cost = 0
    x = x0
    for k in range(T):
        u = np.dot(K, x)
        x = A_true.dot(x) + B_true.dot(u)
        cost += np.dot(x.T, Q.dot(x)) + np.dot(u.T, R.dot(u))

    return cost.flatten()


def lsqr_estimator(A, B, x0, w_mgn, N, T):
    """
    Arguments:
      True model (A, B) (for simulation)
      Initial State x0
      Magnitude of noise in dynamics w_mgn
      Number of rollouts N
      Time Horizon T
    Outputs:
      Estimated model (A_nom,B_nom) from least squares
    """

    d, p = B.shape

    # Storage matrices
    X_store = np.zeros((d, N, T + 1))
    U_store = np.zeros((p, N, T))

    # Simulate
    for k in range(N):
        x = x0
        X_store[:, k, 0] = x0.flatten()
        for t in range(T):
            u = np.random.randn(p, 1)
            w = w_mgn * np.random.randn(d, 1)
            x = A.dot(x) + B.dot(u) + w
            X_store[:, k, t + 1] = x.flatten()
            U_store[:, k, t] = u.flatten()

    # Solve for nominal model
    tmp = np.linalg.lstsq(
        np.vstack((X_store[:, :, 0:T].reshape(d, N * T), U_store.reshape(p, N * T))).T,
        X_store[:, :, 1 : (T + 1)].reshape(d, N * T).T,
    )[0]
    A_nom = tmp[0:d, :].T
    B_nom = tmp[d : (d + p), :].T
    return (A_nom, B_nom)


def random_search_linear_policy(
    A, B, Q, R, x0, w_mgn, N, T, explore_mag=4e-2, step_size=5e-1, batch_size=4
):
    """
    Arguments:
      Model (A, B)
      LQR Costs (Q, R)
      Initial State x0
      Magnitude of noise in dynamics w_mgn
      Number of rollouts N
      Time Horizon T

      hyperparameters:
        explore_mag = magnitude of the noise to explore
        step_size
        batch_size = number of directions per minibatches
        safeguard: maximum absolute value of entries of controller gain

    Outputs:
      Static Control Gain K optimized on LQR cost by random search
    """

    d, p = B.shape

    # initial condition for K
    K0 = 1e-3 * np.random.randn(p, d)

    #### ALGORITHM
    K = K0
    for k in range(N):
        reward_store = []
        mini_batch = np.zeros((p, d))
        for j in range(batch_size):
            V = explore_mag * np.random.randn(p, d)
            for sign in [-1, 1]:
                x = x0
                reward = 0
                for t in range(T):
                    # Update state
                    u = np.dot(K + sign * V, x)
                    w = w_mgn * np.random.randn(d, 1)
                    x = A.dot(x) + B.dot(u) + w
                    # Update reward: += - (x' Q x + u' R u)
                    reward += -np.dot(x.T, Q.dot(x)) - np.dot(u.T, R.dot(u))

                mini_batch += (reward * sign) * V
                reward_store.append(reward)
        K += (step_size / np.std(reward_store) / batch_size) * mini_batch

    return K


def uniform_random_linear_policy(A, B, Q, R, x0, w_mgn, N, T, linf_norm=3):
    """
    Arguments:
      Model (A,B)
      LQR Costs (Q,R)
      Initial State x0
      Magnitude of noise in dynamics w_mgn
      Number of rollouts N
      Time Horizon T

      hyperparameters
          linf_norm = maximum absolute value of entries of controller gain

    Outputs:
      Static Control Gain K optimized on LQR cost by uniformly sampling policies
      in bounded region
    """

    d, p = B.shape

    #### "ALGORITHM"
    best_K = np.empty((p, d))
    best_reward = -float("inf")
    for k in range(N):
        K = np.random.uniform(-linf_norm, linf_norm, (p, d))
        x = x0
        reward = 0
        for t in range(T):
            # Update state
            u = np.dot(K, x)
            w = w_mgn * np.random.randn(d, 1)
            x = A.dot(x) + B.dot(u) + w
            # Update reward: += - (x' Q x + u' R u)
            reward += -np.dot(x.T, Q.dot(x)) - np.dot(u.T, R.dot(u))

        if reward > best_reward:
            best_reward = reward
            best_K = K

    return best_K


def policy_gradient_linear_policy(
    A,
    B,
    Q,
    R,
    x0,
    w_mgn,
    N,
    T,
    explore_mag=5e-2,
    step_size=2,
    batch_size=40,
    safeguard=2,
):
    """
    Arguments:
      Model (A,B)
      LQR Costs (Q,R)
      Initial State x0
      magnitude of noise in dynamics w_mgn
      Number of rollouts N
      Time Horizon T

      hyperparameters
         explore_mag magnitude of the noise to explore
         step_size
         batch_size: number of stochastic gradients per minibatch
         safeguard: maximum absolute value of entries of controller gain

    Outputs:
      Static Control Gain K optimized on LQR cost by Policy Gradient
    """

    d, p = B.shape

    # initial condition for K
    K0 = 1e-3 * np.random.randn(p, d)

    # Storage matrices
    X_store = np.zeros((d, T))
    V_store = np.zeros((p, T))

    #### ALGORITHM
    K = K0
    baseline = 0
    for k in range(N):
        new_baseline = 0
        mini_batch = np.zeros((p, d))
        for j in range(batch_size):
            x = x0
            reward = 0
            for t in range(T):
                v = explore_mag * np.random.randn(p, 1)
                X_store[:, t] = x.flatten()
                V_store[:, t] = v.flatten()
                # Update state
                u = np.dot(K, x) + v
                w = w_mgn * np.random.randn(d, 1)
                x = A.dot(x) + B.dot(u) + w
                # Update reward: += - (x' Q x + u' R u)
                reward += -np.dot(x.T, Q.dot(x)) - np.dot(u.T, R.dot(u))
            mini_batch += ((reward - baseline) / batch_size) * np.dot(
                V_store, X_store.T
            )
            new_baseline += reward / batch_size
        K += step_size * mini_batch
        K = np.minimum(np.maximum(K, -safeguard), safeguard)
        baseline = new_baseline
    return K
