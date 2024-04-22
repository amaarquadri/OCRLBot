import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rotation
import matplotlib.pyplot as plt
from matplotlib import use


def S(vector: np.ndarray) -> np.ndarray:
    """
    Skew-symmetric matrix of a vector.
    """
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def S_inv(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Invert the skew-symmetric matrix to retrieve the original vector.
    """
    return np.array([rotation_matrix[2, 1], rotation_matrix[0, 2], rotation_matrix[1, 0]])


def dynamics(_, state, k_p, k_d):
    # Unpack the state
    R = state[:9].reshape(3, 3)
    omega = state[9:]

    # Desired rotation and angular velocity
    rotation_desired = Rotation.identity()
    omega_desired = np.array([0, 0, 0])

    # Current rotation object and error calculation
    rotation = Rotation.from_matrix(R)
    rotation_error = rotation_desired.inv() * rotation
    omega_error = omega - omega_desired

    R_error = rotation_error.as_matrix()
    torque = -k_p * S_inv(R_error - R_error.T) - k_d * omega_error

    R_dot = R @ S(omega)
    return np.concatenate((R_dot.flatten(), torque))


def main():
    # PD gains
    k_p, k_d = (1, 1)  # Proportional and derivative gains

    initial_state = np.concatenate((Rotation.random().as_matrix().flatten(), np.random.rand(3)))

    solution = solve_ivp(dynamics, (0, 10), initial_state, args=(k_p, k_d), method='RK45', atol=1e-8, rtol=1e-8)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # plot axis angles
    axs[0].plot(solution.t, [Rotation.from_matrix(solution.y[:9, i].reshape(3, 3)).as_rotvec()
                             for i in range(solution.y.shape[1])])
    axs[0].set_title('Axis Angle Evolution')
    axs[0].legend(['x', 'y', 'z'])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Axis Angle (rad)')
    axs[0].set_ylim(-np.pi, np.pi)

    # Angular velocity plot
    axs[1].plot(solution.t, [solution.y[9:, i] for i in range(solution.y.shape[1])])
    axs[1].set_title('Angular Velocity Evolution')
    axs[1].legend(['x', 'y', 'z'])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Angular Velocity (rad/s)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    use("TkAgg")
    main()
