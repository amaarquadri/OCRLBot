from typing import Type, Tuple, List
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use
import quaternion
from scipy.spatial.transform import Rotation
from State import State
from Controls import Controls
from OCRLBot import OCRLBot
from dynamic_mode_decomposition import perform_dmd

J = np.diag([1, 1, 1])  # kg m^2
BOOST_IN_X = False


def f(state: State, controls: Controls) -> State:
    state.position += state.velocity / OCRLBot.FPS
    state.velocity += np.array([0, 0, -OCRLBot.GRAVITY]) / OCRLBot.FPS
    if controls.boost:
        dir = state.orientation.apply([1, 0, 0] if BOOST_IN_X else [0, 0, 1])
        state.velocity += dir * OCRLBot.BOOST_ACCELERATION / OCRLBot.FPS

    q = state.orientation.as_quat()  # x, y, z, w
    q = np.roll(q, 1)  # w, x, y, z
    q = np.quaternion(*q)

    # https://gamedev.stackexchange.com/questions/108920/applying-angular-velocity-to-quaternion
    omega = np.quaternion(0, *state.angular_velocity)
    q += 0.5 * (1 / OCRLBot.FPS) * omega * q
    q /= np.abs(q)
    q = quaternion.as_float_array(q)
    q = np.roll(q, -1)  # x, y, z, w
    state.orientation = Rotation.from_quat(q)

    torque = np.array([controls.roll, controls.pitch, controls.yaw])
    state.angular_velocity += np.linalg.inv(J) @ torque / OCRLBot.FPS
    return state


def collect_synthetic_dmd_data(count=10_000):
    state = State(0,
                  np.array([0., 0., 0.]),
                  np.array([0., 0., 0.]),
                  Rotation.from_euler("ZYX", [0, -90, 0], degrees=True) if BOOST_IN_X else Rotation.identity(),
                  np.array([0., 0., 0.]))

    x_kp1s: List[State] = []
    x_ks: List[State] = []
    u_ks: List[Controls] = []
    for i in range(count):
        state = State(0,
                      np.array([0., 0., 0.]),
                      np.array([0., 0., 0.]),
                      Rotation.from_euler("ZYX", [360 * np.random.random(), -90, 0], degrees=True)
                      if BOOST_IN_X else
                      Rotation.from_euler("ZYX", [360 * np.random.random(), 0, 0], degrees=True),
                      np.array([0., 0., 0.]), )
        controls = Controls.random()
        new_state = f(state, controls)
        x_kp1s.append(deepcopy(new_state))
        x_ks.append(state)
        u_ks.append(controls)

    return x_kp1s, x_ks, u_ks


def simulate(bot_class: Type[OCRLBot], t_f=10) -> List[Tuple[float, State, Controls]]:
    bot = bot_class(bot_class.__name__, 0, 0, enable_logging=False)

    state = State(0,
                  np.array([100., 0., 0.]),
                  np.array([0., 0., 0.]),
                  Rotation.from_euler("ZYX", [0, -90, 0], degrees=True) if BOOST_IN_X else Rotation.identity(),
                  np.array([0., 0., 0.]))

    history = []
    for t in np.arange(0, t_f, 1 / OCRLBot.FPS):
        controls, desired_orientation = bot.update(state)
        history.append((t, deepcopy(state), deepcopy(controls), desired_orientation))
        state = f(state, controls)

    return history


def plot_results(history: List[Tuple[float, State, Controls]]) -> None:
    t = [t for t, *_ in history]

    plt.figure()
    plt.plot(t, [state.position[0] for _, state, *_ in history], label="x")
    plt.plot(t, [state.position[1] for _, state, *_ in history], label="y")
    plt.plot(t, [state.position[2] for _, state, *_ in history], label="z")

    # rotvecs = np.array([np.rad2deg(state.orientation.as_rotvec()) for _, state, *_ in history])
    # rotvecs = np.array([np.rad2deg(desired_orientation.as_rotvec()) for *_, desired_orientation in history])
    # plt.plot(t, rotvecs[:, 0], label="rotvec_x")
    # plt.plot(t, rotvecs[:, 1], label="rotvec_y")
    # plt.plot(t, rotvecs[:, 2], label="rotvec_z")

    # mats = np.array([state.orientation.as_matrix() for _, state, *_ in history])
    # mats = np.array([desired_orientation.as_matrix() for *_, desired_orientation in history])
    # plt.plot(t, mats[:, 0, 2], label="mat_xz")

    # rpys = np.array([state.orientation.as_euler("ZYX", degrees=True)[::-1] for _, state, *_ in history])
    # rpys = np.array([desired_orientation.as_euler("ZYX", degrees=True)[::-1] for *_, desired_orientation in history])
    # plt.plot(t, rpys[:, 0], label="roll")
    # plt.plot(t, rpys[:, 1], label="pitch")
    # plt.plot(t, rpys[:, 2], label="yaw")
    #
    # plt.plot(t, [np.rad2deg(desired_orientation.as_rotvec()[1]) for *_, desired_orientation in history],
    #          label="desired_pitch")

    torques = np.array([[controls.roll, controls.pitch, controls.yaw] for _, _, controls, _ in history])
    # plt.plot(t, torques[:, 0], label="roll_torque")
    # plt.plot(t, torques[:, 1], label="pitch_torque")
    # plt.plot(t, torques[:, 2], label="yaw_torque")

    boost_dirs = np.array([state.orientation.apply([1, 0, 0] if BOOST_IN_X else [0, 0, 1])
                           for _, state, *_ in history])
    # plt.plot(t, boost_dirs[:, 0], label="boost_dir_x")
    # plt.plot(t, boost_dirs[:, 1], label="boost_dir_y")
    # plt.plot(t, boost_dirs[:, 2], label="boost_dir_z")

    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position (Unreal Units)")
    plt.title("Position vs. Time")
    plt.show()


def main():
    # perform DMD
    # A, B = perform_dmd(*collect_synthetic_dmd_data(), body_velocities=False)
    # np.save("models/A.npy", A)
    # np.save("models/B.npy", B)

    # simulate
    from PIDStabilizationBot import PIDStabilizationBot as Bot
    # from LQRBot import LQRBot as Bot
    history = simulate(Bot)
    plot_results(history)


if __name__ == '__main__':
    use("TkAgg")
    main()
