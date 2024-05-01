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
from PIDStabilizationBot import PIDStabilizationBot
from dynamic_mode_decomposition import perform_dmd


BOOST_ACCELERATION = 991.666  # uu/s^2
J = np.diag([1, 1, 1])  # kg m^2


def f(state: State, controls: Controls) -> State:
    state.position += state.velocity / OCRLBot.FPS
    state.velocity += np.array([0, 0, -OCRLBot.GRAVITY]) / OCRLBot.FPS
    if controls.boost:
        # TODO: once we get the simple case working,
        #  change this to use x_dir since that is how the boost is applied
        z_dir = state.orientation.apply([0, 0, 1])
        state.velocity += z_dir * BOOST_ACCELERATION / OCRLBot.FPS

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
                  np.array([0, 0, 0]),
                  np.array([0, 0, 0]),
                  Rotation.from_euler("ZYX", [0, -90, 0], degrees=True),
                  np.array([0, 0, 0]))

    x_kp1s = []
    x_ks = []
    u_ks = []
    for i in range(count):
        controls = Controls.random()
        new_state = f(state, controls)
        x_kp1s.append(deepcopy(new_state))
        x_ks.append(state)
        u_ks.append(controls)

    return x_kp1s, x_ks, u_ks


def simulate(bot_class: Type[OCRLBot], t_f=10) -> List[Tuple[float, State, Controls]]:
    bot = bot_class(bot_class.__name__, 0, 0, enable_logging=False)

    state = State(0,
                  np.array([0., 0., 0.]),
                  np.array([0., 0., 0.]),
                  Rotation.identity(),
                  # Rotation.from_euler("ZYX", [0, -90, 0], degrees=True),
                  np.array([0., 0., 0.]))

    history = []
    for t in np.arange(0, t_f, 1 / OCRLBot.FPS):
        controls = bot.update(state)
        history.append((t, deepcopy(state), deepcopy(controls)))
        state = f(state, controls)

    return history


def plot_results(history: List[Tuple[float, State, Controls]]) -> None:
    t = [t for t, *_ in history]

    plt.figure()
    plt.plot(t, [state.position[0] for _, state, _ in history], label="x")
    plt.plot(t, [state.position[1] for _, state, _ in history], label="y")
    plt.plot(t, [state.position[2] for _, state, _ in history], label="z")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Position vs. Time")
    plt.show()


def main():
    # perform DMD
    A, B = perform_dmd(*collect_synthetic_dmd_data())
    np.save("models/A.npy", A)
    np.save("models/B.npy", B)

    # simulate
    history = simulate(PIDStabilizationBot)
    plot_results(history)


if __name__ == '__main__':
    use("TkAgg")
    main()
