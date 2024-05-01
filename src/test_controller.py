from typing import Type
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation
from State import State
from Controls import Controls
from OCRLBot import OCRLBot
from PIDStabilizationBot import PIDStabilizationBot


BOOST_ACCELERATION = 991.666  # uu/s^2
J = np.diag([1, 1, 1])  # kg m^2
GRAVITY = 9.81  # m/s^2
FPS = 120


def f(state: State, controls: Controls) -> State:
    state.position += state.velocity / FPS
    state.velocity += np.array([0, 0, -GRAVITY]) / FPS
    if controls.boost:
        z_dir = state.orientation.apply([0, 0, 1])
        state.velocity += z_dir * BOOST_ACCELERATION / FPS

    q = state.orientation.as_quat()  # x, y, z, w
    q = np.roll(q, 1)  # w, x, y, z
    q = np.quaternion(*q)

    omega = np.quaternion(0, *state.angular_velocity)
    q += 0.5 * (1 / FPS) * omega * q
    q /= np.abs(q)
    q = quaternion.as_float_array(q)
    q = np.roll(q, -1)  # x, y, z, w
    state.orientation = Rotation.from_quat(q)

    torque = np.array([controls.roll, controls.pitch, controls.yaw])
    state.angular_velocity += np.linalg.inv(J) @ torque / FPS
    return state


def simulate(bot_class: Type[OCRLBot]):
    bot = bot_class(bot_class.__name__, 0, 0, enable_logging=False)

    state = State(0,
                  np.array([0, 0, 0]),
                  np.array([0, 0, 0]),
                  Rotation.from_euler("ZYX", [0, -90, 0], degrees=True),
                  np.array([0, 0, 0]))

    for i in range(100):
        controls = bot.update(state)
        state = f(state, controls)
        print(f"State={repr(state)}, Controls={repr(controls)}")


def main():
    simulate(PIDStabilizationBot)


if __name__ == '__main__':
    main()
