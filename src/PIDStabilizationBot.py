import numpy as np
from scipy.spatial.transform import Rotation
from OCRLBot import OCRLBot
from State import State
from Controls import Controls
from Setpoint import Setpoint


class PIDStabilizationBot(OCRLBot):
    def __init__(self, name, team, index, enable_logging=True):
        super().__init__(name, team, index, enable_logging=enable_logging)
        self.kp_position = 1 * np.identity(3)
        self.kd_position = 0 * np.identity(3)
        self.kp_orientation = 1 * np.identity(3)
        self.kd_orientation = 0 * np.identity(3)

    def get_desired_orientation(self, state: State, setpoint: Setpoint) -> Rotation:
        """
        Get the desired orientation as a Rotation.
        """
        # From Robot Mobility slides, but assuming m = 1
        position_error = state.position - setpoint.position
        desired_acceleration = -self.kp_position @ position_error - self.kd_position @ state.velocity
        desired_acceleration[2] = OCRLBot.GRAVITY  # we control z via the energy of the system instead since our boost is binary

        z_d = desired_acceleration / np.linalg.norm(desired_acceleration)  # units of acceleration instead of force
        n = np.cross([0, 0, 1], z_d)
        n_magnitude = np.linalg.norm(n)
        if np.isclose(n_magnitude, 0):
            R_EB = Rotation.identity()
        else:
            rho = np.minimum(np.arcsin(n_magnitude), np.deg2rad(10))
            # self.logger.debug(f"n={n}, n_magnitude={n_magnitude}, rho={np.rad2deg(rho)}, z_d={z_d}")
            R_EB = Rotation.from_rotvec(rho * n / n_magnitude)
        R_AE = Rotation.from_euler("ZYX", [setpoint.yaw, 0, 0])
        return R_AE * R_EB * Rotation.from_euler("ZYX", [0, np.pi / 2, 0])  # pitch up 90 degrees first

    def calc_torque(self, state: State, desired_orientation: Rotation) -> np.ndarray:
        # From Robot Mobility slides, but assuming J = 1
        R_e = (desired_orientation.inv() * state.orientation).as_matrix()

        def S_inv(rotation_matrix: np.ndarray) -> np.ndarray:
            return np.array([rotation_matrix[2, 1], rotation_matrix[0, 2], rotation_matrix[1, 0]])

        torque = -self.kp_orientation @ S_inv(R_e - R_e.T) - self.kd_orientation @ state.angular_velocity
        torque = np.clip(torque, -1, 1)
        return torque

    @staticmethod
    def should_boost(state: State, setpoint: Setpoint) -> bool:
        v_z = state.velocity[2]
        if v_z == 0:
            # special case since the energy method breaks down when velocity is exactly zero
            return state.position[2] < setpoint.position[2]

        specific_energy = 0.5 * v_z ** 2 + (state.position[2] - setpoint.position[2]) * OCRLBot.GRAVITY  # relative to setpoint
        # boosting adds energy when going up and subtracts energy when going down,
        # so we boost if v_z and specific_energy have opposite signs
        return specific_energy * v_z < 0

    def update(self, state: State) -> Controls:
        controls = Controls()

        # Set the setpoint
        setpoint = Setpoint(position=np.array([0, 0, 1000]), yaw=0)

        # Calculate the controls
        desired_orientation = self.get_desired_orientation(state, setpoint)
        # desired_orientation = Rotation.from_euler("ZYX", [90, 80, 0], degrees=True)
        self.logger.debug(f"Desired orientation={desired_orientation.as_euler('ZYX', degrees=True)}")
        torques = self.calc_torque(state, desired_orientation)
        controls.roll, controls.pitch, controls.yaw = torques
        controls.boost = PIDStabilizationBot.should_boost(state, setpoint)
        self.logger.info(f"Frame: State={repr(state)}, Controls={repr(controls)}")
        return controls
