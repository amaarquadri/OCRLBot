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
        """
        Implements a simple bang-bang controller using energy calculations.
        """
        v_z: float = state.velocity[2]
        delta_z: float = state.position[2] - setpoint.position[2]

        if delta_z == 0:
            return v_z <= 0

        if v_z == 0:
            return delta_z < 0

        if v_z > 0 and delta_z > 0:
            return False  # we're already above the setpoint and moving up, so boosting is counterproductive
        if v_z < 0 and delta_z < 0:
            return True  # we're below the setpoint and moving down, so boosting is an obvious necessity

        # at this point v_z and delta_z must have opposite signs
        specific_kinetic_energy = 0.5 * v_z ** 2
        if delta_z < 0:  # v_z > 0
            specific_gravitational_energy = delta_z * OCRLBot.GRAVITY
            # we're already going up towards the setpoint, so boost if we don't have enough energy to reach there
            return specific_kinetic_energy + specific_gravitational_energy < 0

        # at this point v_z < 0 and delta_z > 0
        # assume that we are boosting (which effectively modifies gravity to be negative)
        # and check to see if we should stop boosting
        specific_gravitational_energy_when_boosting = delta_z * (OCRLBot.GRAVITY - OCRLBot.BOOST_ACCELERATION)
        # stop boosting if we don't have enough energy to reach the setpoint
        return not (specific_kinetic_energy + specific_gravitational_energy_when_boosting < 0)

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
