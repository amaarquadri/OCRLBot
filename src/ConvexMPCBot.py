from time import perf_counter
import numpy as np
import cvxpy as cp
from OCRLBot import OCRLBot
from State import State
from Controls import Controls
from RingCheckpointManager import RingCheckpointManager


class ConvexMPCBot(OCRLBot):
    MPC_HORIZON = 240  # frames

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.checkpoint_manager = RingCheckpointManager("rings/leth_neon_heights_rings_level1.txt")

        self.n, self.m = 12, 4
        self.A = np.load("models/A.npy")
        self.B = np.load("models/B.npy")
        self.Q = np.eye(3)
        self.Q_f = 2 * self.Q
        self.R = np.eye(self.m)


    def update(self, state: State) -> Controls:
        ring = self.checkpoint_manager.get_current_ring_checkpoint(state)
        x_0 = np.array([*state.position, *state.velocity,
                        *state.orientation.as_euler("ZYX", degrees=True),
                        *state.angular_velocity])
        x_g = ring.position

        x = cp.Variable((12, ConvexMPCBot.MPC_HORIZON + 1))
        u = cp.Variable((4, ConvexMPCBot.MPC_HORIZON))

        cost = 0
        for k in range(ConvexMPCBot.MPC_HORIZON):
            cost += 0.5 * cp.quad_form(x[:3, k] - x_g, self.Q) + 0.5 * cp.quad_form(u[:, k], self.R)
        cost += 0.5 * cp.quad_form(x[:3, -1] - x_g, self.Q_f)

        constraints = []
        for k in range(ConvexMPCBot.MPC_HORIZON):
            constraints.append(x[:, k + 1] == self.A @ x[:, k] + self.B @ u[:, k])

        for k in range(ConvexMPCBot.MPC_HORIZON + 1):
            constraints += [x[3, k] <= 10, x[3, k] >= -10]
            constraints += [x[4, k] <= 10, x[4, k] >= -10]

            # v_norm = cp.norm(x[6:9, k], 'inf') #x[6, k]**2 + x[7, k]**2 + x[8, k]**2

            constraints += [cp.norm(x[6:9, k]) <= 2200]  # , v_norm >= v_min]
            constraints += [cp.norm(x[9:12, k]) <= 5.5]  # , cp.norm(x[9:12, k]) >= omega_min]

        constraints.append(x[:, 0] == x_0)
        constraints.append(x[:3, -1] == x_g)

        u_min = -np.ones(self.m)  # Minimum value for each control
        u_max = np.ones(self.m)  # Maximum value for each control
        u_min[-1] = 0
        for k in range(ConvexMPCBot.MPC_HORIZON):
            constraints += [u[:, k] <= u_max, u[:, k] >= u_min]

        prob = cp.Problem(cp.Minimize(cost), constraints)

        # Solve the problem
        start_time = perf_counter()
        print(prob.solve())
        print(f"Optimization took {perf_counter() - start_time:.2f} seconds")

        controls = Controls()
        controls.roll, controls.pitch, controls.yaw, boost = u.value[:, 0]
        controls.boost = boost > 0.5
        return controls
