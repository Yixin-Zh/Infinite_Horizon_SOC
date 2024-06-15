import casadi as ca
import numpy as np
from utils import lissajous as ref_traj
from utils import time_step as dt
import math

# environment parameters
obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
size = 3

# hyperparameters for value function
Q = ca.diag([5, 5 ])  # position cost
q = 0.005  # orientation cost
R = ca.diag([0.008, 0.008])  # control cost

gamma = 0.85  # discount factor

class CEC:
    def __init__(self):
        self.T = 30  # planning horizon
        self.ref_traj = ref_traj
        self.obstacles = obstacles

    def __call__(self, iter: int, state: np.ndarray, init_ref_state: np.ndarray):
        """
        Given the time step, current state
        Args:
            iter (int): time step
            state (np.ndarray): current state
        Returns:
            np.ndarray: control input
        """
        # TODO: define optimization variables
        U = ca.MX.sym("U", 2, self.T-1)  # control (v, w)
        # TODO: define optimization constraints
        # TODO: define optimization cost
        j = 0
        constrains = []
        # initial state constraint
        error = ca.MX(self.compute_error(state, init_ref_state))
        for t in range(self.T-1):
            ref_state = self.ref_traj(iter)
            next_ref_state = self.ref_traj(iter + 1)
            # cost function
            j += (gamma**t)*self.cost_function(error, U[:, t])
            # obstacle constraints
            obstacles_constraint = self.obstacles_constraint(error+ref_state)
            for c in obstacles_constraint:
                constrains.append(c)
            # boundary constraints
            boundary_constraint = self.boundary_constraint(error+ref_state)
            for b in boundary_constraint:
                constrains.append(b)

            next_error = self.error_motion_model(error, ref_state, next_ref_state, U[:, t])
            error = next_error
            # update ref_state for next iteration
            iter += 1
        # terminal state cost
        j += self.cost_function(error, ca.MX([0, 0]))

        g = ca.vertcat(*constrains)

        # TODO: define optimization solver
        nlp = {"x": ca.reshape(U, -1, 1), "f": j, "g": g}
        x = nlp["x"]

        solver = ca.nlpsol("solver", "ipopt", nlp)
        sol = solver(
            x0=np.tile([0, 0], self.T-1),  # initial guess
            lbx=np.tile([0, -1], self.T-1),  # velocity lower bound
            ubx=np.tile([1, 1], self.T-1),  # velocity upper bound
            lbg = np.tile([0], g.shape[0]),  # lower bound for constraints
        )
        x = sol["x"]  # get the solution

        # TODO: extract the control input from the solution
        u = np.array(x[:2]).reshape(-1)

        return u

    def error_motion_model(self, error_state, ref_state, next_ref_state, control, time_step=dt):
        """
        Given the error state, control input, and time step, return the next error state.
        """
        G = ca.MX(3, 2)
        G[0, 0] = ca.cos(error_state[2] + ref_state[2])
        G[0, 1] = 0
        G[1, 0] = ca.sin(error_state[2] + ref_state[2])
        G[1, 1] = 0
        G[2, 0] = 0
        G[2, 1] = 1

        next_error_state = error_state\
                           + ca.mtimes(G, control) * time_step\
                           + (ca.MX(ref_state) - ca.MX(next_ref_state))
        #
        next_error_state[2] = ca.fmod(next_error_state[2] + ca.pi, 2 * ca.pi) - ca.pi
        return next_error_state
    def obstacles_constraint(self, state,):
        """
        Given the state and obstacles, return the obstacle constraints.
        """
        constraints = []
        for obs in obstacles:
            obs = ca.MX(obs)
            constraints.append(ca.norm_2(state[:2] - obs[:2]) - obs[2])
        return constraints

    def boundary_constraint(self, state,):
        """
        Given the state and boundary, return the boundary constraints.
        """
        constraints = []
        lower_bound = ca.MX([-size, -size])
        upper_bound = ca.MX([size, size])
        constraints.append(state[:2] - lower_bound)
        constraints.append(upper_bound - state[:2])
        return constraints

    def cost_function(self, error_state, control):
        """
        Given the error state and control input, return the cost.
        """
        error_pos = error_state[:2]
        error_ori = ca.fmod(error_state[2] + ca.pi, 2 * ca.pi) - ca.pi
        cost = (ca.mtimes(ca.mtimes(error_pos.T, Q), error_pos)
                + q * (1-ca.cos(error_ori))**2
                + ca.mtimes(ca.mtimes(control.T, R), control))
        return cost

    def compute_error(self, state, ref_state):
        """
        Given the state and reference state, return the error state.
        """
        error = state - ref_state
        error[2] = np.fmod(error[2] + np.pi, 2 * np.pi) - np.pi
        return error

if __name__ == "__main__":
    cec = CEC()


