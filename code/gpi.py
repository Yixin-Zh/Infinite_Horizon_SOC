from dataclasses import dataclass
import numpy as np
from value_function import ValueFunction
import utils
import os
from scipy.stats import multivariate_normal
from scipy.linalg import inv, det, cholesky
from joblib import Parallel, delayed
import time
# environment parameters
traj = utils.lissajous
obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
fine_grid = np.linspace(-0.25, 0.25, 7)

# Medium grid for [-0.5, 0.5] divided into 5 parts excluding the overlap with fine grid
medium_grid = np.concatenate([
    np.linspace(-0.65, -0.25, 5),
    np.linspace(0.25, 0.65, 5)
])

# Coarse grid for [-3, 3] divided into 5 parts excluding the overlap with medium and fine grid
coarse_grid = np.concatenate([
    np.linspace(-1.5, -0.65, 6),
    np.linspace(0.65, 1.5, 6)
])

# Combine all grids
ex_space = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
ey_space = ex_space
eth_space = np.linspace(-np.pi, np.pi, 33)
v_space = np.linspace(0,1, 11)
w_space = np.linspace(-1, 1, 11)

# cost function parameters
Q = np.diag([5, 5])
q = 0.02
R = np.diag([.01, .01])
gamma = 0.9
# GPI parameters
num_evals = 10
collision_margin = 0.05
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
v_ex_space = np.linspace(-2, 1, 21) #
v_ey_space = np.linspace(-1, 1, 21) # ?
v_eth_space = np.linspace(-np.pi, np.pi, 40) # ?
v_alpha = 0.1
v_beta_t = 0.1
v_beta_e = 0.1
v_lr = 0.01
v_batch_size = 10

def process_time_step(t, all_pts, all_controls, T, traj, error_motion_model, get_top_pts):
    transition_matrix_t = np.zeros((all_pts.shape[0], all_controls.shape[0], 8, 2), dtype=np.float16)
    traj_t = traj(t)  # Assuming traj function is defined elsewhere
    traj_t_prime = traj(t + 1)
    t1 = time.time()

    for i, pt in enumerate(all_pts):
        next_error = error_motion_model(pt, traj_t, traj_t_prime, all_controls)
        top_pts = get_top_pts(next_error)
        transition_matrix_t[i] = top_pts

    print(f"Time for t={t}: {time.time() - t1} seconds")
    return transition_matrix_t

def compute_costs_for_time_step(t, all_pts, all_controls, traj, cost_function):
    """
    Computes the costs for all states and controls at a given time step.
    """
    ref_state = traj(t)
    costs = np.zeros((all_pts.shape[0], all_controls.shape[0]), dtype=np.float16)
    t1 = time.time()
    for i, pt in enumerate(all_pts):
        error = pt - ref_state
        costs[i] = cost_function(error, ref_state,all_controls)
    print(f"Time for t={t}: {time.time() - t1} seconds")
    return costs

@dataclass
class GpiConfig:
    traj: callable
    obstacles: np.ndarray
    ex_space: np.ndarray
    ey_space: np.ndarray
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray
    Q: np.ndarray
    q: float
    R: np.ndarray
    gamma: float
    num_evals: int  # number of policy evaluations in each iteration
    collision_margin: float
    V: ValueFunction  # your value function implementation
    output_dir: str
    # used by feature-based value function
    v_ex_space: np.ndarray
    v_ey_space: np.ndarray
    v_etheta_space: np.ndarray
    v_alpha: float
    v_beta_t: float
    v_beta_e: float
    v_lr: float
    v_batch_size: int  # batch size if GPU memory is not enough

config = GpiConfig(traj, obstacles, ex_space, ey_space, eth_space, v_space, w_space,
                          Q, q, R, gamma, num_evals, collision_margin,
                          ValueFunction, output_dir, v_ex_space, v_ey_space, v_eth_space, v_alpha, v_beta_t, v_beta_e, v_lr, v_batch_size)
class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config
        # precompute all possible states and controls
        self.all_pts = self.get_all_pts()
        self.all_controls = self.get_all_controls()
        # pdf parameters
        self.D = 3
        self.sigma = np.diag(utils.sigma)  # standard deviation of the error
        self.cov = self.sigma ** 2  # covariance matrix
        self.cov_inv = np.linalg.inv(self.cov) # inverse of the covariance matrix
        self.const = 1.0 / np.sqrt((2 * np.pi) ** self.D * np.linalg.det(self.cov)) # constant term in the PDF formula
        # period of the reference trajectory
        self.T = 100
        self.policy = np.load(os.path.join(self.config.output_dir, "optimal_policy.npy"))

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        error = cur_state - cur_ref_state
        error[2] = np.fmod(error[2] + np.pi, 2 * np.pi) - np.pi  # Normalize angle error

        # Using searchsorted to find the closest indices for each state dimension
        ex_index = np.searchsorted(self.config.ex_space, error[0], side='left') - 1
        ey_index = np.searchsorted(self.config.ey_space, error[1], side='left') - 1
        eth_index = np.searchsorted(self.config.eth_space, error[2], side='left') - 1

        # Ensure the indices are within the valid range
        ex_index = np.clip(ex_index, 0, len(self.config.ex_space) - 1)
        ey_index = np.clip(ey_index, 0, len(self.config.ey_space) - 1)
        eth_index = np.clip(eth_index, 0, len(self.config.eth_space) - 1)

        # Handle angle wrapping by using modulo operation for the angular dimension index
        eth_index = np.mod(eth_index, len(self.config.eth_space))

        # Assuming the state space is a flat array of combined indices, calculate a single index
        cur_index = ex_index * len(self.config.ey_space) * len(self.config.eth_space) + \
                    ey_index * len(self.config.eth_space) + eth_index

        points = self.all_pts[cur_index]
        print("#"*10)
        print(np.linalg.norm(points - error))   # debug
        # Modulate t to [0, T)
        t = t % self.T

        # Get the control index from the policy matrix
        control_index = self.policy[t, cur_index]
        return self.all_controls[control_index]

    def compute_transition_matrix(self):
        results = Parallel(n_jobs=-1)(
            delayed(process_time_step)(t, self.all_pts, self.all_controls, self.T, traj, self.error_motion_model,
                                       self.get_top_pts)
            for t in range(self.T))
        transition_matrix = np.array(results, dtype=np.float16)
        # save the transition matrix
        np.save(os.path.join(self.config.output_dir, "transition_matrix.npy"), transition_matrix)
        return transition_matrix
    def compute_stage_costs(self):
        """
        Compute the stage costs in advance to speed up the GPI algorithm.
        Uses parallel computation to improve performance.
        """
        # Parallel computation across time steps
        results = Parallel(n_jobs=-1)(delayed(compute_costs_for_time_step)(
            t, self.all_pts, self.all_controls, self.config.traj, self.cost_function)
                                      for t in range(self.T))
        stage_costs = np.array(results, dtype=np.float16)
        # save the stage costs
        np.save(os.path.join(self.config.output_dir, "stage_costs.npy"), stage_costs)
    def cost_function(self, error, ref_state, control):
        """
        Compute the cost function.
        Args:
            error: [3,]
            control: [N, 2]
        Returns:
            cost: float
        """
        error_pos = error[:2]
        error_ori = np.fmod(error[2] + np.pi, 2 * np.pi) - np.pi
        cost1 = error_pos @ self.config.Q @ error_pos\
                + self.config.q * (1 - np.cos(error_ori)) ** 2 # shape: [1,]
        state = error + ref_state
        # make sure not in obstacle
        obstacle1 = self.config.obstacles[0]
        obstacle2 = self.config.obstacles[1]
        if np.linalg.norm(state[:2] - obstacle1[:2]) < obstacle1[2] + self.config.collision_margin\
            or np.linalg.norm(state[:2] - obstacle2[:2]) < obstacle2[2] + self.config.collision_margin:
            cost1 += 0
        # # ensure not out of boundary
        # if np.abs(state[0]) > 3 or np.abs(state[1]) > 3:
        #     cost1 += 10
        cost1 = np.tile(cost1, (control.shape[0],))
        cost2 = np.einsum('ij,jk,ik->i', control, self.config.R, control)

        # sum the costs
        cost = cost1 + cost2
        return cost

    def error_motion_model(self, error, ref_state, next_ref_state, control):
        """
        deterministic error motion model
        Args:
            error: [3,]
            ref_state: [3,]
            next_ref_state: [3,]
            control: [N, 2]
        Returns:
            error at time t+1 [N, 3]
        """
        # 1. modulate the angle to [-pi, pi]
        N = control.shape[0]
        error[2] = np.fmod(error[2] + np.pi, 2 * np.pi) - np.pi
        # 2. compute G matrix shape
        G = np.zeros((3, 2))
        G[0, 0] = np.cos(error[2] + ref_state[2])
        G[1, 0] = np.sin(error[2] + ref_state[2])
        G[2, 1] = 1
        # 3. ref_state - next_ref_state
        ref_state = np.tile(ref_state, (N, 1))
        next_ref_state = np.tile(next_ref_state, (N, 1))
        ref_change = ref_state - next_ref_state
        # make error to [N, 3]
        error = np.tile(error, (N, 1))
        # 4. compute the next error
        next_error = error + (G @ control.T * utils.time_step).T + (ref_change)
        # 5. modulate the angle to [-pi, pi]
        # next_error shape: [N, 3]
        next_error[:, 2] = np.fmod(next_error[:, 2] + np.pi, 2 * np.pi) - np.pi
        return next_error

    def get_top_pts(self, error):
        N = error.shape[0]

        ex_indices = np.searchsorted(self.config.ex_space, error[:, 0], side='right') - 1
        ey_indices = np.searchsorted(self.config.ey_space, error[:, 1], side='right') - 1
        eth_indices = np.searchsorted(self.config.eth_space, error[:, 2], side='right') - 1

        # Clipping indices to ensure they stay within the valid range
        ex_indices = np.clip(ex_indices, 1, len(self.config.ex_space) - 2)
        ey_indices = np.clip(ey_indices, 1, len(self.config.ey_space) - 2)
        eth_indices = np.clip(eth_indices, 0, len(self.config.eth_space) - 1)  # Allow full range for angle initially

        # Generate all possible combinations of [-1, 1] shifts for the indices
        shift = np.array([-1, 1])
        dx, dy, dth = np.meshgrid(shift, shift, shift, indexing='ij')
        shifts = np.stack([dx.ravel(), dy.ravel(), dth.ravel()], axis=-1)

        # Compute the new indices using broadcasting
        all_ex_indices = ex_indices[:, None] + shifts[None, :, 0]
        all_ey_indices = ey_indices[:, None] + shifts[None, :, 1]
        all_eth_indices = eth_indices[:, None] + shifts[None, :, 2]

        # Wrap around for angular dimension
        all_eth_indices = np.mod(all_eth_indices, len(self.config.eth_space))

        # Convert these indices to linear indices
        flat_indices = (all_ex_indices * len(self.config.ey_space) * len(self.config.eth_space) +
                        all_ey_indices * len(self.config.eth_space) +
                        all_eth_indices).astype(int)

        # Fetch the closest points
        closest_pts = self.all_pts[flat_indices]

        # Compute Mahalanobis distance and transform to log probabilities
        diff = error[:, None, :] - closest_pts
        log_prob = -0.5 * np.einsum('nij,jk,nik->ni', diff, self.cov_inv, diff)

        # Subtract the max log probability to stabilize numerical computation
        max_log_prob = np.max(log_prob, axis=1, keepdims=True)
        log_prob -= max_log_prob

        # Convert log probabilities to probabilities
        pdf_values = np.exp(log_prob)
        sum_pdf_values = pdf_values.sum(axis=-1, keepdims=True)
        pdf_values /= sum_pdf_values

        # Combine indices and probabilities into a final array
        top = np.concatenate((flat_indices[..., np.newaxis], pdf_values[..., np.newaxis]), axis=2)
        return top.astype(np.float32)


    def get_all_pts(self):
        ex, ey, eth = np.meshgrid(self.config.ex_space, self.config.ey_space, self.config.eth_space, indexing='ij')
        all_pts = np.column_stack((ex.ravel(), ey.ravel(), eth.ravel()))
        return all_pts
    def get_all_controls(self):
        v, w = np.meshgrid(self.config.v_space, self.config.w_space, indexing='ij')
        all_controls = np.column_stack((v.ravel(), w.ravel()))
        return all_controls



if __name__ == "__main__":
    GpiConfig = GpiConfig(traj, obstacles, ex_space, ey_space, eth_space, v_space, w_space,
                          Q, q, R, gamma, num_evals, collision_margin,
                          ValueFunction, output_dir, v_ex_space, v_ey_space, v_eth_space, v_alpha, v_beta_t, v_beta_e, v_lr, v_batch_size)
    gpi = GPI(GpiConfig)
    import time
    start = time.time()
    gpi.compute_transition_matrix()
    stage_cost = gpi.compute_stage_costs()
    print(time.time()-start)
    # transition_matrix = gpi.compute_transition_matrix()

