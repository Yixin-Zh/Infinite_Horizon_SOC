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
ex_space = np.linspace(-2, 2, 21)
ey_space = np.linspace(-2, 2, 21)
eth_space = np.linspace(-np.pi, np.pi, 40)
v_space = np.linspace(0,1, 6)
w_space = np.linspace(-1, 1, 11)
# cost function parameters
Q = np.diag([5, 5])
q = 0.02
R = np.diag([0.008, 0.008])
gamma = 0.9
# GPI parameters
num_evals = 10
collision_margin = 0.1 # ?
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
v_ex_space = np.linspace(-2, 2, 21) #
v_ey_space = np.linspace(-2, 2, 21) # ?
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
        # TODO: your implementation
        raise NotImplementedError

    def state_metric_to_index(self, metric_state: np.ndarray) :
        """
        Convert the metric state to grid indices according to your descretization design.
        Args:
            metric_state (np.ndarray): metric state [3,N]
        Returns:
            array of indices: ex [N,], ey [N,], eth [N,]
        """
        ex = np.digitize(metric_state[0], self.config.ex_space, right=True)
        ey = np.digitize(metric_state[1], self.config.ey_space, right=True)
        eth = np.digitize(metric_state[2], self.config.eth_space, right=True)
        return ex, ey, eth

    def state_index_to_metric(self, state_index):
        """
        Convert the grid indices to metric state according to your descretization design.
        Args:
            state_index (tuple): (ex, ey, eth)
        Returns:
            np.ndarray: metric state
        """
        ex = self.config.ex_space[state_index[0]]
        ey = self.config.ey_space[state_index[1]]
        eth = self.config.eth_space[state_index[2]]
        return ex, ey, eth

    def control_metric_to_index(self, control_metric: np.ndarray):
        """
        Args:
            control_metric: [2, N] array of controls in metric space
        Returns:
            [N, ] array of indices in the control space
        """
        v: np.ndarray = np.digitize(control_metric[0], self.config.v_space, right=True)
        w: np.ndarray = np.digitize(control_metric[1], self.config.w_space, right=True)
        return v, w

    def control_index_to_metric(self, v: np.ndarray, w: np.ndarray):
        """
        Args:
            v: [N, ] array of indices in the v space
            w: [N, ] array of indices in the w space
        Returns:
            [2, N] array of controls in metric space
        """
        return self.config.v_space[v], self.config.w_space[w]

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
        """
        # TODO: your implementation
        raise NotImplementedError

    def init_value_function(self):
        """
        Initialize the value function.
        """
        # TODO: your implementation
        raise NotImplementedError

    def evaluate_value_function(self):
        """
        Evaluate the value function. Implement this function if you are using a feature-based value function.
        """
        # TODO: your implementation
        raise NotImplementedError

    @utils.timer
    def policy_improvement(self):
        """
        Policy improvement step of the GPI algorithm.
        """
        # TODO: your implementation
        raise NotImplementedError

    @utils.timer
    def policy_evaluation(self):
        """
        Policy evaluation step of the GPI algorithm.
        """
        # TODO: your implementation
        raise NotImplementedError

    def compute_policy(self, num_iters: int) -> None:
        """
        Compute the policy for a given number of iterations.
        Args:
            num_iters (int): number of iterations
        """
        # TODO: your implementation
        raise NotImplementedError

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
        G[0, 1] = 0
        G[1, 0] = np.sin(error[2] + ref_state[2])
        # 3 ref_state - next_ref_state
        ref_state = np.tile(ref_state, (N, 1))
        next_ref_state = np.tile(next_ref_state, (N, 1))
        ref_change = ref_state - next_ref_state
        # make error to [N, 3]
        error = np.tile(error, (N, 1))
        # 4. compute the next error
        next_error = error + (G @ control.T * utils.time_step).T + (ref_state - next_ref_state)
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
        eth_indices = np.clip(eth_indices, 1, len(self.config.eth_space) - 2)

        # Generate all possible combinations of [-1, 1] shifts for the indices
        shift = np.array([-1, 1])
        dx, dy, dth = np.meshgrid(shift, shift, shift, indexing='ij')
        shifts = np.stack([dx.ravel(), dy.ravel(), dth.ravel()], axis=-1)

        # Compute the new indices using broadcasting
        all_ex_indices = ex_indices[:, None] + shifts[None, :, 0]
        all_ey_indices = ey_indices[:, None] + shifts[None, :, 1]
        all_eth_indices = eth_indices[:, None] + shifts[None, :, 2]

        # Convert these indices to linear indices
        flat_indices = (all_ex_indices * len(self.config.ey_space) * len(self.config.eth_space) +
                        all_ey_indices * len(self.config.eth_space) +
                        all_eth_indices).astype(int)

        # Fetch the closest points
        closest_pts = self.all_pts[flat_indices]

        # Compute Mahalanobis distance and PDF values
        diff = error[:, None, :] - closest_pts
        mahalanobis_dist = np.einsum('nij,jk,nik->ni', diff, self.cov_inv, diff)
        pdf_values = self.const * np.exp(-0.5 * mahalanobis_dist)
        sum_pdf_values = pdf_values.sum(axis=-1, keepdims=True)
        pdf_values /= np.where(sum_pdf_values == 0, 1e-10, sum_pdf_values)
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
    transition_matrix = gpi.compute_transition_matrix()
    print(time.time()-start)
    # transition_matrix = gpi.compute_transition_matrix()

