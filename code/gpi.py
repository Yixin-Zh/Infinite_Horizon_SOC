from dataclasses import dataclass
import numpy as np
from value_function import ValueFunction
import utils
import os
from scipy.stats import multivariate_normal
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
        self.all_pts = self.get_all_pts()
        self.all_controls = self.get_all_controls()
        self.noise = np.diag([utils.sigma[0], utils.sigma[0], utils.sigma[2]])
        self.T = 100 # period of the reference trajectory
        # TODO: other initialization code

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
        """
        Compute the transition matrix in advance to speed up the GPI algorithm.
        Returns:
            shape: [t, number of points, n_v, n_w, 8, 2]
            Explanation:
                [t, number of points] is the current state
                [n_v, n_w] is the control
                [8, 2] is top 8 points index and probability
        Note: points index refers to the index in self.all_pts
        """
        transition_matrix = np.zeros((self.T, self.all_controls.shape[0], self.all_pts.shape[0], 8, 2))

        for t in range(self.T):
            traj_t = traj(t)
            traj_t_prime = traj(t + 1)
            # make shape [3,] to [N, 3]
            traj_t = np.tile(traj_t, (self.all_pts.shape[0], 1))
            traj_t_prime = np.tile(traj_t_prime, (self.all_pts.shape[0], 1))
            for i, control in enumerate(self.all_controls):
                for j, point in enumerate(self.all_pts):
                    error = point - traj_t
                    next_error = self.error_motion_model(error, traj_t, traj_t_prime, control)
                    top_info = self.get_top_pts(next_error)
                    transition_matrix[t, i, j] = top_info
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
            all inputs are in continuous space
            error: error at time t
            ref_state: reference state at time t
            next_ref_state: reference state at time t+1
            control: control input at time t
        Returns:
            error at time t+1 [3,]
        """
        G = np.array([[np.cos(ref_state[2]+error[2]), 0],
                      [np.sin(ref_state[2]+error[2]), 0],
                      [0, 1]])
        ref_state = np.array(ref_state)
        next_ref_state = np.array(next_ref_state)
        next_error = error + G @ control * utils.time_step + (ref_state - next_ref_state)
        return next_error

    def get_top_pts(self, error):
        """
        Add noise to the error
        Args:
            error: error at time t, deterministic mean value
        Returns:
            error with noise
        """
        mean = error

        p_value = multivariate_normal.pdf(self.all_pts, mean=mean, cov=self.noise)
        p_value = p_value / np.sum(p_value)
        top_idx = np.argsort(p_value)[-8:]
        top_prob = p_value[top_idx]
        top_info = np.concatenate((top_idx[:, None], top_prob[:, None]), axis=1)
        return top_info

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
    points = gpi.get_all_pts()
    top = gpi.get_top_pts(np.array([-2.,-1.6,0.08055366]))
    import time
    start = time.time()
    transition_matrix = gpi.compute_transition_matrix()
    print(time.time()-start)
    # transition_matrix = gpi.compute_transition_matrix()

