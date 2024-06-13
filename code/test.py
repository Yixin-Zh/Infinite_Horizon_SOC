# Initial guess for E and U
import numpy as np
x0_E = np.zeros((3 * self.T, 1))
x0_U = np.zeros((2 * (self.T - 1), 1))
x0 = np.vstack((x0_E, x0_U))

# Lower bounds for E and U
lbx_E = np.full((3 * self.T, 1), -np.inf)
lbx_U = np.tile([0, -1], (self.T - 1, 1)).reshape(-1, 1)
lbx = np.vstack((lbx_E, lbx_U))

# Upper bounds for E and U
ubx_E = np.full((3 * self.T, 1), np.inf)
ubx_U = np.tile([1, 1], (self.T - 1, 1)).reshape(-1, 1)
ubx = np.vstack((ubx_E, ubx_U))

sol = solver(
    x0=x0,  # initial guess
    lbx=lbx,  # lower bounds
    ubx=ubx,  # upper bounds
)
x = sol["x"]  # get the solution
