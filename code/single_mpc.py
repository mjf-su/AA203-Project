import numpy as np
import cvxpy as cp

"""
Compute MPC optimal trajectories for a single vehicle completing a left turn.

State: [x, y, theta (heading)]
Control: [v, phi (steer angle)]
"""
class MPC:
    def __init__(self, xlim, ulim, args):
        self.n = 3 # state size
        self.m = 2 # control size

        self.ro, self.ri = xlim # track limits
        self.vmax, self.pmax = ulim # control limits
        self.N, self.s_goal, self.L = args

    def sim_time(self, T, t0, dt):
        if t0 > 0:
            T = T - t0
        
        assert T % dt == 0
        self.T = T
        self.dt = dt

    def PQR(self, PQR):
        self.P, self.Q, self.R = PQR

    def dynamics(self, s, u, dt):
        """
        Compute discrete dynamics for a particular state and control combination. To be used in inner loop cvxpy optimizations.
        Inputs:
            s : state optimization variable [n,]
            u : control optimization variable [m,]
            dt : discrete time interval
        """

        A = np.eye(self.n) + dt*np.array([[0, 0, -u[0]*np.sin(s[2])],[0, 0, u[0]*np.cos(s[2])], [0, 0, 0]])
        B = dt*np.array([[np.cos(s[2]), 0], [np.sin(s[2]), 0], [np.tan(u[1])/self.L, u[0]*np.sec(u[1])**2/self.L]])

