import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from functools import partial
import jax
import jax.numpy as jnp
import sys

"""
Compute MPC optimal trajectories for a single vehicle completing a left turn.

State: [x, y, theta (heading)]
Control: [v, phi (steer angle)]
"""

# All distance units in meters

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`."""

    A, B = jax.jacobian(f, [0, 1])(s, u)
    c = f(s, u) - A @ s - B @ u
    return A, B, c

class MPC:
    def __init__(self, xlim, ulim, args):
        self.ro, self.ri = xlim # track limits
        self.vmax, self.pmax = ulim # control limits
        self.n, self.m, self.N, self.s_goal, self.L = args
        self.eps = 1e-1

    def set_time(self, T, dt):
        self.sim_steps = int(T / dt)
        self.dt = dt

    def PQR(self, PQR):
        self.P, self.Q, self.R = PQR

    def dynamics(self, s, u):
        """
        Compute discrete non-linear dynamics given state and control. To be used in inner loop cvxpy optimizations.
        Inputs:
            s : state optimization variable [x, y, theta]
            u : control optimization variable [v, phi]
            dt : discrete time interval
        """
        x, y, the = s
        v, phi = u

        ds = jnp.array([
            v*jnp.cos(the),
            v*jnp.sin(the),
            v*jnp.tan(phi)/self.L
        ])

        s_next = s + self.dt*ds
        return s_next
    
    def inner_boundary(self, s):
        """Computes d(s) representing the inner track boundary with three composite shapes"""
        circ_rad = 1.5

        center_cir = jnp.array([14, 18]) # xy

        ds = jnp.array([
            jnp.linalg.norm(s[:2] - center_cir) - circ_rad - self.L/2
        ])

        # center_blk = jnp.array([self.ri, self.ri - circ_rad])/2
        # jnp.linalg.norm(s[:2] - center_blk, ord = jnp.inf) - (self.ri - circ_rad) - tol, 
        # jnp.linalg.norm(s[:2] - jnp.flip(center_blk), ord = jnp.inf) - (self.ri - circ_rad) - tol
        return ds
    
    def visualize(self, s, a):
        fig, ax = plt.subplots()
        ax.plot(s[:, 0, 0], s[:, 0, 1], 'r', label = "traj.")
        ax.plot(self.s_goal[0], self.s_goal[1], '*k')
        ax.plot(s[-1, :, 0], s[-1, :, 1], '--', color = 'k')
        # for i in np.arange(self.sim_steps):
        #     plt.plot(s[i, :, 0], s[i, :, 1], '--*')
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.add_patch(plt.Circle((14, 18), 1.5, color = 'b'))
        fig.savefig("./traj.png")

    def do_mpc(self, s_init):
        s0 = cp.Parameter(self.n,); s0.value = s_init

        s_mpc = cp.Variable((self.N+1, self.n)) # mpc variables
        a_mpc = cp.Variable((self.N, self.m))

        s = np.zeros((self.sim_steps, self.N+1, self.n)) # table for all mpc sims
        a = np.zeros((self.sim_steps, self.N, self.m))

        Nscp = 5
        p = 0.2
        for k in np.arange(self.sim_steps): 
            # --------- Warm Start ----------
            s_prev = np.zeros((self.N+1, self.n)); s_prev[0, :] = s0.value # initial position
            a_prev = np.zeros((self.N, self.m)); # a_prev[:, 0] = self.vmax # start moving forward, no turn
            for i in np.arange(self.N):
                s_prev[i+1] = self.dynamics(s_prev[i], a_prev[i]) # initialize a feasible trajectotry
            # ----------           ----------
            for iter in np.arange(Nscp):
                Ad, Bd, cd = affinize(self.dynamics, s_prev[:-1], a_prev) # Dynamics
                Ad, Bd, cd = np.array(Ad), np.array(Bd), np.array(cd)

                Ab, __, cb = affinize(lambda s, __ : self.inner_boundary(s), s_prev, jnp.concatenate((a_prev, a_prev[-1:]))) # Inner boundary (boundary 1)
                Ab, cb = np.array(Ab), np.array(cb)

                cost = cp.quad_form(s_mpc[-1]-self.s_goal, self.P) + cp.sum([cp.quad_form(s_mpc[i]-self.s_goal, self.Q) + cp.quad_form(a_mpc[i], self.R) for i in np.arange(self.N)])
                
                cons = [s_mpc[0] == s0] # IC
                cons += [cp.abs(s_mpc) <= self.ro] # State space
                cons += [cp.abs(a_mpc[:, 0]) <= self.vmax] + [cp.abs(a_mpc[:, 1]) <= self.pmax] # Control space
                cons += [s_mpc[i+1] == Ad[i] @ s_mpc[i] + Bd[i] @ a_mpc[i] + cd[i] for i in np.arange(self.N)] # Dynamics
                cons += [Ab[i] @ s_mpc[i] + cb[i] >= 0 for i in np.arange(self.N+1)]
                cons += [cp.norm_inf(s_mpc - s_prev) <= p] + [cp.abs(a_mpc - a_prev) <= np.array([[1, 0.2]])]

                prob = cp.Problem(cp.Minimize(cost), cons)
                prob.solve(solver = cp.ECOS)

                if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                    s_prev = s_mpc.value
                    a_prev = a_mpc.value
                else:
                    self.visualize(s[:k], a[:k])
                    sys.exit("CVXPY failed at iteration: " + str(k))
                
            s[k] = s_mpc.value
            a[k] = a_mpc.value

            s0.value = np.array(self.dynamics(s0.value, a_mpc.value[0])); print(s0.value)
            if np.max(np.abs(s0.value - self.s_goal)) < self.eps:
                s = s[:k]
                a = a[:k]
                print("Reached solution!")
                break
        self.visualize(s, a)
    
def main():
    n, m = 3, 2
    T, dt = 10, 0.01
    N, s_goal, L = (30, np.array([11, 20, np.pi/2]), 2)
    ro, ri = 26, 20
    vm, pm = 10, np.pi/4
    mpc = MPC((ro, ri), (vm, pm), (n, m, N, s_goal, L))
    mpc.PQR((1e3*np.eye(n), np.eye(n), np.eye(m)))
    mpc.set_time(T, dt)

    mpc.do_mpc(np.array([16, 15.5, np.pi]))

if __name__ == "__main__":
    main() 

        

