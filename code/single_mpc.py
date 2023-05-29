import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from functools import partial
import jax
import jax.numpy as jnp

"""
Compute MPC optimal trajectories for a single vehicle completing a left turn.

State: [x, y, theta (heading)]
Control: [v, phi (steer angle)]
"""

@partial(jax.jit, static_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0, 0))
def affinize(f, s, u):
    """Affinize the function `f(s, u)` around `(s, u)`."""
    # PART (b) ################################################################
    # INSTRUCTIONS: Use JAX to affinize `f` around `(s,u)` in two lines.

    A, B = jax.jacobian(f, [0, 1])(s, u)
    c = f(s, u) - A @ s - B @ u
    # END PART (b) ############################################################
    return A, B, c

class MPC:
    def __init__(self, xlim, ulim, args):
        self.ro, self.ri = xlim # track limits
        self.vmax, self.pmax = ulim # control limits
        self.n, self.m, self.N, self.s_goal, self.L, self.dt = args

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
    
    def visualize(self, s, a):
        plt.figure()
        plt.plot(s[:, 0, 0], s[:, 0, 1], 'r', label = "traj.")
        for i in np.arange(self.sim_steps):
            plt.plot(s[i, :, 0], s[i, :, 1], '--*', label = "ol. traj", color = 'k')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("./traj.png")

    def do_mpc(self, s_init):
        s0 = cp.Parameter(self.n,); s0.value = s_init

        s_mpc = cp.Variable((self.N+1, self.n))
        a_mpc = cp.Variable((self.N, self.m))

        s_prev = np.zeros((self.N+1, self.n))
        a_prev = np.zeros((self.N, self.m))

        s = np.zeros((self.sim_steps, self.N+1, self.n))
        a = np.zeros((self.sim_steps, self.N, self.m))

        for k in np.arange(self.sim_steps):
            Ad, Bd, cd = affinize(self.dynamics, s_prev[:-1], a_prev)
            Ad, Bd, cd = np.array(Ad), np.array(Bd), np.array(cd)
            cost = cp.quad_form(s_mpc[-1]-self.s_goal, self.P) + cp.sum([cp.quad_form(s_mpc[i]-self.s_goal, self.Q) + cp.quad_form(a_mpc[i], self.R) for i in np.arange(self.N)])
            cons = [s_mpc[0] == s0
            ] + [s_mpc[i+1] == Ad[i] @ s_mpc[i] + Bd[i] @ a_mpc[i] + cd[i] for i in np.arange(self.N)
            ] + [s_mpc[:, 0] <= self.ri, 0 <= s_mpc[:, 0], s_mpc[:, 1] <= self.ri, 0 <= s_mpc[:, 1] 
            ] + [cp.norm(a_mpc[:, 0], "inf") <= self.vmax, cp.norm(a_mpc[:, 1], "inf") <= self.pmax]

            prob = cp.Problem(cp.Minimize(cost), cons)
            prob.solve()

            if prob.status == "Optimal":
                s_prev = s_mpc.value
                a_prev = a_mpc.value
                
                s[k] = s_mpc.value
                a[k] = a_mpc.value

                s0.value = self.dynamics(s0.value, a_mpc.value[0])
            else:
                print("CVXPY failed at iteration: ", k)
                break
        #self.visualize(s, a)
    
def main():
    n, m = 3, 2
    T, dt = 15, 0.1
    N, s_goal, L = (20, np.array([0, 23, np.pi]), 2)
    ro, ri = 26, 20
    vm, pm = 10, np.pi/2 - 1e-2
    mpc = MPC((ro, ri), (vm, pm), (n, m, N, s_goal, L, dt))
    mpc.PQR((np.eye(n), np.eye(n), np.eye(m)))
    mpc.set_time(T, dt)

    mpc.do_mpc(np.array([23, 10, np.pi/2]))

if __name__ == "__main__":
    main() 

        

