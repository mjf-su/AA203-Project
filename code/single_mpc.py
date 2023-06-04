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

class Circle:
    def __init__(self, x, y, rad):
        self.center = jnp.array([x, y])
        self.rad = rad

class MPC:
    def __init__(self, P, Q, R, N, s_goal, eps = 1e-1):
        self.P = P
        self.Q = Q
        self.R = R
        self.N = N
        self.s_goal = s_goal

        self.eps = eps # tolerance for stopping criterion
        self.v1pos = None
        self.v2pos = None

    def set_time(self, T, dt):
        self.T = T
        self.dt = dt

    def set_dynamics(self, vm, pm, L):
        self.vm = vm
        self.pm = pm
        self.L = L

    def track_circle(self, x, y, rad):
        self.circle = Circle(x, y, rad)

    def dynamics(self, s, u):
        """
        Computes discrete non-linear dynamics for a simple car model.
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
        """
        Computes distance to inner track boundary.
        Inputs:
            s : vehicle state [x, y, theta]
        """

        ds = jnp.array([
            jnp.linalg.norm(s[:2] - self.circle.center) - self.circle.rad - self.L/2
        ])
        return ds
    
    def v1_coll(self, s):
        assert self.v2pos is not None
        tol = 1/10

        dc = jnp.array([
            jnp.linalg.norm(s[:2] - self.v2pos) - self.L - tol
        ])
        return dc
    
    def v2_coll(self, s):
        assert self.v1pos is not None
        tol = 1/10

        dc = jnp.array([
            jnp.linalg.norm(s[:2] - self.v1pos) - self.L - tol
        ])
        return dc
    
    def v1_penalty(self, s):
        ds = jnp.array([
            jnp.linalg.norm(s[:2] - self.v2pos)
        ])
        return ds
    
    def v2_penalty(self, s):
        ds = jnp.array([
            jnp.linalg.norm(s[:2] - self.v1pos)
        ])
        return ds

    def leader(self):
        v1cc = np.array(self.v1pos - self.circle.center)
        v1s1 = 7 - np.clip(self.v1pos[0], 5, 7)
        v1s2 = np.clip(self.v1pos[1], 5, 6.5) - 5
        if v1cc[0] < 0:
            if v1cc[1] < 0:
                v1c = self.circle.rad*np.arctan(v1cc[0] / v1cc[1])
            else:
                v1c = self.circle.rad*np.pi/2
        else:
            v1c = 0

        v2cc = np.array(self.v2pos - self.circle.center)
        v2s1 = 7 - np.clip(self.v2pos[0], 5, 7)
        v2s2 = np.clip(self.v2pos[1], 5, 6.5) - 5
        if v2cc[0] < 0: 
            if v2cc[1] < 0:
                v2c = self.circle.rad*np.arctan(v2cc[0] / v2cc[1])
            else:
                v2c = self.circle.rad*np.pi/2
        else:
            v2c = 0

        if v2s1 + v2c + v2s2 < v1s1 + v1c + v1s2:
            return 1
        else:
            return 2

    def visualize(self, s1, a1, s2, a2):
        """
        Visualize executed MPC path with goal and track boundary.
        """
        fig, ax = plt.subplots()

        # Visualize data
        ax.plot(s1[:, 0, 0], s1[:, 0, 1], 'r', label = "path1")
        ax.plot(s2[:, 0, 0], s2[:, 0, 1], 'g', label = "path2")
        ax.plot(self.s_goal[0], self.s_goal[1], '*k', label = "goal")
        ax.plot(s1[-1, :, 0], s1[-1, :, 1], '--', color = 'k')
        ax.plot(s2[-1, :, 0], s2[-1, :, 1], '--', color = 'k')
        
        # Format plot
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.add_patch(plt.Circle(np.array(self.circle.center), self.circle.rad, color = 'b'))
        ax.legend()

        fig.savefig("./traj.png")
        plt.close()

    def scp(self, s_mpc, a_mpc, s0, v1, ahead, p = 0.2, Nscp = 5):
        solve = True

        # --------- Zero Init. ----------
        s_prev = np.zeros(s_mpc.shape); s_prev[0, :] = s0 # initial position
        a_prev = np.zeros(a_mpc.shape)
        for i in np.arange(self.N):
            s_prev[i+1] = self.dynamics(s_prev[i], a_prev[i]) # initialize a feasible trajectotry
        # ----------           ----------
        
        for _ in np.arange(Nscp):
            Ad, Bd, cd = affinize(self.dynamics, s_prev[:-1], a_prev) # Dynamics
            Ad, Bd, cd = np.array(Ad), np.array(Bd), np.array(cd)

            Ab, __, cb = affinize(lambda s, __ : self.inner_boundary(s), s_prev, jnp.concatenate((a_prev, a_prev[-1:]))) # Track boundary
            Ab, cb = np.array(Ab), np.array(cb)

            Ac, __, cc = affinize(lambda s, __ : self.v1_coll(s), s_prev, jnp.concatenate((a_prev, a_prev[-1:]))) if v1 else affinize(lambda s, __ : self.v2_coll(s), s_prev, jnp.concatenate((a_prev, a_prev[-1:]))) # Collision 
            Ac, cc = np.array(Ac), np.array(cc)

            Ar, __, cr = affinize(lambda s, __ : self.v1_penalty(s), s_prev, jnp.concatenate((a_prev, a_prev[-1:]))) if v1 else affinize(lambda s, __ : self.v2_penalty(s), s_prev, jnp.concatenate((a_prev, a_prev[-1:])))
            Ar, cr = np.array(Ar), np.array(cr)

            cost = cp.quad_form(s_mpc[-1]-self.s_goal, self.P) + cp.sum([cp.quad_form(s_mpc[i]-self.s_goal, self.Q) + cp.quad_form(a_mpc[i], self.R) for i in np.arange(self.N)])
            
            cons = [s_mpc[0] == s0] # IC
            cons += [cp.abs(a_mpc[:, 0]) <= self.vm] + [cp.abs(a_mpc[:, 1]) <= self.pm] # Control space
            cons += [s_mpc[i+1] == Ad[i] @ s_mpc[i] + Bd[i] @ a_mpc[i] + cd[i] for i in np.arange(self.N)] # Dynamics
            cons += [Ab[i] @ s_mpc[i] + cb[i] >= 0 for i in np.arange(self.N+1)] # Track limtis 
            cons += [Ac[i] @ s_mpc[i] + cc[i] >= 0 for i in np.arange(self.N+1)] # Collision Constraint
            cons += [cp.norm_inf(s_mpc - s_prev) <= p] + [cp.abs(a_mpc - a_prev) <= np.array([[1, 0.2]])] # Trust regions

            # Encourage increasing lead, decreasing trailing distance
            if ahead:
                cost -= 5e1*cp.sum([Ar[i] @ s_mpc[i] + cr[i] for i in np.arange(self.N+1)])
            else:
                cost += 2.5e2*cp.sum([Ar[i] @ s_mpc[i] + cr[i] for i in np.arange(self.N+1)]) 

            prob = cp.Problem(cp.Minimize(cost), cons)
            prob.solve(solver = cp.SCS)

            if prob.status == "optimal" or prob.status == "optimal_inaccurate":
                s_prev = s_mpc.value
                a_prev = a_mpc.value
            else:
                solve = False
                break
        
        return s_mpc.value, a_mpc.value, solve

    def do_mpc(self, s1_init, s2_init):
        n = self.P.shape[0]
        m = self.R.shape[0]
        sim_steps = int(self.T / self.dt)

        s01 = cp.Parameter(n,); s01.value = s1_init
        s02 = cp.Parameter(n,); s02.value = s2_init

        s_mpc = cp.Variable((self.N+1, n)) # mpc variables
        a_mpc = cp.Variable((self.N, m))

        s1 = np.zeros((sim_steps, self.N+1, n)) # car 1: table for all mpc sims
        a1 = np.zeros((sim_steps, self.N, m))
        s2 = np.zeros(s1.shape) # car 2: table for all mpc sims
        a2 = np.zeros(a1.shape)

        for k in np.arange(sim_steps): 
            if np.max(np.abs(s01.value - self.s_goal)) < self.eps or np.max(np.abs(s02.value - self.s_goal)) < self.eps:
                s1 = s1[:k]
                a1 = a1[:k]
                s2 = s2[:k]
                a2 = a2[:k]

                print("Reached solution!")
                break

            self.v1pos = s01.value[:2] # store xy pos for collision check
            self.v2pos = s02.value[:2] 
            leader = self.leader()

            s1[k], a1[k], solve1 = self.scp(
                s_mpc, 
                a_mpc, 
                s01.value,
                v1 = True,
                ahead = leader == 1) # rollout vehicle 1
            s2[k], a2[k], solve2 = self.scp(
                s_mpc, 
                a_mpc, 
                s02.value,
                v1 = False,
                ahead = leader == 2) # rollout vehicle 2

            s01.value = np.array(self.dynamics(s01.value, a1[k, 0])) if solve1 else s01.value # execute dynamics
            s02.value = np.array(self.dynamics(s02.value, a2[k, 0])) if solve2 else s02.value
            if k > 0:
                self.visualize(s1[:k], a1[:k], s2[:k], a2[:k])

            if not solve1 or not solve2:
                print("CVXPY failed: infeasible solution at step " + str(k))
                break
            elif np.linalg.norm(s01.value - s02.value) < self.L: # Collision!
                s1 = s1[:k]
                a1 = a1[:k]
                s2 = s2[:k]
                a2 = a2[:k]

                print("Vehicle Collision at step " + str(k))
                break
            else:
                print("Vehicle1: ", s01.value, " Vehicle2: ", s02.value)
        self.visualize(s1, a1, s2, a2)
    
def main():
    n = 3 # state, control dimension 
    m = 2
    P = 2e4*np.eye(n) # cost-to-go approx.
    Q = np.eye(n) # state stage cost
    R = np.eye(m) # control stage cost

    N = 30
    T = 10
    s_goal = np.array([2.75, 6, np.pi/2]) 
    s1_init = np.array([6, 3.125, np.pi])
    s2_init = np.array([6, 2.5, np.pi])

    L = 1/2 # car width (approx. as a circle)
    vm = 10 # maximum velocity
    pm = np.pi/4 # maximum steering angle
    
    mpc = MPC(P, Q, R, N, s_goal)
    
    # Configure optimization
    mpc.set_time(T, dt = 0.01)
    mpc.set_dynamics(vm, pm, L)
    mpc.track_circle(5, 5, 1.5)

    # Execute
    mpc.do_mpc(s1_init, s2_init)

if __name__ == "__main__":
    main() 

        

