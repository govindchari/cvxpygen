import cvxpy as cp
import numpy as np
import scipy as sp
import time
import sys
from cvxpygen import cpg

np.random.seed(0)

T = 10
dt = 0.25
n = 4

Q = np.diag(np.random.uniform(0, 10, 2 * n))
R = np.diag(np.random.uniform(0, 10, n))

band = -2 * np.eye(n)
for i in range(1, n):
    band[i, i - 1] = 1
    band[i - 1, i] = 1
Ac = np.block([[np.zeros((n, n)), np.eye(n)], [band, np.zeros((n, n))]])
Bc = np.block([[np.zeros((n, n))], [np.eye(n)]])

A = sp.linalg.expm(Ac * dt)
B = np.linalg.inv(Ac) @ (A - np.eye(2 * n)) @ Bc

u = cp.Variable((n, T))
x = cp.Variable((2 * n, T + 1))

umax = 5
xmax = 2
xlim = xmax * np.ones(2 * n)
ulim = umax * np.ones(n)

x0 = cp.Parameter(2*n, name='x0')
x0.value = np.clip(np.random.randn(2 * n), -0.90 * xlim, 0.90 * xlim)
obj = 0
con = [x[:, 0] == x0]
for k in range(T):
    obj += cp.quad_form(x[:, k], Q) + cp.quad_form(u[:, k], R)
    con += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]
    con += [-xlim <= x[:, k], x[:, k] <= xlim]
    con += [-ulim <= u[:, k], u[:, k] <= ulim]
obj += cp.quad_form(x[:, T], Q)

problem = cp.Problem(cp.Minimize(obj), con)

# cpg.generate_code(problem, code_dir='oscillating_masses', solver='QOCOGEN', wrapper=True)
from oscillating_masses.cpg_solver import cpg_solve
problem.register_solve('CPG', cpg_solve)
x0.value = 0.25 * np.ones(2*n)
t0 = time.time()
val = problem.solve(solver='CLARABEL', verbose=False)
t1 = time.time()
sys.stdout.write('\nCVXPY\nSolve time: %.3f ms\n' % (1000*(t1-t0)))
sys.stdout.write('Objective function value: %.6f\n' % val)
cpx = x.value

# solve problem with C code via python wrapper
t0 = time.time()
val = problem.solve(method='CPG', updated_params=['x0'], verbose=False)
t1 = time.time()
sys.stdout.write('\nCVXPYgen\nSolve time: %.3f ms\n' % (1000 * (t1 - t0)))
sys.stdout.write('Objective function value: %.6f\n' % val)
cpgx = x.value

print(np.linalg.norm(cpx-cpgx))