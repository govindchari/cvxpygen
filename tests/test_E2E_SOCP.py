
import pytest
import cvxpy as cp
import numpy as np
import glob
import os
import importlib
import itertools
import pickle
import sys
sys.path.append('../')
import cvxpygen as cpg


def ADP_problem():

    # define dimensions
    n, m = 6, 3

    # define variables
    u = cp.Variable(m, name='u')

    # define parameters
    Rsqrt = cp.Parameter((m, m), name='Rsqrt', diag=True)
    f = cp.Parameter(n, name='f')
    G = cp.Parameter((n, m), name='G')

    # define objective
    objective = cp.Minimize(cp.sum_squares(f + G @ u) + cp.sum_squares(Rsqrt @ u))

    # define constraints
    constraints = [cp.norm(u, 2) <= 1]

    # define problem
    return cp.Problem(objective, constraints)


def assign_data(prob, name, seed):

    np.random.seed(seed)

    if name == 'ADP':

        def dynamics(x):
            # continuous-time dynmaics
            A_cont = np.array([[0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1],
                               [0, 0, 0, -x[3], 0, 0],
                               [0, 0, 0, 0, -x[4], 0],
                               [0, 0, 0, 0, 0, -x[5]]])
            mass = 1
            B_cont = np.concatenate((np.zeros((3, 3)),
                                     (1 / mass) * np.diag(x[3:])), axis=0)
            # discrete-time dynamics
            td = 0.1
            return np.eye(6) + td * A_cont, td * B_cont

        state = -2*np.ones(6) + 4*np.random.rand(6)
        Psqrt = np.eye(6)
        A, B = dynamics(state)
        prob.param_dict['Rsqrt'].value = np.sqrt(0.1) * np.eye(3)
        prob.param_dict['f'].value = np.matmul(Psqrt, np.matmul(A, state))
        prob.param_dict['G'].value = np.matmul(Psqrt, B)

    return prob


N_RAND = 10

name_solver_style_seed = [['ADP'],
                          ['ECOS'],
                          ['explicit', 'implicit'],
                          list(np.arange(N_RAND))]

name_to_prob = {'ADP': ADP_problem()}


@pytest.mark.parametrize('name, solver, style, seed', list(itertools.product(*name_solver_style_seed)))
def test(name, solver, style, seed):

    prob = name_to_prob[name]

    if seed == 0:
        if style == 'explicit':
            cpg.generate_code(prob, code_dir='test_%s_%s_explicit' % (name, solver), solver=solver, explicit=True,
                              problem_name='%s_%s_ex' % (name, solver))
            assert len(glob.glob(os.path.join('test_%s_%s_explicit' % (name, solver), 'cpg_module.*'))) > 0
        if style == 'implicit':
            cpg.generate_code(prob, code_dir='test_%s_%s_implicit' % (name, solver), solver=solver, explicit=False,
                              problem_name='%s_%s_im' % (name, solver))
            assert len(glob.glob(os.path.join('test_%s_%s_implicit' % (name, solver), 'cpg_module.*'))) > 0

    with open('test_%s_%s_%s/problem.pickle' % (name, solver, style), 'rb') as f:
        prob = pickle.load(f)

    module = importlib.import_module('test_%s_%s_%s.cpg_solver' % (name, solver, style))
    prob.register_solve('CPG', module.cpg_solve)

    prob = assign_data(prob, name, seed)

    val_py = prob.solve()
    val_ex = prob.solve(method='CPG')

    if not np.isinf(val_py):
        assert abs((val_ex - val_py) / val_py) < 0.1
