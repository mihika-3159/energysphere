import cvxpy as cp
import numpy as np

def optimize_investment(expected_roi, risk, budget):
    n = len(expected_roi)
    x = cp.Variable(n)
    objective = cp.Maximize(expected_roi @ x - 0.5 * cp.quad_form(x, np.diag(risk)))
    constraints = [cp.sum(x) <= budget, x >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value
