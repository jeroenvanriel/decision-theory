import cvxpy as cp
import numpy as np
import random as rand

def solve(N):
    scenarios = []
    for _ in range(N):
        U1 = rand.uniform(38,44)
        U2 = rand.uniform(38,44)
        U = rand.uniform(min(U1,U2), max(U1,U2))
        scenarios.append(U)

    # cost per unit
    c = 0.5
    # backorder cost
    b = 0.75
    # holding cost
    h = 0.1

    x = cp.Variable(1)
    t = cp.Variable(N)
    objective = cp.Minimize(cp.sum(t))
    constraints = [x >= 0]

    for i, dk in enumerate(scenarios):
        constraints.append((c-b) * x - t[i] <= -b * dk)
        constraints.append((c+h) * x - t[i] <= h * dk)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    print(x.value)

solve(100)
solve(200)
solve(300)
