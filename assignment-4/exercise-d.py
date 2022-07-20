import cvxpy as cp
import numpy as np
import random as rand

# cost per unit
c = 0.5
# backorder cost
b = 0.75
# holding cost
h = 0.1

def G(x, d):
    if x < d:
        return -0.25 * x + 0.75 * d
    if x >= d:
        return 0.6 * x - 0.1 * d

def generate_scenarios(N, storage):
    scenarios = []
    for _ in range(N):
        U1 = rand.uniform(38,44)
        U2 = rand.uniform(38,44)
        U = rand.uniform(min(U1,U2), max(U1,U2))
        scenarios.append(U - storage)
    return scenarios

def solve(N, storage=0):
    scenarios = generate_scenarios(N, storage)

    x = cp.Variable(1)
    t = cp.Variable(N)
    objective = cp.Minimize(cp.sum(t) / N)
    constraints = [x >= 0]

    for i, dk in enumerate(scenarios):
        constraints.append((c-b) * x - t[i] <= -b * dk)
        constraints.append((c+h) * x - t[i] <= h * dk)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    candidate = x.value
    cost = prob.value
    return candidate, cost

def evaluate_candidate(x, N_prime, storage=0):
    scenarios = generate_scenarios(N_prime, storage)
    elements = []
    for dk in scenarios:
        elements.append(G(x, dk))

    # alpha = 0.05
    za = 1.64

    # 100(1-alpha)% confidence upper bound
    U = np.mean(elements) + za * np.var(elements, ddof=1)
    return U

N = 10000
s = 10 # storage size
print(f'storage size: {s}')
x1, c1 = solve(N)
print(f'candidate: {x1}, cost: {c1}')
x2, c2 = solve(N, storage=s)
print(f'candidate: {x2}, cost: {c2 + s*c}')

N_prime = 1000000
U1 = evaluate_candidate(x1, N_prime)
U2 = evaluate_candidate(x2, N_prime, storage=s) + s*c

print(f'U1={U1}, U2={U2}, difference={U1-U2}')

