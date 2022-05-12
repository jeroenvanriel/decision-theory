import sys, random
from collections import defaultdict


# Example 1.3.2 Inventory control
# A storage depot is used to keep production items in stock. At most 2 items can
# be stored at the same time. At the end of each week, the inventory level (i.e., the
# number of items in stock) is monitored and a decision is made about the number
# of new items to be ordered from the production facility. An order that is placed
# on Friday is delivered on Monday at 7.30 a.m. The cost of an order consist
# of a fixed amount of e100 and an additional e100 per ordered item. Requests
# for items arrive randomly at the storage depot: With probability 14 there is
# no demand during a week, with probability 12 exactly one item is requested
# during a week and with probability 14 the weekly demand equals 2 items. If
# the weekly demand exceeds the inventory stock, it is fulfilled directly from the
# production facility at the expense of e300 per item. The depot manager wishes
# to minimize the expected ordering costs over a pre-determined finite horizon
# planning period. The items in stock at the end of the planning period render
# no value.


# FORMAL PROBLEM DEFINITION

states = ["00", "01", "02", "1", "2"]

def rewards_0(i):
    i = "0"+i
    return {
        (0, i, "00"): 0,
        (0, i, "01"): 300,
        (0, i, "02"): 600,
        (1, i, "1"): 200,
        (1, i, "00"): 200,
        (1, i, "01"): 500,
        (2, i, "2"): 300,
        (2, i, "1"): 300,
        (2, i, "00"): 300,
    }

rewards_1 = {
    (0, "1", "1"): 0,
    (0, "1", "00"): 0,
    (0, "1", "01"): 300,
    (1, "1", "2"): 200,
    (1, "1", "1"): 200,
    (1, "1", "00"): 200,
}

rewards_2 = {
    (0, "2", "2"): 0,
    (0, "2", "1"): 0,
    (0, "2", "00"): 0,
}

rewards = {**rewards_0("0"), **rewards_0("1"), **rewards_0("2"), **rewards_1, **rewards_2}

def transitions_0(i): 
    i = "0"+i
    return {
        (0, i, "00"): 0.25,
        (0, i, "01"): 0.5 ,
        (0, i, "02"): 0.25,
        (1, i, "1"):  0.25,
        (1, i, "00"): 0.5 ,
        (1, i, "01"): 0.25,
        (2, i, "2"):  0.25,
        (2, i, "1"):  0.5 ,
        (2, i, "00"): 0.25,
    }

transitions_1 = {
    (0, "1", "1"):  0.25,
    (0, "1", "00"): 0.5 ,
    (0, "1", "01"): 0.25,
    (1, "1", "2"):  0.25,
    (1, "1", "1"):  0.5 ,
    (1, "1", "00"): 0.25,
}

transitions_2 = {
    (0, "2", "2"):  0.25,
    (0, "2", "1"):  0.5,
    (0, "2", "00"): 0.25,
}

transitions = {**transitions_0("0"), **transitions_0("1"), **transitions_0("2"), **transitions_1, **transitions_2}

def reward(a, i, j):
    return rewards.get((a, i, j), 0)

def transition(a, i, j):
    return transitions.get((a, i, j), 0)

def actions(i):
    return { "00": [0,1,2], "01": [0,1,2], "02": [0,1,2],
             "1": [0,1], "2": [0] }[i]


# CALCULATE OPTIMAL DECISION RULE

V = []
arg = []

# V_0
V.append({s: 0 for s in states})
# -1 encodes "no further action allowed"
arg.append({s: -1 for s in states})

def V_n(i, V_prev):
    minimum = float('inf')
    for a in actions(i):
        s = 0
        for j in states:
            p = transition(a, i, j)
            r = reward(a, i, j)
            s += p * (r + V_prev[j])

        if s < minimum:
            minimum = s
            arg = a

    return minimum, arg

V_new = defaultdict(int)
arg_new = defaultdict(int)
for i in states:
    V_new[i], arg_new[i] = V_n(i, V[0])
V.append(V_new)
arg.append(arg_new)

V_new = defaultdict(int)
arg_new = defaultdict(int)
for i in states:
    V_new[i], arg_new[i] = V_n(i, V[1])
V.append(V_new)
arg.append(arg_new)

for i in range(len(V)):
    print(i)
    for key in V[i].keys():
        k = key
        if key == "00":
            k = "0"
        if key in ["01", "02"]:
            continue
        print(k, V[i][key], arg[i][key])


# SIMULATION

# decision rule determined using the above
f = [{2: 0, 1: 0, 0: 2}, {2: 0, 1: 0, 0: 1}]
initial = 2

N = 1000000
costs = []
for _ in range(N):
    state = initial
    cost = 0
    for n in [0,1]: # timestep
        a = f[n][state]
        if a > 0:
            cost += 100 + 100 * a

        demand = -1
        x = random.random()
        if 0 <= x < 0.25:
            demand = 0
        elif 0.25 <= x < 0.75:
            demand = 1
        elif 0.75 <= x <= 1:
            demand = 2

        if demand > state + a:
            cost += (demand - state - a) * 300

        state = max(0, state + a - demand)

    costs.append(cost)

print(sum(costs)/N)

