import gym
import numpy as np
from queue import Queue
from random import random, choice
from collections import defaultdict

env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"])


class EpsilonSoftPolicy:
    def __init__(self, epsilon, env):
        self.epsilon = epsilon
        self.obs_space = range(env.observation_space.n)
        self.action_space = range(env.action_space.n)
        self.actions = {
                state : env.action_space.sample()
                for state in self.obs_space
        }

    def act(self, state):
        if random() < self.epsilon:
            # choose random action
            return choice(self.action_space)
        else:
            # choose greedy action
            return self.actions[state]

    def improve(self, state, a_star):
        self.actions[state] = a_star

    def print(self):
        print(self.actions)


def generate_episode(policy):
    state_actions, rewards = [], []
    t = 0
    state = env.reset()
    while True:
        action = policy.act(state)
        state_actions.append((state, action))

        t += 1
        state, reward, done, info = env.step(action)
        rewards.append(reward)

        if done:
            break

    return state_actions, rewards

class SMA:
    """Simple moving average with incremental update."""
    def __init__(self, k):
        self.q = Queue(k)
        self.k = k
        self.SMA = 0

    def put(self, p):
        if self.q.full():
            self.SMA += (p - self.q.get()) / self.k
        else:
            self.SMA += p / self.k
        self.q.put(p)

    def get(self):
        return self.SMA

# on-policy Monte Carlo control

epsilon = 0.1
gamma = 0.95
policy = EpsilonSoftPolicy(epsilon, env)
sma = SMA(1000)

Q = np.empty((env.observation_space.n, env.action_space.n))
N = np.ones((env.observation_space.n, env.action_space.n))

n = 0
while True:
    sa, rewards = generate_episode(policy)
    G = 0
    for t in range(len(sa) - 1, -1, -1):
        G = gamma * G + rewards[t]
        if sa[t] not in sa[0:t]:
            # incremental update of state action value
            Q[sa[t]] = Q[sa[t]] + (G - Q[sa[t]]) / N[sa[t]]
            N[sa[t]] += 1

            # improvement step
            state, action = sa[t]
            a_star = np.argmax(Q[state,:])
            policy.improve(state, a_star)

    sma.put(G)
    if n >= 1000:
        n = 0
        print(sma.get())
    else:
        n += 1

env.close()

