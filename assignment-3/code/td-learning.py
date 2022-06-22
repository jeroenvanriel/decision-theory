import gym
import numpy as np
from queue import Queue
from random import random, choice
from sma import SMA


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


env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"])

Q = np.zeros((env.observation_space.n, env.action_space.n))

epsilon = 0.1
policy = EpsilonSoftPolicy(epsilon, env)
gamma = 0.95
alpha = 0.5

sma = SMA(1000)

# sarsa: on-policy TD control

n = 0
while True:
    G = 0
    s = env.reset()
    a = policy.act(s)
    while True:
        sn, reward, done, info = env.step(a)
        G = gamma * G + reward

        an = policy.act(sn)

        Q[s,a] = Q[s,a] + alpha * (reward + gamma * Q[sn,an] - Q[s,a])

        # update greedy action
        a_star = np.argmax(Q[s,:])
        policy.improve(s, a_star)

        s = sn
        a = an

        if done:
            sma.put(G)
            if n >= 1000:
                n = 0 
                print(sma.get())
            else:
                n += 1
            break
