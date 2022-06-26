import gym
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from random import random, choice


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

    def change_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

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


env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"])

Q = np.zeros((env.observation_space.n, env.action_space.n))
gamma = 0.95
alpha = 0.5

sma = SMA(1000)

epsilon_all = [0.1]
nriterations = 5000
finalrewards = np.zeros((len(epsilon_all),nriterations))
for x in range(len(epsilon_all)):
    epsilon = epsilon_all[x]
    policy = EpsilonSoftPolicy(epsilon, env)
    n = 0
    T=0
    while T<nriterations:
        #policy.change_epsilon(min(epsilon,max(2*epsilon-(1/50)*epsilon*T,0)))
        G = 0
        s = env.reset()
        a = policy.act(s)
        while True:
            sn, reward, done, info = env.step(a)
            G = gamma * G + reward
            a_star_sn = np.argmax(Q[sn,:])

            Q[s,a] = Q[s,a] + alpha * (reward + gamma * Q[sn,a_star_sn] - Q[s,a])

            # update greedy action
            a_star = np.argmax(Q[s,:])
            policy.improve(s, a_star)

            s = sn

            if done:
                sma.put(G)
                if n >= 1000:
                    n = 0 
                    finalrewards[x][T] = sma.get()
                    print(x,T)
                    T=T+1
                else:
                    n += 1
                break
                       


x_plot=range(nriterations)
plt.plot(x_plot,finalrewards[0])
plt.title("Fraction of runs getting to the goal for Q-learning")
plt.xlabel("time")
plt.ylabel("Fraction of runs getting to the goal")
plt.savefig('Q_frozen_lake_1_epsilon.png', dpi=300, bbox_inches='tight')
plt.show
