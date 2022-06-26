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
gamma = 0.9
alpha = 0.5
episodes = 1000
T = 10000 # maximum number of steps in one episode

sma = SMA(100)
episode_sma = np.zeros((episodes))

def update_epsilon(epsilon):
    #return min(epsilon, max(0, 2 * epsilon - (1 / 50) * epsilon * episodes))
    return epsilon - 0.001

policy = EpsilonSoftPolicy(1, env)
for episode in range(episodes):
    total_reward = 0
    s = env.reset()
    for t in range(T):
        a = policy.act(s)
        sn, reward, done, info = env.step(a)
        total_reward += reward

        Q[s,a] = Q[s,a] + alpha * (reward + gamma * np.max(Q[sn,:]) - Q[s,a])

        # update greedy action
        a_star = np.argmax(Q[s,:])
        policy.improve(s, a_star)

        s = sn

        if done:
            sma.put(total_reward)
            episode_sma[episode] = sma.get()
            policy.epsilon = update_epsilon(policy.epsilon)
            print(f'epsilon: {policy.epsilon}, episode: {episode}, sma: {sma.get()}, steps: {t}')
            break
 
x_plot = range(episodes)
plt.plot(x_plot, episode_sma)
plt.title('Success rate for Q-learning')
plt.xlabel('episode')
plt.ylabel('success')
plt.savefig('q-frozen-lake.png', dpi=300, bbox_inches='tight')
plt.show

