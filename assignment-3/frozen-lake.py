import gym
from random import random

env = gym.make('FrozenLake-v1')

print(env.action_space)
print(env.observation_space)

# on-policy Monte Carlo control

class Policy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def act(self, state):
        if random() > self.epsilon:
            # choose greedy
            pass
        else:
            # choose random other action
            pass

        return 

    def improve(self):
        pass

def generate_episode(policy):
    states, actions, rewards = [], [], []
    t = 0
    state = env.reset()
    while True:
        action = policy.act(state)
        states.append(state)
        actions.append(action)

        t += 1
        state, reward, done, info = env.step(action)
        rewards.append(reward)

        print(f"time: {t}")
        print(f"reward: {reward}")
        print(f"observation: {state}")

        if done:
            print("Episode finished")
            break

    return states, actions, rewards


policy = Policy()
generate_episode(policy)


env.close()

