import gym
import ale_py

env = gym.make("ALE/Breakout-v5", render_mode='human')

print(env.action_space)
print(env.observation_space)

print("Press Enter to continue...")
input()

env.reset()

# Run the game
for t in range(10000):
    # Choose a random action
    action = env.action_space.sample()

    # Take the action, make an observation from environment and obtain reward
    observation, reward, done, info = env.step(action)

    print("At time ", t, " we obtained reward ", reward, " and observed:")
    print(observation)

    if done:
        print("Episode finished")
        break

env.close()
