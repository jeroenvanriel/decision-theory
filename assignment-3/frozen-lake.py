import gym

env = gym.make('FrozenLake-v1')

print(env.action_space)
print(env.observation_space)


env.reset

for t in range(10000):
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)

    print(f"time: {t}")
    print(f"reward: {reward}")
    print(f"observation: {observation}")

    if done:
        print("Episode finished")
        break

env.close()

