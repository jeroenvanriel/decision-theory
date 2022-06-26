import gym, torch, cv2
import numpy as np
from collections import deque
from datetime import datetime
from gym.wrappers import TransformObservation, FrameStack, RecordVideo

from gym import logger
logger.set_level(logger.INFO)

# for logging in azure machine learning studio
from azureml.core import Run
run = Run.get_context()

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device {DEV}')

def get_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def preprocess(frame):
    frame = frame[34:-16, :, :] # crop
    frame = cv2.resize(frame, (84, 84))
    frame = frame.mean(-1) # to grayscale
    frame = frame.astype('float') / 255 # scale pixels
    return frame

class FrameSkip(gym.Wrapper):
    def __init__(self, env, frames):
        super().__init__(env)
        self.env = env
        self.frames = frames

    def step(self, action):
        reward = 0
        for _ in range(self.frames):
            s, r, done, info = self.env.step(action)
            reward += r
            if done:
                break

        return s, reward, done, info

env = gym.make('ALE/Breakout-v5')
env.metadata['render.modes'] = env.metadata.get('render_modes', []) # fix a gym bug
env = RecordVideo(env, 'outputs/', episode_trigger=lambda x: x % 200 == 0)
env = TransformObservation(env, preprocess)
env = FrameSkip(env, 4)
env = FrameStack(env, num_stack=4)

from agent import Agent

if __name__ == "__main__":
    EPISODES = 10000
    MEMORY_SIZE = 10000 # steps
    MIN_MEMORY = 64 # steps
    bs = 64
    lr = 5e-5
    gamma = 0.95
    agent = Agent(env, lr, gamma, MEMORY_SIZE, MIN_MEMORY, bs, DEV)
 
    reward_sma = deque(maxlen=100) # simple moving average

    total_step = 0 # number of steps over all episodes
    for episode in range(EPISODES):
        s = env.reset()

        total_reward = 0
        total_loss = 0
        total_max_q = 0
        for t in range(100000): # max number of actual emulation steps
            a = agent.get_action(s)
            s_next, r, done, _ = env.step(a)
            agent.store(s, a, r, s_next, done)

            loss, max_q = agent.train()

            total_loss += loss
            total_max_q += max_q
            total_reward += r

            total_step += 1
            if total_step % 1000 == 0:
                agent.epsilon = max(0.01, agent.epsilon * 0.99)
    
            if done: # end of episode
                agent.update_target_dqn()

                # periodic model saving
                if total_step > MIN_MEMORY and episode % 200 == 0:
                    print('saving models')
                    # torch.save(agent.target_dqn, f'outputs/target-{episode}.pt')
                    torch.save(agent.online_dqn, f'outputs/online-{episode}.pt')
                    print('saved!')

                # debugging
                # print(f'opt: {agent.opt_exp_count}')
                # print(f'mem: {len(agent.memory)}')

                # logging
                reward_sma.append(total_reward)
                avg_max_q = total_max_q / t
                run.log('reward', total_reward)
                run.log('reward_sma', np.mean(reward_sma))
                run.log('loss', total_loss)
                run.log('avg_max_q', avg_max_q)
                run.log('epsilon', agent.epsilon)
                print(f'[{get_time()}] episode: {episode} reward: {int(total_reward)} reward_sma: {np.mean(reward_sma):.3f} loss: {total_loss:.3f} avg_max_q: {avg_max_q:.3f} epsilon: {agent.epsilon:.2f} last_step: {t} total_step: {total_step}')
                break

