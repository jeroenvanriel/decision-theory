import gym, torch, cv2, random
import numpy as np
from collections import deque
from datetime import datetime
from gym.wrappers import TransformObservation, FrameStack, RecordVideo

from gym import logger
logger.set_level(logger.INFO)

# for logging in azure machine learning studio
#from azureml.core import Run
#run = Run.get_context()

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
        self.lives = 5

    def step(self, action):
        action = {0:0, 1:2, 2:3}[action] # map to NOPE, LEFT, RIGHT
        reward = 0
        for _ in range(self.frames):
            s, r, done, info = self.env.step(action)
            lives_next = info['lives']
            reward += r
            if done:
                break

        if self.lives != lives_next and not done:
            s, r, done, info = self.env.step(1) # FIRE to restart
            s, r, done, info = self.env.step(1) # FIRE again to prevent hanging
            self.lives = lives_next

        return s, reward, done, info
    
    def reset(self):
        super().reset()
        s, _, _, info = self.env.step(1) # FIRE to start
        self.lives = 5
        return s

def wrap(env):
    env = TransformObservation(env, preprocess)
    env = FrameSkip(env, 2)
    env = FrameStack(env, num_stack=4)
    return env

env = wrap(gym.make('ALE/Breakout-v5'))

# maximum number of emulation steps
# emulator itself stops around 13500
T_MAX = 1000

# evaluation is done in a different environment, with recording enabled
EVAL_INTERVAL = 100
EVAL_EPISODES = 10
env_eval = gym.make('ALE/Breakout-v5')
env_eval.metadata['render.modes'] = env_eval.metadata.get('render_modes', []) # fix a gym bug
vid = RecordVideo(env_eval, 'outputs/', episode_trigger=lambda: True)
env_eval = wrap(vid)

def evaluate(agent, episode):
    epsilon = agent.epsilon
    agent.epsilon = 0

    rewards = []
    for t in range(EVAL_EPISODES):
        if t == 0: # big hack to only record first with the episode number
            vid.episode_id = episode
            vid.episode_trigger = lambda x: True
        else:
            vid.episode_trigger = lambda x: False
        s = env_eval.reset()
        reward = 0
        for _ in range(T_MAX):
            a = agent.get_action(s)
            s, r, done, _ = env_eval.step(a)
            reward += r
            if done:
                break
        rewards.append(reward)

    agent.epsilon = epsilon # reset back
    return np.mean(rewards)

if __name__ == "__main__":
    from agent import Agent
    EPISODES = 10000
    MEMORY_SIZE = 100000 # steps
    MIN_MEMORY = 10000 # steps
    epsilon_step = 1 / EPISODES
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
        for t in range(T_MAX): # max number of actual emulation steps
            a = agent.get_action(s)
            s_next, r, done, info = env.step(a)
            agent.store(s, a, r, s_next, done)
            s = s_next

            loss, max_q = agent.train()

            total_loss += loss
            total_max_q += max_q
            total_reward += r
            total_step += 1
    
            if done: # end of episode
                if total_step >= MIN_MEMORY:
                    agent.epsilon -= epsilon_step
                    agent.update_target_dqn()

                # periodic evaluation and model saving
                if total_step > MIN_MEMORY and episode % EVAL_INTERVAL == 0:
                    torch.save(agent.online_dqn, f'outputs/online-{episode}.pt')
                    # torch.save(agent.target_dqn, f'outputs/target-{episode}.pt')
                    print('model saved!')

                    reward_eval = evaluate(agent, episode)
                    print(f'reward_eval: {reward_eval}')
                    #run.log('reward_eval', reward_eval)

                # logging
                reward_sma.append(total_reward)
                avg_max_q = total_max_q / t
                #run.log('reward', total_reward)
                #run.log('reward_sma', np.mean(reward_sma))
                #run.log('loss', total_loss)
                #run.log('avg_max_q', avg_max_q)
                #run.log('epsilon', agent.epsilon)
                print(f'[{get_time()}] episode: {episode} reward: {int(total_reward)} reward_sma: {np.mean(reward_sma):.3f} loss: {total_loss:.3f} avg_max_q: {avg_max_q:.3f} epsilon: {agent.epsilon:.2f} last_step: {t} total_step: {total_step}')
                break
 
