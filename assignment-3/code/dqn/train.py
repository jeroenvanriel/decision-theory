import gym, torch, cv2
import numpy as np
from collections import deque
from datetime import datetime
from gym.wrappers import TransformObservation, FrameStack

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

env = gym.make('ALE/Breakout-v5')
env = TransformObservation(env, preprocess)
# env = FrameStack(env, num_stack=4) # we implement this manually

from agent import Agent

if __name__ == "__main__":
    EPISODES = 10000
    MEMORY_SIZE = 50000 # steps
    MIN_MEMORY = 10000 # steps
    bs = 32
    lr = 0.0001
    gamma = 0.99
    agent = Agent(env, lr, gamma, MEMORY_SIZE, MIN_MEMORY, bs, DEV)
 
    frame_skip = 4

    reward_sma = deque(maxlen=100) # simple moving average

    total_step = 0 # number of steps over all episodes
    for episode in range(EPISODES):
        s_next = env.reset()
        # stack the 'last' 4 frames
        sk = np.stack((s_next, s_next, s_next, s_next))

        total_reward = 0
        total_loss = 0
        total_max_q = 0
        for t in range(100000): # max number of emulation steps

            if t % frame_skip != 0:
                # skip frame and take same action
                s_next, r, done, _ = env.step(a)
                total_reward += r
                continue

            sk = np.stack((s_next, sk[0], sk[1], sk[2]))
            a = agent.get_action(sk)
            s_next, r, done, _ = env.step(a)
            sk_next = np.stack((s_next, sk[0], sk[1], sk[2]))
            agent.store(sk, a, r, sk_next, done)

            loss, max_q = agent.train()

            total_reward += r
            total_loss += loss
            total_max_q += max_q
            total_step += 1
            if total_step % 1000 == 0:
                agent.epsilon = max(0.01, agent.epsilon * 0.99)
    
            if done: # end of episode
                agent.update_target_dqn()

                # periodic model saving
                if total_step > MIN_MEMORY and episode % 50 == 0:
                    print('saving models')
                    torch.save(agent.target_dqn, f'outputs/target-{episode}.pt')
                    torch.save(agent.online_dqn, f'outputs/online-{episode}.pt')
                    print('saved!')

                # logging
                reward_sma.append(total_reward)
                run.log('reward', total_reward)
                run.log('reward_sma', np.mean(reward_sma))
                run.log('loss', total_loss)
                run.log('avg_max_q', total_max_q / t)
                run.log('epsilon', agent.epsilon)
                print(f'[{get_time()}] episode: {episode} reward: {int(total_reward)} reward_sma: {np.mean(reward_sma):.3f} loss: {total_loss:.3f} avg_max_q: {(total_max_q / t):.3f} epsilon: {agent.epsilon:.2f} last_step: {t} total_step: {total_step}')
                break
 
