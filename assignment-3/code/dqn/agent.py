import torch
import random
import numpy as np
from torch.optim import Adam
from torch.nn.functional import mse_loss
from collections import deque

from model import DQN1

class Agent:
    def __init__(self, env, lr, gamma, memory_size, min_memory, batch_size, device):
        self.epsilon = 1

        self.dev = device
        self.online_dqn = DQN1(env.action_space.n).to(device).train()
        self.target_dqn = DQN1(env.action_space.n).to(device)
        self.target_dqn.load_state_dict(self.online_dqn.state_dict())
        self.target_dqn.eval()

        self.gamma = gamma

        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.min_memory = min_memory

        self.optimizer = Adam(self.online_dqn.parameters(), lr=lr)
    
    def store(self, s, a, r, s_next, done):
        self.memory.append([s, a, r, s_next, done])
    
    def get_action(self, state):
        if random.random() < self.epsilon:
            # choose random action
            return random.choice(range(self.online_dqn.n_actions))
        else:
            # return greedy action
            x = torch.unsqueeze(torch.from_numpy(np.asarray(state)), dim=0).float().to(self.dev)
            return torch.argmax(self.online_dqn(x))
    
    def train(self):
        if len(self.memory) < self.min_memory:
            return 0, 0
        
        # sample minibatch
        s_b, a_b, r_b, s_next_b, done_b = zip(*random.sample(self.memory, self.batch_size))

        s_b = torch.from_numpy(np.asarray(s_b)).float().to(self.dev)
        a_b = torch.tensor(a_b, dtype=torch.long, device=self.dev)
        r_b = torch.tensor(r_b, dtype=torch.float, device=self.dev)
        s_next_b = torch.from_numpy(np.asarray(s_next_b)).float().to(self.dev)
        done_b = torch.tensor(done_b, dtype=torch.float, device=self.dev)


        def q_for_action(qs, actions):
            """Get the q values from (64, 4) corresponding to the action."""
            return qs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # get q values for current states
        q_vals = self.online_dqn(s_b)
        # actual q values for selected actions
        actual_q_vals = q_for_action(q_vals, a_b)

        # target q values
        next_q_vals = self.online_dqn(s_next_b) # shape: N x 4
        next_actions = next_q_vals.max(1)[1]
        next_target_q_vals = q_for_action(self.target_dqn(s_next_b), next_actions)

        # double Q learning target
        target = r_b + self.gamma * next_target_q_vals * (1 - done_b)

        self.optimizer.zero_grad()
        loss = mse_loss(actual_q_vals, target)
        loss.backward()
        self.optimizer.step()   

        return loss.item(), q_vals.max().item()
    
    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.online_dqn.state_dict())
        self.target_dqn.eval()

