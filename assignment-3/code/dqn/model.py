import torch.nn as nn

class DQN1(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            # 4 input frames
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        for i in [0, 2, 4, 7, 9]:
            nn.init.xavier_uniform(self.model[i].weight)

    def forward(self, x):
        return self.model(x)

class DQN2(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            # 4 input frames
            nn.Conv2d(4, 32, 8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.model(x)

