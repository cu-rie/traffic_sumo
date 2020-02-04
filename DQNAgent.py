import numpy as np
import torch
import torch.nn as nn
import random

from collections import deque
from nn.MultiLayerPerceptron import MultiLayerPerceptron as MLP


class DQNAgent(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=[400, 300, 100], hidden_activation='ReLU'):
        super(DQNAgent, self).__init__()
        self.load_model = False

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Hyper-parameters for DQN
        self.gamma = 0.99
        self.alpha = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Replay memory setting
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.train_start = 100

        # Model construction
        self.model = MLP(input_dim=state_dim, output_dim=action_dim, hidden_dim=hidden_dim,
                         hidden_activation=hidden_activation, out_activation=None)

        self.target_model = MLP(input_dim=state_dim, output_dim=action_dim, hidden_dim=hidden_dim,
                                hidden_activation=hidden_activation, out_activation=None)

        # Initialization of the model
        self.update_target_model(self.model, self.target_model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)

    def forward(self, state):
        return self.model(state)

    def get_action(self, state_):
        state = np.reshape(state_, [1, self.state_dim])

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        else:
            q_values = self.model(state)
            return np.argmax(q_values[0])

    def update_target_model(self, source=None, target=None, tau: float = 1.0):
        if source or target is None:
            source = self.model
            target = self.target_model

        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)

    def append_sample(self, state_, action_, reward_, terminal_, next_state_):
        self.memory.append((state_, action_, reward_, terminal_, next_state_))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        batch = random.sample(self.memory, self.batch_size)
        s_batch = np.zeros((self.batch_size, self.state_dim))
        s2_batch = np.zeros((self.batch_size, self.state_dim))
        a_batch, r_batch, t_batch = [], [], []

        for i in range(self.batch_size):
            s_batch[i] = batch[i][0]
            a_batch.append(batch[i][1])
            r_batch.append(batch[i][2])
            t_batch.append(batch[i][3])
            s2_batch[i] = batch[i][4]

        currentQ = self.model(s_batch)
        with torch.no_grad():
            nextQ = self.target_model(s2_batch)

        y = currentQ
        for i in range(self.batch_size):
            if t_batch[i]:
                y[i][a_batch[i]] = r_batch[i]
            else:
                y[i][a_batch[i]] = r_batch[i] + self.gamma * np.amax(nextQ[i])

        # define loss
        loss = torch.nn.functional.mse_loss(s_batch, y)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
