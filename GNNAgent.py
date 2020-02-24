import numpy as np
import torch
import torch.nn as nn
import random

from collections import deque
from nn.GraphLayer import RelationalGraphNetwork
import dgl


class GNNAgent(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=[400, 300, 100], hidden_activation='ReLU'):
        super(GNNAgent, self).__init__()
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
        self.model = RelationalGraphNetwork()
        self.target_model = RelationalGraphNetwork()

        # Initialization of the model
        self.update_target_model(self.model, self.target_model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)

    def forward(self, state):
        return self.model(state)

    def get_action(self, state):
        # state = np.reshape(state_, [1, self.state_dim])

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        else:
            # state = torch.Tensor(state)
            q_values = self.model(state)

            return q_values.argmax().int()

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
        s_batch, s2_batch, a_batch, r_batch, t_batch = [], [], [], [], []

        for i in range(self.batch_size):
            s_batch.append(batch[i][0])
            a_batch.append(batch[i][1])
            r_batch.append(batch[i][2])
            t_batch.append(batch[i][3])
            s2_batch.append(batch[i][4])

        # pytorch interaction
        s_batch = dgl.batch(s_batch)
        s2_batch = dgl.batch(s2_batch)
        a_batch = torch.Tensor(a_batch).int()
        r_batch = torch.Tensor(r_batch)
        t_batch = torch.Tensor(t_batch)

        currentQ = self.model(s_batch)
        with torch.no_grad():
            nextQ = self.target_model(s2_batch)

        state_action_values = currentQ.gather(1, a_batch.long().reshape(-1, 1))

        next_state_values = torch.zeros(self.batch_size, device=r_batch.device)

        for i in range(self.batch_size):
            if t_batch[i]:
                next_state_values[i] = r_batch[i]
            else:
                next_state_values[i] = r_batch[i] + self.gamma * torch.max(nextQ[i])

        # define loss
        loss = (state_action_values - next_state_values).pow(2).sum() / self.batch_size

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().data
