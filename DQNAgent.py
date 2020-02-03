

import sys
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential


class DQNAgent:
    
    def __init__(self, state_dim, action_dim):
        
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
        self.memory = deque(maxlen = 10000)
        self.batch_size = 32
        self.train_start = 100
        
        # Model construction
        self.model = self.build_model()
        self.target_model = self.build_model()
        
        # Initialization of the model
        self.update_target_model()
        
        if self.load_model:
            self.model.load_weights("saved_model.h5")
            
    
    def build_model(self):
        
        model = Sequential()
        model.add(Dense(400, input_dim = self.state_dim, activation = 'relu', kernel_initializer = 'he_uniform'))
        model.add(Dense(300, activation = 'relu', kernel_initializer = 'he_uniform'))
        model.add(Dense(200, activation = 'relu', kernel_initializer = 'he_uniform'))
        model.add(Dense(self.action_dim, activation = 'linear', kernel_initializer = 'he_uniform'))
        
        model.summary()
        model.compile(loss = 'mse', optimizer = Adam(lr = self.alpha))
        
        return model
    
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        
        
    def get_action(self, state_):
        
        state = np.reshape(state_, [1, self.state_dim])
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])
            
    
    
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
            
            
        currentQ = self.model.predict(s_batch)
        nextQ = self.target_model.predict(s2_batch)
        
        y = currentQ
        for i in range(self.batch_size):
            if t_batch[i]:
                y[i][a_batch[i]] = r_batch[i]
            else:
                y[i][a_batch[i]] = r_batch[i] + self.gamma * np.amax(nextQ[i])
                
        self.model.fit(s_batch, y, batch_size = self.batch_size, epochs = 1, verbose = 0)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
        