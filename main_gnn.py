##### Importing Modules #####
# %matplotlib inline

from TrafficEnv import *
from GNNAgent import GNNAgent

import numpy as np
import matplotlib.pyplot as plt
from util.graphutil import state2graphfunc

# import pandas as pd
################################


##### Experiment Setup #######

MAX_EPISODES = 500
MAX_EP_STEPS = 40

STATE_DIM = 4
ACTION_DIM = 4

avgRange = 10
rewards = []
losss = []
episode_end = []

################################


env = TrafficEnv('')
agent = GNNAgent(STATE_DIM, ACTION_DIM)

for episode in range(MAX_EPISODES):

    env.startSUMO()
    currentState = state2graphfunc(env.reset())

    accumulatedRewards = 0.0
    accumulatedLoss = 0.0

    for t in range(MAX_EP_STEPS):

        currentAction = agent.get_action(currentState)
        nextState, reward, terminal = env.step(currentAction)
        nextState = state2graphfunc(nextState)

        agent.append_sample(currentState, currentAction, reward, terminal, nextState)

        if len(agent.memory) >= agent.train_start:
            loss = agent.train_model()
            accumulatedLoss += loss

        currentState = nextState
        accumulatedRewards += reward

        if terminal or t == (MAX_EP_STEPS - 1):
            agent.update_target_model()
            rewards.append(accumulatedRewards)
            episode_end.append(t)
            losss.append(accumulatedLoss)

            env.endSUMO()
            env.timestep = -1
            print("[Episode : {}]  Rewards : {:.4f}, Epsilon : {:.3f} timestep : {} Loss: {:.3f}"
                  .format(episode, accumulatedRewards, agent.epsilon, t, accumulatedLoss))
            break

    if len(rewards) > 51 and np.mean(rewards[(len(rewards) - min(50, len(rewards))): len(rewards)]) > 19:
        name = "saved_model_" + str(episode)
        agent.model.save_weights(name=name)

agent.model.save_weights(name='saved_model_full.h5')

##### Draw the graph to test the convergence #####
smoothedRewards = np.copy(rewards)
for i in range(avgRange, episode):
    smoothedRewards[i] = np.mean(rewards[i - avgRange: i + 1])

plt.figure(1)
plt.plot(smoothedRewards, label='Smoothed rewards', linewidth=2.2)
plt.plot(rewards, label='episodic rewards', alpha=0.2)
plt.xlabel('Episodes')
plt.ylabel('Accumulated Rewards')
plt.legend()
plt.savefig('result_GNN.png')
##################################################


##### Export the Total episodic rewards #####
import csv

f = open('saved_model/gnn/rewards.csv', 'w')
for i in range(len(rewards)):
    f.write(str(rewards[i]))
    f.write(str("\n"))
f.close()
##############################################
