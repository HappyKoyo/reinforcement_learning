#!/usr/bin/env python
# -*- coding: utf-8 -*
import gym
import time
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers  # for saving image
from keras import backend as K
 
DQN_MODE = 1    # 0:DDQN 1:DQN
LENDER_MODE = 1 # 0:not lender 1:lender
 
NUM_EPISODES = 299  # number of episodes
MAX_STEPS = 200  # number of steps each episode
GOAL_MEAN_REWARD = 195
CONSECUTIVE_ITERATIONS = 10  # number of trials to calculate the mean value
# ---
GAMMA = 0.99            # discount factor
HIDDEN_SIZE = 16        # number of hidden layer neuron 
LEARNING_RATE = 0.00001 # learning rate of q-network
MEMORY_SIZE = 10000     # buffer memory size
BATCH_SIZE = 32         

# Object function (huber loss)
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return K.mean(loss)
 
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
 
    #Experience Replay
    def replay(self, memory, batch_size, gamma, targetqn):
        inputs = np.zeros((batch_size, 4))
        targets = np.zeros((batch_size, 2))
        mini_batch = memory.sample(batch_size)
 
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b
 
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]
                
            targets[i] = self.model.predict(state_b)    # output of q-network
            targets[i][action_b] = target               # training data
            self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochs is iteration of data 
 
# Experience Memory
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
 
    def add(self, experience):
        self.buffer.append(experience)
 
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
 
    def len(self):
        return len(self.buffer)

# select action
class Actor:
    def get_action(self, state, episode, targetQN):   # [C]ｔ＋１での行動を返す
        # ε-greedy
        epsilon = 0.001 + 0.9 / (1.0+episode)
 
        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = targetQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # action to get maximum reward
 
        else:
            action = np.random.choice([0, 1])  # action randomly
 
        return action

 
if __name__=='__main__':
    env = gym.make('CartPole-v0')
    islearned = False  # flag learning has finished
    isrender = False  # flag rendering
    total_reward_vec = np.zeros(CONSECUTIVE_ITERATIONS)  # reward reference total value
    mainQN = QNetwork(hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE)     # main q-network
    targetqn = QNetwork(hidden_size=HIDDEN_SIZE, learning_rate=LEARNING_RATE)   # target q-network
    # plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # visualization of network 
    memory = Memory(max_size=MEMORY_SIZE)
    actor = Actor()
     
    for episode in range(NUM_EPISODES):
        env.reset()  # initialize cartpole
        state, reward, done, _ = env.step(env.action_space.sample())  # select rondomly in first step
        state = np.reshape(state, [1, 4])   
        episode_reward = 0

        targetQN = mainQN   

        for t in range(MAX_STEPS + 1):   
            if (islearned == True) and LENDER_MODE:  # when learning has finished, render cartpole
                env.render()
                time.sleep(0.1)
                print(state[0, 0])  

            action = actor.get_action(state, episode, mainQN)   # select action 
            next_state, reward, done, info = env.step(action)   # calculate s_{t+1},_R{t}
            next_state = np.reshape(next_state, [1, 4])

            # give a every action reward
            if done: # game finished
                next_state = np.zeros(state.shape)
                if t < 195:
                    reward = -1
                else:
                    reward = 1
            else:
                reward = 0  # every iteration while game continue

            episode_reward += 1 # reward while 1 episode  

            memory.add((state, action, reward, next_state)) # save this experience to memory
            state = next_state

            # learn to network
            if (memory.len() > BATCH_SIZE) and not islearned:
                mainQN.replay(memory, BATCH_SIZE, GAMMA, targetQN)

            if DQN_MODE: # if DQN is selected
                targetQN = mainQN  

            # game finished
            if done:
                total_reward_vec = np.hstack((total_reward_vec[1:], episode_reward))  # save reward
                print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, total_reward_vec.mean()))
                break

        # if reward mean is bigger than GOAL_MEAN_REWARD, finish learning
        if total_reward_vec.mean() >= GOAL_MEAN_REWARD:
            print('Episode %d train agent successfuly!' % episode)
            islearned = True
            if isrender == False:   # 学習済みフラグを更新
                isrender = True

                # env = wrappers.Monitor(env, './movie/cartpoleDDQN')  # if you need to save cartpole video
