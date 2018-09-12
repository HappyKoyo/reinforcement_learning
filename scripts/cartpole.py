# coding:utf-8
import gym # Open AI gym
import numpy as np
import time

# Parameters
MAX_STEPS = 200 # max number of steps every episodes
NUM_DIZITIZED = 5 # number of digitized observing data
NUM_CONSECUTIVE_ITERATIONS = 100 # number of reward storing for ending condition
NUM_EPISODES = 2000 #max iteration episodes
GOAL_AVERAGE_REWARD = 195 #reward reference total value

# Generate vector has 'num' numbers from 'clip_min' to 'clip_max'
def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]
    
# Digitize observed state to discrete values
def digitizeState(observation):
    cart_pos, cart_v, pole_angle, pole_v = observation
    # set Q table and calculate 
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, NUM_DIZITIZED)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, NUM_DIZITIZED)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, NUM_DIZITIZED)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, NUM_DIZITIZED)),
    ]
    return sum([x * (NUM_DIZITIZED**i) for i, x in enumerate(digitized)])

# Select next action using ε-greedy
def selectAction(next_state, episode):
    #ε-greedy
    epsilon = 0.5 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0,1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0,1])
    return next_action

# Update Q table
def updateQTable(q_table, state, action, reward, nex_state):
    gamma = 0.99
    alpha = 0.5
    # left or right
    next_max_q = max(q_table[next_state][0], q_table[next_state][1])
    q_table[state, action] = (1 - alpha) * q_table[state,  action] + alpha * (reward + gamma * next_max_q)
    return q_table

if __name__=='__main__':
    env = gym.make('CartPole-v0') # game mode
    q_table = np.random.uniform(
            low=-1, high=1, size=(NUM_DIZITIZED**4,env.action_space.n))
    total_reward_vec = np.zeros(NUM_CONSECUTIVE_ITERATIONS) # vector has reward of each execution
    final_x = np.zeros((NUM_EPISODES, 1)) # when learning has finished, position is stored by each t=200 execution.
    islearned = False # flag learning has finished
    isrender = False # flag rendering 

    for episode in range(NUM_EPISODES):
        # initialize environment
        observation = env.reset()
        state = digitizeState(observation)
        action = np.argmax(q_table[state])
        episode_reward = 0

        # learning each episode
        for t in range(MAX_STEPS):
            if islearned:
                env.render() # depict cartpole
                time.sleep(0.1)
                #print (observation[0]) # cart position

            # evaluate s_{t+1} and r_{t} from execution the action a_t.
            observation, reward, done, info = env.step(action)
            print reward

            # give reward
            if done: # game over
                if t < 195:
                    reward = -200
                else:
                    reward = 1

            else: # keep standing
                reward = 1

            episode_reward += reward # add reward

            # deside discrete state s_{t+1}, and update Q-function
            next_state = digitizeState(observation)
            q_table = updateQTable(q_table, state, action, reward, next_state)

            # deside next action a_{t+1}
            action = selectAction(next_state, episode) # a_{t+1}

            state = next_state

            # process when program finished
            if done: # game over
                #print('%d Episode finished after %f time steps / mean %f' %
                #      (episode, t + 1, total_reward_vec.mean()))
                total_reward_vec = np.hstack((total_reward_vec[1:],
                                            episode_reward))

                if islearned:
                    final_x[episode, 0] = observation[0]
                break
        
        # success conditions
        if (total_reward_vec.mean() >= GOAL_AVERAGE_REWARD):
            #print('Episode %d train agent successfuly!' % episode)
            islearned = True
            # save success movie
            if isrender == False:
                #if you want to save this movie, please uncomment following line.
                #env = gym.wrappers.Monitor(env, './movie/cartpole-experiment-1')
                isrender = True
                    
    if islearned:
        np.savetxt('final_x.csv',final_x,delimiter=",")

