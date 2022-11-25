# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
import numpy as np
from SimpleDQNAgent import *



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    n_episodes = 3
    batch_size = 32
    epsilon = 1.0
    episode_start = 1
    epsilon_decay = 0.99
    gamma = 0.95
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss_fn = tf.keras.losses.MeanSquaredError()
    agent = SimpleDQNAgent(buffer_size=5000,n_actions=1,action_size=action_size,state_size=state_size,optimizer=optimizer,\
                           loss_fn=loss_fn,gamma=gamma)


    done = False
    score_history = []
    for episode in range(n_episodes):
        state = env.reset()[0]
        score = 0
        for step in range(200):
            epsilon = max(epsilon*epsilon_decay,0.05)
            action = agent.epsilon_greedy(state,epsilon)
            next_state, reward, done,_,_ = env.step(action)
            agent.remember((state,action,reward,next_state,done))
            score+=reward
            if done:
                break
            else:
                state = next_state
            if episode>=episode_start:
                agent.learn(batch_size)
        score_history.append(score)
        if episode%10==0:
            print('Average score over last 100 episodes : %.2f'%(np.mean(score_history[-100:])))






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
