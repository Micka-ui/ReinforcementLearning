# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
import numpy as np
from ReplayBuffer import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="human")
    obs = env.reset()
    print(env.action_space)
    print(env.observation_space)
    replayBuffer = ReplayBuffer(size_buffer = 2,action_size=1,state_size=env.observation_space.shape[0])
    print(env.observation_space.shape[0])
    n_actions = 200
    done = False
    for i in range(n_actions):
        while not done:
            env.render()
            action = np.random.randint(2)
            next_obs, reward, done,info ,_ = env.step(action)
            replayBuffer.store(np.array(obs[0]),action,reward,np.array(next_obs[0]),done)

            obs = next_obs
    print(replayBuffer.states)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
