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
    print(env.action_space.n)
    print(env.observation_space)
    print(env.observation_space.shape[0])




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
