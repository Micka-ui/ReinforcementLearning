# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gym
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode="human")
    state = env.reset()
    print(state[0])
    print(env.action_space)
    print(env.observation_space)
    n_actions = 100
    done = False
    for i in range(n_actions):
        while not done:
            env.render()
            action = np.random.randint(2)
            obs, reward, done,info ,_ = env.step(action)
            print(obs,reward,done, info,_) info

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
