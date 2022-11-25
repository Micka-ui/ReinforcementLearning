import numpy as np
import tensorflow as tf

class ReplayBuffer():
    def __init__(self, size_buffer):
        self.size_buffer = size_buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def store(self, state, action, reward, next_state, done):
        ind = len(self.states)
        l1 = [self.states, self.actions, self.rewards, self.next_states, self.dones]
        l2 = [state, action, reward, next_state, done]
        if ind + 1 >= self.size_buffer:
            new_ind = (ind + 1) % self.size_buffer
            for l, e in zip(l1, l2):
                l[new_ind] = e
        else:
            for l, e in zip(l1, l2):
                l.append(e)

    def sample_batch(self,batch_size = 32):
        n = len(self.states)
        l1 = [self.states, self.actions, self.rewards, self.next_states, self.dones]
        if n<=batch_size:
            print('states number < BATCH_SIZE')
        else:
            batch_indices = np.random.randint(n,size = batch_size)
            res = list(map(lambda x : np.array(x[batch_indices]),l1))
        return res

