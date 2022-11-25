import numpy as np

class ReplayBuffer(object):
    def __init__(self, size_buffer,action_size,state_size):
        self.size_buffer = size_buffer
        self.action_size = action_size
        self.state_size = state_size
        self.states = np.zeros((size_buffer,state_size))
        self.actions = np.zeros((size_buffer,action_size))
        self.rewards = np.zeros((size_buffer,1))
        self.next_states = np.zeros((size_buffer,state_size))
        self.dones = np.zeros((size_buffer,1))
        self.current_ind = 0

    def store(self, state, action, reward, next_state, done):
        l1 = [self.states, self.actions, self.rewards, self.next_states, self.dones]
        l2 = [state, action, reward, next_state, done]
        for l,e  in zip(l1,l2):
            l[self.current_ind,:] = e
        self.current_ind+=1
        self.current_ind = self.current_ind%self.size_buffer


    def sample_batch(self,batch_size = 32):
        n = len(self.states)
        l1 = [self.states, self.actions, self.rewards, self.next_states, self.dones]
        res = []
        if n<=batch_size:
            print('states number < BATCH_SIZE')
        else:
            batch_indices = np.random.randint(n,size = batch_size)
            for item in l1:
                res.append(item[batch_indices,:])

        return res

