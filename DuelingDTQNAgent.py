from ReplayBuffer import *
from DuelingQNetwork import *
import sys

class DuelingDTQNAgent(object):
    def __init__(self,buffer_size,n_actions,action_size,state_size,optimizer,loss_fn,gamma):
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.n_actions = n_actions
        self.state_size = state_size
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.Qnetwork = DuelingQNetwork(action_size,state_size)


        self.targetQnetwork = self.Qnetwork = DuelingQNetwork(action_size,state_size)
        self.targetQnetwork.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.targetQnetwork.set_weights(self.Qnetwork.get_weights())
        self.memory = ReplayBuffer(size_buffer=buffer_size,action_size=n_actions,state_size=state_size)

    def epsilon_greedy(self,state,epsilon):
            rand = np.random.rand()
            if rand<epsilon:
                action = np.random.randint(self.action_size)
            else:
                Q_values = self.Qnetwork(state[np.newaxis],training = False).numpy()
                action = np.argmax(Q_values[0])
            return action

    def remember(self,exp):
        state,action,reward,next_state,done = exp
        self.memory.store(state,action,reward,next_state,done)

    def learn(self,batch_size):
        batch = self.memory.sample_batch(batch_size)
        states,actions,rewards,next_states,dones = batch
        next_Q_values = self.Qnetwork(next_states,training=False).numpy()
        best_next_actions = np.argmax(next_Q_values,axis=1,keepdims=True)
        next_mask = tf.one_hot(best_next_actions.squeeze(), self.action_size)
        max_next_Q_values = tf.reduce_sum(self.targetQnetwork(next_states)*next_mask,axis=1,keepdims=True)
        target_Q_values = (rewards + (1-dones)*self.gamma*max_next_Q_values)
        mask = tf.one_hot(actions.squeeze(),self.action_size)
        with tf.GradientTape() as tape:
            all_Q_values = self.Qnetwork(states,training = True)
            Q_values = tf.reduce_sum(all_Q_values*mask,axis=1,keepdims=True)
            loss = self.loss_fn(Q_values,target_Q_values)
        gradients = tape.gradient(loss,self.Qnetwork.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.Qnetwork.trainable_variables))