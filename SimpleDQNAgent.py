import numpy as np
import tensorflow as tf
import os
import sys

class SimpleDQNAgent(object):
    def __init__(self,buffer_size,action_size,state_size,optimizer,loss_fn):

        self.buffer_size = buffer_size
        self.action_size = action_size
        self.state_size = state_size
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.Qnetwork = tf.keras.Sequential([
                        tf.keras.layers.Dense(128,activation = 'elu',input_shape=[self.state_size],
                        tf.keras.layers.Dense(128,activation = 'elu'),
                        tf.keras.layers.Dense(self.action_size)]
        )

    def epsilon_greedy(self,state,epsilon):
            rand = np.random.randint()
            if rand<epsilon:
                action = np.random.randint(action_size)
            else:
                Q_values = self.Qnetwork.predict((states[np.newaxis]))
                action = np.argmax(Q_values)
            return action


