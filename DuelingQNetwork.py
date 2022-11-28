import tensorflow as tf

class DuelingQNetwork(tf.keras.Model):

    def __init__(self,actions_size,state_size):
        super(DuelingQNetwork,self).__init__()
        self.action_size = actions_size
        self.state_size = state_size

        self.value_stream = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='elu', input_shape=[self.state_size]),
            tf.keras.layers.Dense(1)
        ])

        self.advantage = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='elu', input_shape=[self.state_size]),
            tf.keras.layers.Dense(self.action_size)
        ])

    def call(self, state):
        values = self.value_stream(state)
        advantage = self.advantage(state)

        Q_values = values + (advantage - tf.reduce_mean(advantage))

        return Q_values

