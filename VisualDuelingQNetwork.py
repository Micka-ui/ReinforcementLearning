import tensorflow as tf

class DuelingQNetwork(tf.keras.Model):

    def __init__(self,actions_size,state_size):
        super(DuelingQNetwork,self).__init__()
        self.action_size = actions_size
        self.state_size = state_size

        self.features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation = 'relu',input_shape= self.state_size ),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32,activation='elu')
        ])

        self.value_stream = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='elu', input_shape=[32]),
            tf.keras.layers.Dense(1)
        ])

        self.advantage = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='elu', input_shape=[32]),
            tf.keras.layers.Dense(self.action_size)
        ])

    def call(self, state):
        features = self.features(state)
        values = self.value_stream(features)
        advantage = self.advantage(features)

        Q_values = values + (advantage - tf.reduce_mean(advantage))

        return Q_values

