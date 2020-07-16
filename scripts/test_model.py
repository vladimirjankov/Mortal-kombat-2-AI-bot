import numpy as np
import retro
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D,SimpleRNN,GlobalAveragePooling2D
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy, MaxBoltzmannQPolicy
from rl.memory import SequentialMemory
from mortalkombat_env import PlayerOneNetworkControllerWrapper, ObservationWraperMK, make_env


# crates env for motal kombat
env = make_env()

# adds neural network layers
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=3, strides=1, activation="relu", input_shape=(4, 112, 160),  data_format="channels_first"))
model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation="relu" ))
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

#set the policy for action selection
policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)

#deep q learning agent
player1 = DQNAgent(model=model,
                   nb_actions=env.action_space.n,
                   enable_dueling_network=True,
                   enable_double_dqn=True,
                   memory=memory,
                   nb_steps_warmup=200,
                   target_model_update=1e-2,
                   policy=policy)

player1.compile(Adam(lr=1e-3), metrics=['mae'])

#loads weights and tests the model
player1.load_weights('mk_13.h5f')
player1.test(env,nb_episodes= 10, visualize=False)
