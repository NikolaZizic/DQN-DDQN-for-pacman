# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:12:19 2021

@author: Nikola Zizic
"""

import gym
import ale_py

import tensorflow as tf


from tf_agents.networks import sequential
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import base64
import imageio
import matplotlib
import matplotlib.pyplot as plt

import cv2

from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.networks.q_network import QNetwork

from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam 




from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules





env_name = 'MsPacman-v0'

train_py_env = suite_gym.load(env_name)

class ProcessFrame84(gym.ObservationWrapper):
     def __init__(self, env=None):
         super(ProcessFrame84, self).__init__(env)
         self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
     
     def observation(self, obs):
         return ProcessFrame84.process(obs)
    
     @staticmethod
     def process(frame):
         if frame.size == 210 * 160 * 3:
             img = np.reshape(frame, [210, 160,  3]).astype(np.float32)
         elif frame.size == 250 * 160 * 3:
             img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
         else:
             assert False, "Unknown resolution"     
             img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
             resized_screen = cv2.resize(img, (84, 110),            
                              interpolation=cv2.INTER_AREA)
             x_t = resized_screen[18:102, :]
             x_t = np.reshape(x_t, [84, 84, 1])
             return x_t.astype(np.uint8)


def make_env (env_name):
    env = ProcessFrame84(env_name)
    return env

train_py_env_edit = make_env(train_py_env)



from tf_agents.environments.tf_py_environment import TFPyEnvironment

train_env = TFPyEnvironment(train_py_env_edit)


preprocessing_layer = keras.layers.Lambda(
    lambda obs: tf.cast(obs, np.float32) / 255.)

conv_layers_params =[(32, (8,8), 4), (64, (4,4), 2), (64, (3,3), 1)]
fc_layers_params =[256]  # fully connected layer

###### Work in progress

# In order to speed up the process of learning we should make the agent recognise only 5 possible actions instead of the 9 that premade with AIgym.
# For now can't prove if this works beacuse it seems this will only work on a GPU and we are working on our CPU.

action_spec = tensor_spec.BoundedTensorSpec((),
                                            tf.int64,
                                            minimum=0,
                                            maximum=4,
                                            name='action')


#######


q_network = QNetwork(train_env.observation_spec(),
                     train_env.action_spec(),
                     preprocessing_layers=preprocessing_layer,
                     conv_layer_params=conv_layers_params,
                     fc_layer_params=fc_layers_params)

print(train_env.action_spec())




train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε

dqn_agent = DqnAgent(train_env.time_step_spec(),
                     train_env.action_spec(),
                     q_network=q_network,
                     optimizer=optimizer,
                     target_update_period=2000, # <=> 32,000 ALE frames
                     td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                     gamma=0.99, # discount factor
                     train_step_counter=train_step,
                     epsilon_greedy=lambda: epsilon_fn(train_step))

dqn_agent.initialize()



from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=dqn_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=100000) # reduce if ROM error

replay_buffer_observer = replay_buffer.add_batch



class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")




from tf_agents.metrics import tf_metrics

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)



from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    train_env,
    dqn_agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration



from tf_agents.policies.random_tf_policy import RandomTFPolicy

initial_collect_policy = RandomTFPolicy(train_env.time_step_spec(),
                                        train_env.action_spec())
init_driver = DynamicStepDriver(
    train_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()




dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)




from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
dqn_agent.train = function(dqn_agent.train)





def train_agent(n_iterations):
    time_step = None
    policy_state = dqn_agent.collect_policy.get_initial_state(train_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = dqn_agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 100 == 0:
            log_metrics(train_metrics)
            

train_agent(n_iterations=101)

