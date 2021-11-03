# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 22:11:41 2021

@author: Andrija
"""

import gym
import random



env = gym.make('MsPacman-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n

env.unwrapped.get_action_meanings()

episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0
    
    while not done : 
        env.render()
        action = random.choice([0,1,2,3,4,5,6,7,8])
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode : {} Score : {}'.format(episode, score))
env.close()