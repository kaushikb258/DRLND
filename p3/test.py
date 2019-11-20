import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

from unityagents import UnityEnvironment
from ddpg import Agent


#----------------------------------------------------------------------------------------

env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

num_agents = len(env_info.agents)
print('number of agents: ', num_agents)


action_size = brain.vector_action_space_size
print('action size: ', action_size)

 
states = env_info.vector_observations
state_size = states.shape[1]
print('state size: ', state_size)


#--------------------------------------------------------------------------------------

agent = Agent(state_size, action_size)

#--------------------------------------------------------------------------------------

def test_agent():

    agent.actor_local.load_state_dict(torch.load('ckpt/actor.pth'))
    agent.critic_local.load_state_dict(torch.load('ckpt/critic.pth'))

    env_info = env.reset(train_mode=False)[brain_name]      

    s = env_info.vector_observations                  

    ep_reward = 0.0

    while True:

      a = agent.act(s, epsilon=0.0, add_noise=False)                        

      env_info = env.step(a)[brain_name]           

      s2 = env_info.vector_observations         

      r = env_info.rewards                         

      done = env_info.local_done                        

      ep_reward += np.mean(np.array(r))                         

      s = s2                              

      if np.any(done):                                 
          break

    return ep_reward

#--------------------------------------------------------------------------------------------------------------

Nepisodes = 10


for ep in range(Nepisodes):
   ep_reward = test_agent()
   print('episode: ', ep, ' | episode reward: ', ep_reward)


env.close()
#--------------------------------------------------------------------------------------------------------------
