import numpy as np
import random
import copy
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

from AC import Actor, Critic
from buffer import *
from noise import *



#---------------------------------------------------------------------------------------------------

BUFFER_SIZE = int(1e6)  
BATCH_SIZE = 256        

GAMMA = 0.99            
TAU = 1e-3              
LR_A = 2e-4         
LR_C = 1e-3       

UPDATE_EVERY = 20       
NUM_UPDATES = 10         


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------------------------------------------------------


class Agent():

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
  

        # actor
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)

        # critic
        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)

        # optimizers
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LR_A)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LR_C)

        # OU noise
        self.noise = OUNoise(action_size, theta=0.15, sigma=0.1) 

        # replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, device)
        
        self.params_copy(self.actor_target, self.actor_local)
        self.params_copy(self.critic_target, self.critic_local)


    def step(self, states, actions, rewards, next_states, dones, tstep):        
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.memory.add(state, action, reward, next_state, done)

            if len(self.memory) > BATCH_SIZE and tstep % UPDATE_EVERY == 0:   
               for _ in range(NUM_UPDATES):
                   experiences = self.memory.sample(BATCH_SIZE)
                   self.train_model(experiences)
                   

    def act(self, state, epsilon, add_noise=True):

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += epsilon * self.noise.sample()
        return action


    def reset(self):
        self.noise.reset()


    def train_model(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        # Q(s_t+1, a_t+1)   
        actions_next = self.actor_target(next_states)
        Q_tp1 = self.critic_target(next_states, actions_next)

        # Q targets, y_t; Bellman equation
        y_t = rewards + (GAMMA * Q_tp1 * (1 - dones))

        Q_t = self.critic_local(states, actions)
        self.critic_loss = F.mse_loss(Q_t, y_t)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

        actions_pred = self.actor_local(states)
        self.actor_loss = -self.critic_local(states, actions_pred).mean()   

        self.update_networks()

    
    def update_networks(self):

        # actor update
        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        # update target networks using TAU
        self.weighted_update(self.critic_local, self.critic_target, TAU)
        self.weighted_update(self.actor_local, self.actor_target, TAU)

        self.noise.reset() 


    def weighted_update(self, source, target, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau*s.data + (1.0-tau)*t.data)

    def params_copy(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

#-------------------------------------------------------------------------------------
