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

EPSILON = 1.0           
DELTA_EPSILON = 1e-6    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#---------------------------------------------------------------------------------------------------


class Agent():

    def __init__(self, state_size, action_size, buffer_type='standard'):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON
        self.buffer_type = buffer_type
  

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
        self.noise = OUNoise(action_size)

        # replay buffer
        if (self.buffer_type == 'standard'):
            self.memory = ReplayBuffer(BUFFER_SIZE, device)
        else:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE)  

        self.params_copy(self.actor_target, self.actor_local)
        self.params_copy(self.critic_target, self.critic_local)


    def step(self, states, actions, rewards, next_states, dones, tstep):
        
        if (self.buffer_type == 'standard'):

            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.memory.add(state, action, reward, next_state, done)

            if len(self.memory) > BATCH_SIZE and tstep % UPDATE_EVERY == 0:   
               for _ in range(NUM_UPDATES):
                   experiences = self.memory.sample(BATCH_SIZE)
                   self.train_model_s(experiences)
                   
        elif (self.buffer_type == 'prioritized'): 
 
            s = torch.FloatTensor(states).cuda()
            a = torch.FloatTensor(actions).cuda()
            r = torch.FloatTensor(rewards).cuda()
            s2 = torch.FloatTensor(next_states).cuda()
            d = torch.FloatTensor(dones).cuda()

            actions_next = self.actor_target(s2)
            Q_tp1 = self.critic_target(s2, actions_next).squeeze(1)
            y_t = r + (GAMMA * Q_tp1 * (1.0 - d))
            Q_t = self.critic_local(s, a).squeeze(1)
            TD_error = torch.abs(Q_t - y_t).cpu().data.numpy()

            for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):   
                self.memory.add(TD_error[i], (state, action, reward, next_state, done))
 
            if self.memory.size() > BATCH_SIZE and tstep % UPDATE_EVERY == 0:    
               for _ in range(NUM_UPDATES):
                   experiences, idxs, is_weight = self.memory.sample(BATCH_SIZE) 
                   experiences = self.convert_tuple_format(experiences)
                   self.train_model_p(experiences, GAMMA, idxs, is_weight)      


    def convert_tuple_format(self, mini_batch):
        s, a, r, s2, d = [], [], [], [], []
        for b in mini_batch:
            s_, a_, r_, s2_, d_ = b
            s.append(s_)
            a.append(a_)
            r.append(r_)
            s2.append(s2_)
            d.append(d_)   
        return (s, a, r, s2, d)


    def act(self, state, add_noise=True):

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.noise.sample()
        return action


    def reset(self):
        self.noise.reset()


    # standard exp replay
    def train_model_s(self, experiences):

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

    
    # prioritized exp replay
    def train_model_p(self, experiences, gamma, idxs=None, is_weight=None): 

        states, actions, rewards, next_states, dones = experiences

        s = torch.FloatTensor(states).cuda() 
        a = torch.FloatTensor(actions).cuda()
        r = torch.FloatTensor(rewards).cuda()
        s2 = torch.FloatTensor(next_states).cuda()
        d = torch.FloatTensor(dones).cuda()
       
        # Q(s_t+1, a_t+1)   
        actions_next = self.actor_target(s2)
        Q_tp1 = self.critic_target(s2, actions_next).squeeze(1)

        # Q targets, y_t; Bellman equation
        y_t = r + (GAMMA * Q_tp1 * (1.0 - d))
        Q_t = self.critic_local(s, a).squeeze(1)
 
        is_w = torch.FloatTensor(is_weight).cuda()
        self.critic_loss = torch.mean(is_w * (Q_t - y_t) ** 2)


        # update priority
        TD_errors = torch.abs(Q_t - y_t).cpu().data.numpy()
        for i in range(BATCH_SIZE):
           idx = idxs[i]
           self.memory.update(idx, TD_errors[i])


        # critic update; clip grad norm for stability
        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 2.0)
        self.critic_optim.step()


        # actor loss; gradient of this term is the "policy gradient"
        actions_pred = self.actor_local(s)
        self.actor_loss = -self.critic_local(s, actions_pred).mean()

        self.update_networks()



    def update_networks(self):

        # actor update
        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()

        # update target networks using TAU
        self.weighted_update(self.critic_local, self.critic_target, TAU)
        self.weighted_update(self.actor_local, self.actor_target, TAU)

        # linearly anneal epsilon; noise factor
        self.epsilon -= DELTA_EPSILON
        self.noise.reset() 


    def weighted_update(self, source, target, tau):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(tau*s.data + (1.0-tau)*t.data)

    def params_copy(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(s.data)

#-------------------------------------------------------------------------------------
