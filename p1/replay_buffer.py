import numpy as np 
import random 
import torch 
import torch.nn.functional as F 
import torch.optim as optim 

from collections import namedtuple, deque 

#---------------------------------------------------------------------------------------------------------------------

class ReplayBuffer():

      def __init__(self, s_dim, a_dim, batch_size, buffer_capacity):

            self.s_dim = s_dim
            self.a_dim = a_dim
            self.batch_size = batch_size
            self.buffer_capacity = buffer_capacity      
            self.device = 'cuda'

            self.buffer = deque(maxlen=self.buffer_capacity)
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

		
      def add(self, state, action, reward, next_state, done):
             e = self.experience(state, action, reward, next_state, done) 
             self.buffer.append(e)

      def sample(self):
             experiences = random.sample(self.buffer, k=self.batch_size)

             states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
             actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
             rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
             next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
             dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

             states = states.to(self.device)  
             actions = actions.to(self.device)
             rewards = rewards.to(self.device)
             next_states = next_states.to(self.device)
             dones = dones.to(self.device)

             return (states, actions, rewards, next_states, dones)


      def __len__(self):
             return len(self.buffer)

#---------------------------------------------------------------------------------------------------------------------

