import numpy as np 
import random 
import torch 
import torch.nn.functional as F 
import torch.optim as optim 

from collections import namedtuple, deque 
from model import QNetwork 
from replay_buffer import ReplayBuffer

#---------------------------------------------------------------------------------------------------------------------


class Agent():

      def __init__(self, s_dim, a_dim, gamma, batch_size, lr, buffer_capacity, tau):

            self.s_dim = s_dim
            self.a_dim = a_dim
            self.gamma = gamma
            self.batch_size = batch_size 
            self.lr = lr
            self.buffer_capacity = buffer_capacity
            self.tau = tau 
            self.device = 'cuda'
 
            self.qnet_primary = QNetwork(self.s_dim, self.a_dim).to(self.device)
            self.qnet_target = QNetwork(self.s_dim, self.a_dim).to(self.device)

            self.optim = optim.Adam(self.qnet_primary.parameters(), lr=self.lr)

            self.memory = ReplayBuffer(self.s_dim, self.a_dim, self.batch_size, self.buffer_capacity)
			
            self.t_step = 0


      def add2buff_and_train(self, state, action, reward, next_state, done):
		
           # add to replay buffer
           self.memory.add(state, action, reward, next_state, done)

           # train one step from a mini-batch sampled from replay buffer  
           self.t_step = (self.t_step + 1) % 4
           if (self.t_step == 0):
              if len(self.memory) > self.batch_size: 
                   experience = self.memory.sample()
                   self.learn(experience, self.gamma)


      def act(self, state, epsilon):
		
           state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
	
           self.qnet_primary.eval()

           with torch.no_grad():
                Q_sa = self.qnet_primary.forward(state)
		
           self.qnet_primary.train()
		 
           if random.random() > epsilon:
               # exploit 
               return np.argmax(Q_sa.cpu().data.numpy())
           else: 
               # explore 
               return random.choice(np.arange(self.a_dim))


      def learn(self, experiences, gamma):
		
             states, actions, rewards, next_states, dones = experiences 

             Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
             Q_targets_next = Q_targets_next.type(torch.FloatTensor)
             Q_targets_next = Q_targets_next.to(self.device)

             # Bellman equation
             Q_targets = rewards + (gamma * Q_targets_next * (1.0 - dones))

             Q_expected = torch.gather(self.qnet_primary(states), dim=1, index=actions)
	
             loss = F.mse_loss(Q_expected, Q_targets)	
             self.optim.zero_grad()
             loss.backward()
             self.optim.step()

             self.target_update(self.qnet_primary, self.qnet_target, self.tau) 


      def target_update(self, local_model, target_model, tau):
             for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
                  target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)

      def save_model(self):
           torch.save(self.qnet_primary, 'model_qnet.pt') 

#---------------------------------------------------------------------------------------------------------------------

