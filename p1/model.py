import torch 
import torch.nn as nn
import torch.nn.functional as F 


class QNetwork(nn.Module):

      def __init__(self, s_dim, a_dim):
           super().__init__()
           self.s_dim = s_dim
           self.a_dim = a_dim
           self.nhid1 = 128
           self.nhid2 = 128
           self.layer1 = nn.Linear(self.s_dim, self.nhid1)
           self.layer2 = nn.Linear(self.nhid1, self.nhid2)
           self.output = nn.Linear(self.nhid2, self.a_dim) 

      def forward(self, s):
           h = F.relu(self.layer1(s))
           h = F.relu(self.layer2(h))  
           Q = self.output(h)
           return Q
