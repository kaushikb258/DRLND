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

Nepisodes = 5000
Nsteps = 5000

#--------------------------------------------------------------------------------------

def train_agent():

    mean_ep_rewards = []    

    ibackup = 0  
    thresh = 0.05

     

    for ep in range(Nepisodes):

        env_info = env.reset(train_mode=True)[brain_name]       
        state = env_info.vector_observations                  
        ep_reward = np.zeros(num_agents)                          
        agent.reset()

        epsilon = 1.0        


        for t in range(Nsteps):

            action = agent.act(state, epsilon, add_noise=True)
            env_info = env.step(action)[brain_name]           

            next_state = env_info.vector_observations         
            reward = env_info.rewards                         
            done = env_info.local_done                        

            agent.step(state, action, reward, next_state, done, t)

            state = next_state                               
            ep_reward += reward                                        

            if np.any(done):                                  
                break
       
        ep_reward = np.mean(ep_reward)
        mean_ep_rewards.append(ep_reward)

        last100 = 0.0

        if (len(mean_ep_rewards) > 100): 
             last100 = mean_ep_rewards[-100:]
             last100 = np.mean(np.array(last100))                 

        print('episode: ', ep, ' | ep_reward: ', round(ep_reward,4), ' | mean of last 100 episodes: ', round(last100,4), ' | epsilon: ', round(epsilon,3))      
        

        if (ep % 10 == 0):
            torch.save(agent.actor_local.state_dict(), 'ckpt/actor.pth')
            torch.save(agent.critic_local.state_dict(), 'ckpt/critic.pth')
             

        if (last100 >= thresh):
             torch.save(agent.actor_local.state_dict(), 'ckpt/actor_backup.pth')
             torch.save(agent.critic_local.state_dict(), 'ckpt/critic_backup.pth')
             ibackup = 1
             thresh += 0.05
        
        if (last100 < thresh-0.1 and ibackup == 1):
             agent.actor_local.load_state_dict(torch.load('ckpt/actor_backup.pth'))
             agent.critic_local.load_state_dict(torch.load('ckpt/critic_backup.pth'))


        if (last100 >= 1.0):
             torch.save(agent.actor_local.state_dict(), 'ckpt/actor.pth')
             torch.save(agent.critic_local.state_dict(), 'ckpt/critic.pth')
             print('problem solved in ', ep, ' episodes!')     
             break


        f=open("performance.txt", "a+")
        f.write(str(ep) + " " + str(ep_reward) + "\n")  
        f.close()


    mean_ep_rewards = np.array(mean_ep_rewards)
    xx = np.arange(len(mean_ep_rewards)) 
    plt.plot(xx, mean_ep_rewards)
    plt.xlabel('episode ', fontsize=12)
    plt.ylabel('episode rewards ', fontsize=12)
    plt.savefig('rewards_ddpg.png')
            
#--------------------------------------------------------------------------------------------------------------

train_agent()

env.close()

#--------------------------------------------------------------------------------------------------------------

