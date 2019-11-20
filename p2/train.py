import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

from unityagents import UnityEnvironment
from ddpg import Agent


#----------------------------------------------------------------------------------------

env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')

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

#buffer_type = 'standard'
buffer_type = 'prioritized'


agent = Agent(state_size, action_size, buffer_type)

Nepisodes = 1500
Nsteps = 1000

#--------------------------------------------------------------------------------------

def train_agent():

    mean_ep_rewards = []    

        
    for ep in range(Nepisodes):

        env_info = env.reset(train_mode=True)[brain_name]       
        state = env_info.vector_observations                  
        ep_reward = np.zeros(num_agents)                          
        agent.reset()
        
        for t in range(Nsteps):

            action = agent.act(state, add_noise=True)
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

        print('episode: ', ep, ' | ep_reward: ', ep_reward, ' | mean of last 100 episodes: ', last100)      
        

        if ep % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'ckpt/' + buffer_type + '/actor.pth')
            torch.save(agent.critic_local.state_dict(), 'ckpt/' + buffer_type + '/critic.pth')
             
           
        if last100 >= 37.0:
             torch.save(agent.actor_local.state_dict(), 'ckpt/' + buffer_type + '/actor.pth')
             torch.save(agent.critic_local.state_dict(), 'ckpt/' + buffer_type + '/critic.pth')
             print('problem solved in ', ep, ' episodes!')     



        f=open("performance_" + buffer_type + ".txt", "a+")
        f.write(str(ep) + " " + str(ep_reward) + "\n")  
        f.close()


    mean_ep_rewards = np.array(mean_ep_rewards)
    xx = np.arange(len(mean_ep_rewards)) 
    plt.plot(xx, mean_ep_rewards)
    plt.xlabel('episode ', fontsize=12)
    plt.ylabel('episode rewards ', fontsize=12)
    plt.savefig('rewards_' + buffer_type + '_ddpg.png')
            
#--------------------------------------------------------------------------------------------------------------

train_agent()

env.close()

#--------------------------------------------------------------------------------------------------------------

# following is for testing agent

'''
agent.actor_local.load_state_dict(torch.load('best_checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('best_checkpoint_critic.pth'))

env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
scores = np.zeros(num_agents)                          # initialize the score (for each agent)
while True:
    actions = agent.act(states)                        # select an action (for each agent)
    env_info = env.step(actions)[brain_name]           # send all actions to tne environment
    next_states = env_info.vector_observations         # get next state (for each agent)
    rewards = env_info.rewards                         # get reward (for each agent)
    dones = env_info.local_done                        # see if episode finished
    scores += env_info.rewards                         # update the score (for each agent)
    states = next_states                               # roll over states to next time step
    if np.any(dones):                                  # exit loop if episode finished
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

env.close()
'''
