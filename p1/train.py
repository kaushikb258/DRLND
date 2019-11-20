import torch
import torch.nn
import numpy as np
import random
from unityagents import UnityEnvironment

from dqn_agent import *


#-----------------------------------------------------------------------------------------------

gamma = 0.99
nepisodes = 5000
batch_size = 64
buffer_capacity = 50000

# for epsilon-greedy
eps_start = 0.75
eps_max_steps = 1500 
eps_min = 0.02
eps_decay = 'halflife' #'linear'
eps_halflife = 250


lr = 1.0e-3
tau = 1.0e-3

mypath = "/home/kb/rlnd/p1/Banana_Linux/Banana.x86_64"

#-----------------------------------------------------------------------------------------------

def play_one_episode(env, agent, eps, brain_name):

    env_info = env.reset(train_mode=True)[brain_name]
    s = env_info.vector_observations[0]   

    done = False
    total_reward = 0

    while not done:

        a = agent.act(s, eps)
        env_info = env.step(a)[brain_name]
        s2 = env_info.vector_observations[0]   
        r = env_info.rewards[0]                  
        done = env_info.local_done[0] 

        total_reward += r

        agent.add2buff_and_train(s, a, r, s2, done)

        s = s2

    return total_reward

#-----------------------------------------------------------------------------------------------

def get_env_dim(env):
   
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]

    action_size = brain.vector_action_space_size
    print('action size: ', action_size)

    state = env_info.vector_observations[0]
    state_size = len(state)
    print('state size: ', state_size)

    return state_size, action_size, brain_name

#-----------------------------------------------------------------------------------------------

def main():

        env = UnityEnvironment(file_name=mypath, worker_id=1, seed=1)        
        s_dim, a_dim, brain_name = get_env_dim(env)

        agent = Agent(s_dim, a_dim, gamma, batch_size, lr, buffer_capacity, tau)
        
        rewards = [] 

        for ep in range(nepisodes):

            if (eps_decay == 'linear'): 
               eps = eps_start - float(ep)/float(eps_max_steps)*(eps_start - eps_min)
            elif (eps_decay == 'halflife'):
               eps = eps_start*np.exp(-float(ep)/float(eps_halflife))   
            eps = max(eps, eps_min)


            ep_rew = play_one_episode(env, agent, eps, brain_name)

            print('episode: ', ep, ' | episode reward: ', ep_rew, ' |epsilon: ', round(eps,3))
            
            rewards.append(ep_rew)

            if (len(rewards) > 100): 
                last_few = np.mean(rewards[-100:])
                if (last_few >= 13.0):
                    print('game solved, with mean reward of last 100 episodes: ', round(last_few, 3)) 
                    break
    
        env.close()

        rewards = np.array(rewards)
        np.save('rewards', rewards)           

        agent.save_model()


if __name__ == '__main__':
    main()
