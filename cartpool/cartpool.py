
# coding: utf-8

# In[1]:

import gym
import numpy as np


# In[2]:

env = gym.make('CartPole-v0')


# In[3]:

space = env.reset()
done=False


# In[4]:

theta = np.random.random([4])
gamma = 0.5
lr = 0.005

for i in range(1000):
    print("Episode: %d"%(i))
    space = env.reset()
    done = False
    reward_list = []
    space_list = []
    if i in range(100, 500):
        lr = 0.001
    elif i in range(500, 10000):
        lr = 0.0

    while not done: 
        env.render()
        act = int(round(1/(1+np.exp(-space.dot(theta)))))
        space, reward, done, _ = env.step(act)
        reward_list.append(reward)
        space_list.append(space)
    discount_reward = 0
    if(len(reward_list)>0):
        for idx in range(len(reward_list)-1,-1, -1):
            reward = reward_list[idx]
            space = space_list[idx]
            discount_reward = gamma * discount_reward + reward
            theta = theta + lr*np.exp(-space.dot(theta))/(1+np.exp(-space.dot(theta))) * space
        print(theta)
        print(len(reward_list))
        

