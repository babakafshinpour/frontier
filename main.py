import copy
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ddpg_NN import DDPGagent_NN

from model import CytokineDynamic
from rl.environment import CustomEnv
from utils import *

# observation includes the estimation of dynamic type and the I50.

env = CustomEnv(
    steriod_max=1,
    observation_min=[0.0, 0.0, 0.0] + [0.0] * 5 + [0],  # ,0],
    observation_max=[4.5, 4.5, 1.0] + [0.0] * 5 + [15],  # ,600],
    p0=0.1,
    a0=0,
    s0=0,
    I50=10,
    mu_plasma=0.03,  # (0.004,0.03),
    step_size=1,
    eps_perb=0,
    behaviour=3,
    time_scale=4,
    cytokine_model="normalized",
)

agent = DDPGagent_NN(
    env,
    hidden_size=100,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-4,
    gamma=0.995,
    tau=0.05,
)

agent.actor.cuda(1)
agent.actor_target.cuda(1)
agent.critic.cuda(1)
agent.critic_target.cuda(1)

batch_size = 48 * 20
rewards = []
avg_rewards = []

best_agent = None
best_avg_rewards = -np.inf

for episode in range(1000):
    p0 = np.random.uniform(0.5, 1.5)
    time_scale = np.random.choice(a=[0.25, 0.5, 1, 2, 4])
    behaviour = np.random.choice(a=[1, 2, 3, 4, 5])
    mu_plasma = np.random.uniform(0.004, 0.03)
    I50 = np.random.uniform(5, 10)

    noise = OUNoise(env.action_space)

    env = CustomEnv(
        steriod_max=1,
        observation_min=[0.0, 0.0, 0.0] + [0.0] * 5 + [0],  # ,0],
        observation_max=[4.5, 4.5, 1.0] + [0.0] * 5 + [15],  # ,600],
        p0=p0,
        a0=0,
        s0=0,
        I50=I50,
        mu_plasma=mu_plasma,  # mu_plasma,  #(0.004,0.03),
        step_size=1,
        eps_perb=0.3,
        behaviour=behaviour,
        time_scale=time_scale,
        cytokine_model="normalized",
    )

    state = env.reset(p0=p0)

    # time_shift = np.random.choice(a=range(0,24))
    # env.cytokine.step_to(time_shift)

    noise.reset()
    episode_reward = 0
    flag = np.random.rand(1, 1) > 0.5
    state_list = [state]
    action_list = [np.array([0])]
    for step in range(48):
        action = agent.get_action(state)
        # print(action)
        action = noise.get_action(action, step)
        # print(action)
        # if flag and (step<10):
        #    action[0] = 0.0
        new_state, reward, done, info = env.step(action)
        action = info["action"]

        state_list.append(new_state)
        action_list.append(action)

        agent.memory.push(new_state, action, reward, new_state, done)
        # agent.memory.push([s.copy() for s in state_list], [a.copy() for a in action_list], reward, new_state, done)

        # new_state, reward, done, info = env.step(np.array([0.0]))
        # action = info["action"]
        # agent.memory.push(state, action, reward, new_state, done)

        # print(action)

        state = new_state
        episode_reward += reward

        if done:
            break

    if len(agent.memory) > batch_size:
        agent.update(batch_size)

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

    sys.stdout.write(
        "episode: {}, reward: {}, average _reward: {}, sum: {} \n".format(
            episode,
            np.round(episode_reward, decimals=2),
            np.mean(rewards[-10:]),
            sum(env.action_list[-24:]),
        )
    )

    if (best_avg_rewards < avg_rewards[-1]) and (len(agent.memory) > batch_size):
        best_agent = copy.deepcopy(agent)
        best_avg_rewards = avg_rewards[-1]
