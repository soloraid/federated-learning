from builtins import enumerate

import gym
import numpy as np
import torch as T
from tqdm import tqdm

from train import train_dqn, train_federated
from agent import Agent

update_rate = 20
global_agent = Agent(state_size=8, action_size=4, seed=0)

envs = [gym.make('LunarLander-v2'), gym.make('LunarLander-v2'), gym.make('LunarLander-v2')]
envs[0].seed(1)
envs[1].seed(2)
envs[2].seed(3)

agents = [Agent(state_size=8, action_size=4, seed=1), Agent(state_size=8, action_size=4, seed=2),
          Agent(state_size=8, action_size=4, seed=3)]

previous_reward = -np.inf
rewards = np.zeros(50)
for j in tqdm(range(1, 50)):
    scores = []
    for index, agent in enumerate(agents):
        agent_score = train_federated(envs[index], agent, n_episodes=100, index=index)
        scores.append(agent_score)
    rewards[j] = np.mean(scores)
    states_local = global_agent.q_network_local.state_dict()
    states_target = global_agent.q_network_target.state_dict()
    local_dicts = []
    target_dicts = []
    for agent in agents:
        local_dicts.append(agent.q_network_local.state_dict())
        target_dicts.append(agent.q_network_target.state_dict())
    for key in states_local:
        states_local[key] -= states_local[key]
        for index, local_dict in enumerate(local_dicts):
            states_local[key] += scores[index] * local_dict[key]
        states_local[key] /= sum(scores)
    for key in states_target:
        states_target[key] -= states_target[key]
        for index, target_dict in enumerate(target_dicts):
            states_target[key] += scores[index] * target_dict[key]
        states_target[key] /= sum(scores)
    for index, agent in enumerate(agents):
        agent.q_network_local.load_state_dict(states_local)
        agent.q_network_target.load_state_dict(states_target)
    print("Episode: {}\t Total Reward: {:.2f}".format(j, rewards[j]))
    if rewards[j] > previous_reward:
        previous_reward = rewards[j]
        T.save(global_agent.q_network_local.state_dict(), 'checkpoint_global_' + str(j) + '.pth')
