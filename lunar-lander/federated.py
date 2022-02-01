import random
from builtins import enumerate

import gym
import numpy as np
import torch as T
from tqdm import tqdm
from agent import Agent


class Federated:
    def __init__(self, n_agents, update_rate, episodes, epsilon=1.0):
        self.n_agents = n_agents
        self.update_rate = update_rate
        self.global_rewards = np.zeros(update_rate)
        self.global_agent = Agent(state_size=8, action_size=4, seed=0)
        self.agents = []
        self.agents_scores = []
        self.agents_epsilons = []
        self.environments = []
        self.agents_rewards = []
        for index in range(1, n_agents + 1):
            seed = index
            self.agents.append(Agent(state_size=8, action_size=4, seed=seed))
            env = gym.make('LunarLander-v2')
            env.seed(seed=seed)
            self.environments.append(env)
        self.train_federated(epsilon, episodes)

    def train_federated(self, global_epsilon, episodes):
        previous_reward = -np.inf
        for rate_index in tqdm(range(0, self.update_rate)):
            scores = []
            epsilons = []
            for index, agent in enumerate(self.agents):
                agent_score, epsilon = self.train_dqn(self.environments[index], agent, n_episodes=episodes + 1,
                                                      index=index + 1, print_every=100,
                                                      eps_start=global_epsilon)
                scores.append(agent_score)
                epsilons.append(epsilon)
                self.agents_scores.append(agent_score)
                self.agents_epsilons.append(epsilon)

            self.global_rewards[rate_index] = np.mean(scores)
            global_epsilon = np.mean(epsilons)
            update_scores = np.array(scores) - min(scores) + 1
            self.aggregation_models(update_scores)
            self.set_local_networks()
            if self.global_rewards[rate_index] > previous_reward:
                previous_reward = self.global_rewards[rate_index]
                T.save(self.global_agent.q_network_local.state_dict(),
                       'checkpoint_global_' + str(self.n_agents) + '_agents_iteration_' + str(rate_index+1) + '.pth')

        scores = str(self.n_agents) + '_agents_scores.csv'
        global_rewards = str(self.n_agents) + '_agents_global_rewards.csv'
        epsilons = str(self.n_agents) + '_agents_epsilons.csv'
        np.savetxt(scores, self.agents_scores)
        np.savetxt(global_rewards, self.global_rewards)
        np.savetxt(epsilons, self.agents_epsilons)

    def aggregation_models(self, scores):
        states_local = self.global_agent.q_network_local.state_dict()
        states_target = self.global_agent.q_network_target.state_dict()
        local_dicts = []
        target_dicts = []
        for agent in self.agents:
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

        self.global_agent.q_network_local.load_state_dict(states_local)
        self.global_agent.q_network_target.load_state_dict(states_target)

    def set_local_networks(self):
        for agent in self.agents:
            agent.q_network_local.load_state_dict(
                self.global_agent.q_network_target.state_dict())
            agent.q_network_target.load_state_dict(
                self.global_agent.q_network_target.state_dict())

    @staticmethod
    def train_dqn(environment, agent, n_episodes, index, print_every=100, eps_start=1.0,
                  max_step_per_episode=1000, eps_end=0.01, eps_decay=0.995):
        eps = eps_start
        prev_mean_score = -np.inf
        scores = []
        for e in range(0, n_episodes):
            state = environment.reset()
            score = 0

            for _ in range(max_step_per_episode):
                action = agent.choose_action(state, eps)
                next_state, reward, done, _ = environment.step(action)
                score += reward
                agent.step(state, action, reward, next_state, done)
                if done:
                    break
                state = next_state
            eps = max(eps * eps_decay, eps_end)
            scores.append(score)

            if e % print_every == 0 and e != 0:
                mean_score = np.mean(scores[-1:-100:-1])
                print("agent_{}.Episode: {}\t Total Reward: {:.2f}".format(index, e, mean_score))
                if prev_mean_score < mean_score:
                    prev_mean_score = mean_score
                    if prev_mean_score >= 200:
                        break

        return prev_mean_score, eps
