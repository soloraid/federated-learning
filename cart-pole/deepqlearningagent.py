import copy
import random
from collections import deque
import torch
from torch import nn

import replayBuffer as rb


def build_nn(layer_sizes):
    assert len(layer_sizes) > 1
    layers = []
    for index in range(len(layer_sizes) - 1):
        linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
        action = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
        layers += (linear, action)
    return nn.Sequential(*layers)


class deepQLearningAgent:

    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size):
        torch.manual_seed(seed)
        self.q_network = build_nn(layer_sizes)
        self.target_network = copy.deepcopy(self.q_network)
        self.q_network.cuda()
        self.target_network.cuda()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float().cuda()
        self.experience_replay = rb.ReplayBuffer((layer_sizes[0],), batch_size=128, max_size=exp_replay_size)

    def load_pretrained_model(self, model_path):
        self.q_network.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="cartpole-dqn.pth"):
        torch.save(self.q_network.state_dict(), model_path)

    def get_action(self, state, action_space_len, epsilon):
        with torch.no_grad():
            q_value_states = self.q_network(torch.from_numpy(state).float().cuda())
        max_q_value, action = torch.max(q_value_states, axis=0)
        action = action if torch.rand(1,).item() > epsilon else torch.randint(0, action_space_len, (1,))
        return action

    def get_q_next_state(self, state):
        with torch.no_grad():
            q_values_next_state = self.target_network(state.cuda())
        max_q_value_next_state, _ = torch.max(q_values_next_state, axis=1)
        return max_q_value_next_state

    def collect_experience(self, observation, action, reward, observation_next):
        self.experience_replay.store(observation, action, reward, observation_next)
        return

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(rb.ReplayBuffer)
        return self.experience_replay.sample(sample_size)

    def train(self, batch_size):
        current_state, action, current_reward, next_state = self.sample_from_experience(sample_size=batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.network_sync_counter = 0

        q_values = self.q_network(current_state.cuda())
        prediction_return, _ = torch.max(q_values, axis=1)

        q_next_state = self.get_q_next_state(next_state)
        target_return = current_reward.cuda() + self.gamma * q_next_state

        loss = self.loss_fn(prediction_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()
