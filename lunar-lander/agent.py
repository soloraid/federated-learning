import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def update_target_network(local_model,
                          target_model,
                          tau):
    for target_w, local_w in zip(target_model.parameters(), local_model.parameters()):
        target_w.data.copy_(tau * local_w.data + (1.0 - tau) * target_w.data)


class Agent:

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.q_network_local = QNetwork(state_size, action_size, seed).to(device)
        self.q_network_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.counter_step = 1

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if self.counter_step % UPDATE_EVERY == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
            self.counter_step = 1
        else:
            self.counter_step += 1

    def choose_action(self, state, eps=0.):
        state = torch.from_numpy(np.array(state)).unsqueeze(0).float().to(device)
        self.q_network_target.eval()
        with torch.no_grad():
            action_values = self.q_network_target(state)

        actions_len = self.action_size
        policy = np.ones(actions_len) * eps / actions_len
        policy[torch.argmax(action_values)] = 1 - eps + eps / actions_len
        action = np.random.choice(range(actions_len), p=policy)

        return action

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        self.q_network_local.train()
        Q_pred_current_state = self.q_network_local(states)
        Q_pred = Q_pred_current_state.gather(dim=1, index=actions)

        self.q_network_target.eval()
        with torch.no_grad():
            Q_pred_next_state = self.q_network_target(next_states)

        Q_targets = rewards + (gamma * Q_pred_next_state.data.max(1)[0].unsqueeze(1) * (1 - dones))

        self.optimizer.zero_grad()
        criterion = torch.nn.MSELoss()
        loss = criterion(Q_pred, Q_targets)
        loss.backward()
        self.optimizer.step()

        update_target_network(self.q_network_local, self.q_network_target, TAU)


class ReplayBuffer:
    def __init__(self,
                 action_size,
                 buffer_size,
                 batch_size,
                 seed
                 ):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        tuples = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.vstack([t.state for t in tuples])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in tuples])).long().to(device)
        rewards = torch.from_numpy(np.vstack([t.reward for t in tuples])).float().to(device)
        next_states = torch.from_numpy(np.vstack([t.next_state for t in tuples])).float().to(device)
        dones = torch.from_numpy(np.vstack([int(t.done) for t in tuples])).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
