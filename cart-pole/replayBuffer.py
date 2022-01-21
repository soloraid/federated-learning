import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_shape,
                 batch_size, max_size=50000):
        self.next = 0
        self.size = 0
        self.max_size = max_size
        self.batch_size = batch_size

        self.states = np.empty(shape=(max_size, *state_shape), dtype=np.float)
        self.actions = np.empty(shape=max_size, dtype=np.int64)
        self.rewards = np.empty(shape=max_size)
        self.states_p = np.empty(shape=(max_size, *state_shape), dtype=np.float)

    def __len__(self): return self.size

    def store(self, state, action, reward, state_p):
        self.states[self.next] = state
        self.actions[self.next] = action
        self.rewards[self.next] = reward
        self.states_p[self.next] = state_p

        self.next += 1
        self.size = min(self.size + 1, self.max_size)
        self.next = self.next % self.max_size

    def sample(self, batch_size=None):
        batch_size = self.batch_size \
            if batch_size is None else batch_size
        indices = np.random.choice(self.size, size=batch_size,
                                   replace=False)
        return torch.Tensor(self.states[indices]).float(), \
               torch.Tensor(self.actions[indices]).float(), \
               torch.Tensor(self.rewards[indices]).float(), \
               torch.Tensor(self.states_p[indices]).float()

    def clear(self):
        self.next = 0
        self.size = 0
        self.states = np.empty_like(self.states)
        self.actions = np.empty_like(self.actions)
        self.rewards = np.empty_like(self.rewards)
        self.states_p = np.empty_like(self.states_p)
        self.is_terminals = np.empty_like(self.is_terminals)
