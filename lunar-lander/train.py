import time
import torch
import numpy as np

from tqdm.asyncio import tqdm


def train_dqn(env, agent, n_episodes,
              max_step_per_episode=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, print_every=100):
    start = time.time()
    eps = eps_start
    prev_mean_score = -np.inf
    scores = []
    eps_history = [eps]
    for e in tqdm(range(1, n_episodes+1, 1)):
        state = env.reset()
        score = 0

        for _ in range(max_step_per_episode):
            action = agent.choose_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.step(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        eps = max(eps * eps_decay, eps_end)
        scores.append(score)
        eps_history.append(eps)

        if e % print_every == 0:
            mean_score = np.mean(scores[-1:-print_every:-1])
            print("Episode: {}\t Total Reward: {:.2f}".format(e, mean_score))
            if mean_score > prev_mean_score:
                # torch.save(agent.q_network_local.state_dict(), 'checkpoint_' + str(e) + '.pth')
                prev_mean_score = mean_score
                if prev_mean_score >= 200:
                    torch.save(agent.q_network_local.state_dict(), 'checkpoint_' + str(e) + '.pth')
                    break

    print("Took {:.2f} minutes for {} episodes.".format((time.time() - start) / 60, n_episodes))

    return scores, eps_history


def train_federated(env, agent, n_episodes, index, print_every=100, eps_start=1.0,
                    max_step_per_episode=1000, eps_end=0.01, eps_decay=0.995):
    eps = eps_start
    prev_mean_score = -np.inf
    scores = []
    for e in range(0, n_episodes):
        state = env.reset()
        score = 0

        for _ in range(max_step_per_episode):
            action = agent.choose_action(state, eps)
            next_state, reward, done, _ = env.step(action)
            score += reward
            agent.step(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
        eps = max(eps * eps_decay, eps_end)
        scores.append(score)

        if e % print_every == 0 and e != 0:
            mean_score = np.mean(scores[-1:-100:-1])
            print("{}.Episode: {}\t Total Reward: {:.2f}".format(index, e, mean_score))
            if mean_score > prev_mean_score:
                prev_mean_score = mean_score
                if prev_mean_score >= 200:
                    break

    return prev_mean_score, eps
