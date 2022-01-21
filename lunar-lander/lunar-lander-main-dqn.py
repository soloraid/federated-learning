from time import sleep

import gym
import pandas as pd
from numpy import savetxt
from tqdm import tqdm

from deepqlearningagent import Agent
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v2')
scores, eps_history = [], []
n_games = 4000
input_dimension = env.observation_space.shape[0]
output_dimension = env.action_space.n
agent = Agent(gamma=0.99, epsilon=1, lr=1e-3, input_dims=[input_dimension], batch_size=64, n_actions=4)


def demo():
    agent.load_pretrained_model("lunar-lander-dqn.pth")
    agent.epsilon = 0
    reward_arr = []
    for i in tqdm(range(10)):
        obs, done, rew = env.reset(), False, 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            rew += reward
            sleep(0.0001)
            env.render()
        reward_arr.append(rew)
    print("average reward per episode :", sum(reward_arr) / len(reward_arr))


def learning():
    index = 0
    for i in tqdm(range(1, n_games)):
        episode_len = 0
        score = 0
        done = False
        observation = env.reset()
        while not done:
            episode_len += 1
            index += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_

            if index == 4:
                index = 0
                agent.learn()
        if score > 200:
            break
        if i % 100 == 0:
            print('av_score: ', np.mean(scores[-1:-100:-1]))
            if np.mean(scores[-1:-100:-1]) > 200:
                break
        # print(episode_len)
        agent.epsilon_update()
        scores.append(score)
        eps_history.append(agent.epsilon)

    print("Saving trained model")
    agent.save_trained_model("lunar-lander-dqn.pth")
    savetxt('rewards.csv', scores, delimiter=',')
    savetxt('eps_history.csv', eps_history, delimiter=',')


def plotting():
    file1 = pd.read_csv('rewards.csv')
    file2 = pd.read_csv('eps_history.csv')

    fig1, ax1 = plt.subplots()
    ax1.plot(file1, color='orange', linewidth=3, linestyle='--')
    ax1.set_title('rewards per episode')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('total-reward')
    ax1.grid(True)
    fig2, ax2 = plt.subplots()
    ax2.plot(file2, color='green', linewidth=3)
    ax2.set_title('epsilon per episode')
    ax2.set_xlabel('episode')
    ax2.set_ylabel('epsilon')
    ax2.grid(True)
    plt.show()


if __name__ == '__main__':
    # learning()
    demo()
    plotting()
