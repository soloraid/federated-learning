from time import sleep
import pandas as pd

import gym
from numpy import savetxt

import deepqlearningagent as dqn
from tqdm import tqdm
import matplotlib.pyplot as plt

world = 'CartPole-v0'
seed = 1423
environment_cart_pole = gym.make(world)
input_dimension = environment_cart_pole.observation_space.shape[0]
output_dimension = environment_cart_pole.action_space.n
lr = 1e-3
sync_freq = 5
exp_replay_size = 256
episodes = 10000
agent = dqn.deepQLearningAgent(seed=seed, layer_sizes=[input_dimension, 64, output_dimension], lr=lr,
                               sync_freq=sync_freq,
                               exp_replay_size=exp_replay_size)


def learning():
    losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
    index = 0

    epsilon = 1
    for i in tqdm(range(episodes)):
        observation, done, losses, episode_len, totalReward = environment_cart_pole.reset(), False, 0, 0, 0
        while not done:
            episode_len += 1
            action = agent.get_action(observation, environment_cart_pole.action_space.n, epsilon)
            observation_next, reward, done, _ = environment_cart_pole.step(action.item())
            agent.collect_experience(observation, action.item(), reward, observation_next)

            observation = observation_next
            totalReward += reward
            index += 1

            if index > 128:
                index = 0
                for j in range(4):
                    loss = agent.train(batch_size=16)
                    losses += loss

        if epsilon > 0.05:
            epsilon -= 1 / 5000

        losses_list.append(losses / episode_len)
        reward_list.append(totalReward)
        episode_len_list.append(episode_len)
        epsilon_list.append(epsilon)

    print("Saving trained model")
    agent.save_trained_model("lunar-lander-dqn.pth")
    savetxt('losses1.csv', losses_list, delimiter=',')
    savetxt('rewards1.csv', reward_list, delimiter=',')
    savetxt('episodes_len1.csv', losses_list, delimiter=',')
    savetxt('epsilons1.csv', epsilon_list, delimiter=',')


def demo():
    agent.load_pretrained_model("cartpole-dqn.pth")

    reward_arr = []
    for i in tqdm(range(10)):
        obs, done, rew = environment_cart_pole.reset(), False, 0
        while not done:
            A = agent.get_action(obs, environment_cart_pole.action_space.n, epsilon=0)
            obs, reward, done, info = environment_cart_pole.step(A.item())
            rew += reward
            sleep(0.0001)
            environment_cart_pole.render()
        reward_arr.append(rew)
    print("average reward per episode :", sum(reward_arr) / len(reward_arr))


def plotting():
    file1 = pd.read_csv('rewards.csv')
    file2 = pd.read_csv('epsilons.csv')
    file3 = pd.read_csv('losses.csv')

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
    fig3, ax3 = plt.subplots()
    ax3.plot(file3, color='red', linewidth=3)
    ax3.set_title('loss per episode')
    ax3.set_xlabel('episode')
    ax3.set_ylabel('loss')
    ax3.grid(True)
    plt.show()


# learning()
demo()
plotting()
