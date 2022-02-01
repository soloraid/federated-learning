import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import params


def plot(n_agents):
    fig1, agents_scores_ax = plt.subplots()
    fig2, global_rewards_ax = plt.subplots()
    fig3, agents_epsilons_ax = plt.subplots()
    agents_scores_ax.set_title('agents mean scores per 100 episodes')
    agents_scores_ax.set_xlabel('per 100 episodes')
    agents_scores_ax.set_ylabel('reward')
    agents_scores_ax.set_facecolor('gray')
    agents_scores_ax.grid(True)
    agents_epsilons_ax.set_title('agents epsilon per update_rate')
    agents_epsilons_ax.set_xlabel('update_rate')
    agents_epsilons_ax.set_ylabel('epsilon')
    agents_epsilons_ax.set_facecolor('gray')
    agents_epsilons_ax.grid(True)
    global_rewards_ax.set_title('global_reward per update_rate')
    global_rewards_ax.set_xlabel('update_rate')
    global_rewards_ax.set_ylabel('global_reward')
    global_rewards_ax.set_facecolor('gray')
    global_rewards_ax.grid(True)
    for index in range(1, n_agents + 1):
        agents_scores = pd.read_csv(str(index) + params.agents_scores_file, header=None)
        global_rewards = pd.read_csv(str(index) + params.global_rewards_file, header=None)
        agents_epsilons = pd.read_csv(str(index) + params.agents_epsilons_file, header=None)
        agents_scores.head()
        global_rewards.head()
        agents_epsilons.head()
        global_reward = global_rewards
        for agent in range(0, index):
            label = 'number of agents ' + str(index) + ': agent_' + str(agent+1)
            scores = np.array(agents_scores[agent::index])
            epsilons = np.array(agents_epsilons[agent::index])
            agents_scores_ax.plot(scores, marker='o', linestyle='--', linewidth=3, label=label)
            agents_epsilons_ax.plot(epsilons, marker='o', linestyle='--', linewidth=3, label=label)

        global_rewards_ax.plot(global_reward, marker='o', linestyle='--', linewidth=3,
                               label='number of agent ' + str(index))

    agents_scores_ax.legend(loc='lower right')
    agents_epsilons_ax.legend(loc='upper right')
    global_rewards_ax.legend(loc='lower right')
    fig1.savefig('agents_scores.png')
    fig2.savefig('agents_epsilons.png')
    fig3.savefig('global_rewards.png')
    plt.show()
