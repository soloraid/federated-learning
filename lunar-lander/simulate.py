import time
import torch
import matplotlib.pyplot as plt

scores = []


def simulate_env(env, agent_dqn, model_path, n_simulation=5):
    agent_dqn.q_network_local.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent_dqn.q_network_target.load_state_dict(torch.load(model_path, map_location='cpu'))
    for _ in range(n_simulation):
        state = env.reset()
        plt.imshow(env.render(mode='rgb_array'))
        time.sleep(0.0001)
        score = 0
        for i in range(1000):
            action = agent_dqn.choose_action(state)
            plt.imshow(env.render(mode='rgb_array'))
            time.sleep(0.0001)
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break

        scores.append(score)

    env.close()


def plot_simulate_report(n_simulation=10):
    print(scores)
    fig1, scores_ax = plt.subplots()
    scores_ax.plot(scores[0:n_simulation:1], marker='o', linestyle='--', linewidth=3, label='number of agents 1')
    scores_ax.plot(scores[n_simulation:n_simulation * 2:1], marker='o', linestyle='--', linewidth=3, label='number of agents 2')
    scores_ax.plot(scores[n_simulation * 2:n_simulation * 3:1], marker='o', linestyle='--', linewidth=3, label='number of agents 3')
    scores_ax.legend(loc='lower right')
    scores_ax.set_title('simulate scores in n_agents final models')
    scores_ax.set_xlabel('update_rate')
    scores_ax.set_ylabel('reward')
    scores_ax.set_facecolor('gray')
    scores_ax.grid(True)
    fig1.savefig('scores_test_final.png')
    plt.show()
