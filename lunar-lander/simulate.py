from agent import Agent
import time
import torch
import matplotlib.pyplot as plt

agent = Agent(state_size=8, action_size=4, seed=0)


def simulate_env(env, agent_dqn, model_path, n_simulation=5):
    agent_dqn.q_network_local.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent_dqn.q_network_target.load_state_dict(torch.load(model_path, map_location='cpu'))

    for _ in range(n_simulation):
        state = env.reset()
        plt.imshow(env.render(mode='rgb_array'))
        time.sleep(0.01)
        score = 0
        for _ in range(1000):
            action = agent_dqn.choose_action(state)
            plt.imshow(env.render(mode='rgb_array'))
            time.sleep(0.0001)
            state, reward, done, _ = env.step(action)
            if done:
                break
            score += reward

        print(score)

    env.close()
