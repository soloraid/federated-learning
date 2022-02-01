import gym
from federated import Federated
from agent import Agent
from simulate import simulate_env
from simulate import plot_simulate_report
import plot_train_report


choose = 0
# 1. Initialize_environment
env = gym.make('LunarLander-v2')
env.seed(0)
state = env.reset()
# 2. Initialize Agent
agent = Agent(state_size=8, action_size=4, seed=1)

while 0 <= choose <= 5:
    print('{0}:\n{1}\n{2}\n{3}\n{4}\n{5}\n'.format('Menu',
                                                   '1.episodes=200, update_rate=8, n_agents=1',
                                                   '2.episodes=200, update_rate=8, n_agents=2',
                                                   '3.episodes=200, update_rate=8, n_agents=3',
                                                   '4.plot_models_results',
                                                   '5.simulation',
                                                   '6.exit'))
    choose = int(input('\n\nEnter your choice: '))
    if choose == 1:
        Federated(n_agents=1, update_rate=8, episodes=200)
    if choose == 2:
        Federated(n_agents=2, update_rate=8, episodes=200)
        Federated(n_agents=3, update_rate=8, episodes=200)
    if choose == 3:
        Federated(n_agents=3, update_rate=8, episodes=200)
    if choose == 4:
        plot_train_report.plot(3)
    if choose == 5:
        n_simulation = 10
        agent1 = Agent(state_size=8, action_size=4, seed=1)
        agent2 = Agent(state_size=8, action_size=4, seed=2)
        agent3 = Agent(state_size=8, action_size=4, seed=3)
        env1 = gym.make('LunarLander-v2')
        env1.seed(1)
        env2 = gym.make('LunarLander-v2')
        env2.seed(2)
        env3 = gym.make('LunarLander-v2')
        env3.seed(3)
        simulate_env(env1, agent1, model_path='results/checkpoint_global_1_agents_iteration_1.pth', n_simulation=n_simulation)
        simulate_env(env2, agent2, model_path='results/checkpoint_global_2_agents_iteration_1.pth', n_simulation=n_simulation)
        simulate_env(env3, agent3, model_path='results/checkpoint_global_3_agents_iteration_1.pth', n_simulation=n_simulation)
        plot_simulate_report(n_simulation=n_simulation)
