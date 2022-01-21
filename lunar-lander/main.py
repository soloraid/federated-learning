import gym
from numpy import savetxt

from train import train_dqn
from agent import Agent
from simulate import simulate_env
from plotting import plot

# 1. Initialize_environment
env = gym.make('LunarLander-v2')
env.seed(0)
state = env.reset()

# 2. Initialize Agent
agent = Agent(state_size=8, action_size=4, seed=3)

# 3. Train Agent
# scores, epsilon_history = train_dqn(env, agent, n_episodes=4000)
# savetxt('rewards.csv', scores, delimiter=',')
# savetxt('eps_history.csv', epsilon_history, delimiter=',')
# 4. Simulate Agent in the Environment
# simulate_env(env, agent, model_path='checkpoint_old_1700.pth')
simulate_env(env, agent, model_path='checkpoint_global_25.pth')
plot()