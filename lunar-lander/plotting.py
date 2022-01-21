import pandas as pd
from matplotlib import pyplot as plt


def plot():
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
