# Note that this script was created by ChatGPT for plotting purposes
import json
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data', type=str, help='Path to file to plot data from (eval or train)', required=True)
parser.add_argument('--ema_span', type=int, default=20, help='Span for Moving Exponential Average')
args = parser.parse_args()

# Path to the log file
log_file_path = args.path_to_data
log_directory_tokens = log_file_path.split('/')[-2].split('-')    # Log directory is always 2nd to last token
dmc_environment, dmc_task = log_directory_tokens[0], log_directory_tokens[1]

collected_data_type = ''
if 'eval' in log_file_path:
    collected_data_type = 'Eval'
elif 'train' in log_file_path:
    collected_data_type = 'Train'

results_directory = 'results'

# Function to load data from a log file
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Load and process the data
data = load_data(log_file_path)
steps = np.array([entry['step'] for entry in data])
episode_rewards = np.array([entry['episode_reward'] for entry in data])
mean_episode_rewards = np.array([entry['mean_episode_reward'] for entry in data])

# Calculate the Exponential Moving Average for episode and mean episode rewards
ema_span = 20  # Span for EMA calculation; can be adjusted
episode_rewards_ema = pd.Series(episode_rewards).ewm(span=ema_span, adjust=False).mean()
mean_episode_rewards_ema = pd.Series(mean_episode_rewards).ewm(span=ema_span, adjust=False).mean()

# Plot Episode Reward 
plt.figure(figsize=(10, 5))
plt.plot(steps, episode_rewards, label='Episode Reward', marker='o', color='blue')
plt.plot(steps, episode_rewards_ema, label="Episode Rewards EMA", linestyle='-', color='red')
plot_title = f'{collected_data_type} Episode Reward (CURL {dmc_environment}-{dmc_task})'
plt.title(plot_title)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_directory, plot_title.replace(" ", "-")))
plt.show()
plt.close()
