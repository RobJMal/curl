# Note that this script was created by ChatGPT for plotting purposes
import json
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--environment', type=str, help='Name of DMC environment (ex. fish, walker)', required=True)
parser.add_argument('--task', type=str, help='Name of task for environment', required=True)
parser.add_argument('--path_to_data', type=str, help='Path to file to plot data from (eval or train)', required=True)
parser.add_argument('--ema_span', type=int, default=20, help='Span for Moving Exponential Average')
args = parser.parse_args()

logfiles_locations = {
    'fish-swim_test-0_eval' : 'tmp/fish/fish-swim-04-25-im84-b128-s411637-pixel/eval.log',
    'fish-swim_test-0_train' : 'tmp/fish/fish-swim-04-25-im84-b128-s411637-pixel/train.log',
    'cartpole-swingup_test-0_eval' : 'tmp/cartpole/cartpole-swingup-04-27-im84-b128-s348228-pixel/eval.log',
    'walker-walk_test-0_eval' : 'tmp/walker/walk/walker-walk-04-29-im84-b128-s673499-pixel/eval.log',
    'swimmer-swimmer6_test-0_eval' : 'tmp/swimmer/swimmer-swimmer6-04-30-im84-b128-s683281-pixel/eval.log',
}

# Path to the log file
log_file_path = args.path_to_data

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
plot_title = f'{collected_data_type} Episode Reward (CURL {args.environment}-{args.task})'
plt.title(plot_title)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_directory, plot_title.replace(" ", "-")))
plt.show()
plt.close()
