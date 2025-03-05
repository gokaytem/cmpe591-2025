import json
import matplotlib.pyplot as plt

# Load the training progress from the JSON file
with open('training_progress_2_0-1000.json', 'r') as f:
    training_progress = json.load(f)

rewards_list = training_progress['rewards_list']
rps_list = training_progress['rps_list']

# Plotting the rewards and RPS for episodes 0-2000
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards_list)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward over Episodes (0-1000)')

plt.subplot(1, 2, 2)
plt.plot(rps_list)
plt.xlabel('Episode')
plt.ylabel('RPS (Reward per Step)')
plt.title('RPS over Episodes (0-1000)')

plt.tight_layout()
plt.savefig('rewards_and_rps_2_0-1000.png')  # Save the figure
plt.show()