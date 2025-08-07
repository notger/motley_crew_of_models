"""Neural net implementation of Q-learning on the FrozenLake environment.

Want to check that I got the idea right, so I use the classic implementation as a reference.

The idea is: Instead of a Q-table, we use a neural network to approximate the action generation
based on the state we are in. This should(!) allow us to scale to larger state spaces and incorporate
tricky things like slippery surfaces, continuous state spaces, past history, etc.

This is a first and dirty attempt, so we will definitely ignore some styling guidelines.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)


# Process parameters:
discount_factor = 0.95  # Discount factor for future rewards.
delta = 0.5  # Threshold to determine exploration vs exploitation.
delta_decay_factor = 0.999  # Decay of said threshold, leading it towards eventual exploitation.
learning_rate = 0.001
num_episodes = 1000  # Number of episodes to run.


# Initialize the neural network to approximate the Q-values.
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# Initialize the Q-network and optimizer
q_network = QNetwork(env.observation_space.n, env.action_space.n)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)


def generate_torch_state(state):
    """Convert the state to a one-hot encoded tensor."""
    torch_state = torch.zeros(env.observation_space.n, dtype=torch.float32)
    torch_state[state] = 1.0
    return torch_state
    

# Define the training loop:
for episode in range(num_episodes):
    state, info = env.reset()
    state = generate_torch_state(state)

    delta *= delta_decay_factor  # Decay the exploration threshold
    finished = False

    while not finished:
        # Choose either a random action or the best action from the Q-network:
        if np.random.random() < delta or torch.sum(q_network(state)) == 0:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(q_network(state)).item()

        # Take the action in the environment and receive the new state, reward, and finished flag:
        new_state, reward, finished, truncated, info = env.step(action)
        new_state = generate_torch_state(new_state)

        # Compute the target Q-value using the Bellman equation:
        with torch.no_grad():
            target = reward + discount_factor * torch.max(q_network(new_state))

        # Compute the current Q-value:
        current_q_value = q_network(state)[0]

        print(q_network(state))

        # Compute the loss and update the Q-network:
        loss = nn.functional.mse_loss(current_q_value, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the state to the new state:
        state = new_state
