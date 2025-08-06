"""Classic implementation to check whether I got the idea right.

Leans on OpenAI Gym for the environment."""

import gymnasium as gym
import numpy as np


env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)


# Process parameters:
discount_factor = 0.95  # Discount factor for future rewards.
delta = 0.5  # Threshold to determine exploration vs exploitation.
delta_decay_factor = 0.999  # Decay of said threshold, leading it towards eventual exploitation.
learning_rate = 0.8
num_episodes = 1000  # Number of episodes to run.


# Initialize the Q-table with zeros, to the dimensions of the state and action spaces.
# This already shows the weakness of this approach, as it scales quadratically with the
# size of the states and the action space. For large discrete spaces or any non-discrete spaces, 
# this is not feasible at all.
q_table = np.zeros((env.observation_space.n, env.action_space.n))


# Define the training loop.
for episode in range(num_episodes):
    state, info = env.reset()

    delta *= delta_decay_factor  # Decay the exploration threshold.
    finished = False

    while not finished:
        # First choose either a random action or the so far best action from the Q-table:
        if np.random.random() < delta or np.sum(q_table[state, :]) == 0:
            action = env.action_space.sample()

        else:
            action = np.argmax(q_table[state, :])

        # Take the action in the environment and receive the new state, reward, and finished flag:
        new_state, reward, finished, truncated, info = env.step(action)

        # Update the Q-table using the Bellman equation:
        q_table[state, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        # Update the state to the new state:
        state = new_state


print(q_table)
