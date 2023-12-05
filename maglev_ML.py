import random
import torch
from torch import nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import gymnasium as gym
from torch.distributions.normal import Normal
from maglev_env import MagneticEnv, DT


class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        # NOTE think more about these values
        hidden_space1 = 16
        hidden_space2 = 16

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        # NOTE do we want relu on this?
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         for each normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted means of the action space's normal distribution
            action_stddevs: predicted standard deviation of the action space's normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs

class Policy:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm.
        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        self.action_space_dims = action_space_dims

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns action(s), conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action(s) to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        action_means = action_means.squeeze()
        action_stddevs = action_stddevs.squeeze()
        # create a normal distribution from the predicted
        #   mean and standard deviation and sample all actions action
        actions = np.zeros(self.action_space_dims)
        for action_dim in range(self.action_space_dims):
            distrib = Normal(action_means[action_dim] + self.eps, action_stddevs[action_dim] + self.eps)
            action = distrib.sample()
            prob = distrib.log_prob(action)
            actions[action_dim] = action.numpy()

            self.probs.append(prob)

        return actions

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []

if __name__ == '__main__':

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device") 

    DO_RENDER = True
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)

    mag_coords = [np.array([0.,-1.,3.]),np.array([0.,1.,3.])]
    spawn_range = ((-0.1,0.1),(-0.1,0.1),(0,1))
    desired_range = ((0,0),(0,0),(0.1,0.9))
    # Create and wrap the environment
    env = MagneticEnv(mag_coords, DT)
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

    total_num_episodes = int(5e3)  # Total number of episodes
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]

    # Reinitialize agent every seed
    agent = Policy(obs_space_dims, action_space_dims)
    reward_over_episodes = []

    for episode in range(total_num_episodes):
        obs, info = wrapped_env.reset(seed=RANDOM_SEED, options=(spawn_range,desired_range))
        done = False

        while not done:
            action = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            if DO_RENDER: env.render()

            done = terminated or truncated

        reward_over_episodes.append(sum(agent.rewards)/len(agent.rewards))
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    #TODO fix this, shows the 3D view graph needlessly
    plt.figure()
    plt.plot(reward_over_episodes)
    plt.show()