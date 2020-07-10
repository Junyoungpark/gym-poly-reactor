import torch

import torch.nn as nn
import numpy as np


def update_target(target_params, source_params, tau=0.1):
    for t, s in zip(target_params, source_params):
        pass


class DDPGAgent(nn.Module):
    def __init__(self, state_dim, action_dim, action_min: list, action_max: list, gamma=0.99):
        super(DDPGAgent, self).__init__()
        self.action_min = np.array(action_min)
        self.action_max = np.array(action_max)
        self.gamma = gamma

        self.critic = Critic(state_dim, action_dim)
        self.actor = Actor(state_dim, action_dim)

        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.memory = None

        self.loss_ftn = nn.MSELoss()

    def get_action(self, state):
        action_before_norm = self.actor(state)

        action_after_norm = action_before_norm.data * (self.action_max - self.action_min) + self.action_min

        return action_after_norm.data

    def fit(self, batch):
        batch = self.memory.sample()

        state_batch = None
        action_batch = None
        next_state_batch = None
        reward_batch = None
        terminal_batch = None

        next_q_val = self.critic_target(next_state_batch, self.actor_target(next_state_batch))
        target_q_batch = reward_batch + self.gamma * terminal_batch * next_q_val

        q_batch = self.critic(state_batch, action_batch)

        value_loss = self.loss_ftn(q_batch, target_q_batch)

        # Critic update
        self.critic.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        # Actor update
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[64, 64], output_dim=1):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        input_dims = [state_dim + action_dim] + hidden_dim
        output_dims = hidden_dim + [output_dim]

        for in_dim, out_dim in zip(input_dims, output_dims):
            self.layers.append(nn.Linear(in_dim, out_dim))

        for i in range(len(hidden_dim)):
            self.activations.append(nn.LeakyReLU())

        self.activations.append(nn.Identity())

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        for l, act in zip(self.layers, self.activations):
            x = l(x)
            x = act(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[64, 64]):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        input_dims = [state_dim] + hidden_dim
        output_dims = hidden_dim + [action_dim]

        for in_dim, out_dim in zip(input_dims, output_dims):
            self.layers.append(nn.Linear(in_dim, out_dim))

        for i in range(len(hidden_dim)):
            self.activations.append(nn.LeakyReLU())

        self.activations.append(nn.Tanh())

    def forward(self, state):
        x = state
        for l, activation in zip(self.layers, self.activations):
            x = l(x)
            x = activation(x)
        return x
