import torch

import torch.nn as nn
import numpy as np


class DDPGAgent(nn.Module):
    def __init__(self, state_dim, action_dim, action_min: list, action_max: list):
        super(DDPGAgent, self).__init__()
        self.action_min = action_min
        self.action_max = action_max

        self.critic = Critic(state_dim, action_dim)
        self.actor = Actor(state_dim, action_dim)

    def get_action(self, state):
        action = self.actor(state)
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[64, 64], output_dim=1):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()

        input_dims = [state_dim + action_dim] + hidden_dim
        output_dims = hidden_dim + [output_dim]

        for in_dim, out_dim in zip(input_dims, output_dims):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        for l in self.layers:
            x = l(x)

        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[64, 64]):
        super(Actor, self).__init__()
        self.layers = nn.ModuleList()

        input_dims = [state_dim] + hidden_dim
        output_dims = hidden_dim + [action_dim]

        for in_dim, out_dim in zip(input_dims, output_dims):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(nn.ReLU())
        # self.l1 = nn.Linear(state_dim, hidden_dim[0])

    def forward(self, state):
        x = state
        for l in self.layers:
            x = l(x)
        return x


if __name__ == '__main__':
    l = nn.Sequential()
    l.append(nn.Linear(2, 3))
    l.append(nn.Linear(3, 4))

    A = torch.Tensor([2, 3])

    l(A)
