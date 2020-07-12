# import torch
import numpy as np


class RandomAgent():
    def __init__(self, state_dim, action_min: list, action_max: list):
        self.action_min = action_min
        self.action_max = action_max

    def get_action(self, state):
        action = np.random.uniform(low=self.action_min, high=self.action_max)
        return action
