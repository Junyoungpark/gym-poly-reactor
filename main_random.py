import matplotlib.pyplot as plt
import numpy as np

from gym_poly_reactor.envs.poly_reactor_env import PolyReactor
from gym_poly_reactor.agents.random_agent import RandomAgent

if __name__ == '__main__':
    # Action space
    abs_zero = 273.15
    m_DOT_F_MIN, m_DOT_F_MAX = 0, 30000  # [kgh^-1]
    T_IN_M_MIN, T_IN_M_MAX = 60 + abs_zero, 100 + abs_zero  # [Kelvin]
    T_IN_AWT_MIN, T_IN_AWT_MAX = 60 + abs_zero, 100 + abs_zero  # [Kelvin]

    action_min = [m_DOT_F_MIN, T_IN_M_MIN, T_IN_AWT_MIN]
    action_max = [m_DOT_F_MAX, T_IN_M_MAX, T_IN_AWT_MAX]

    env = PolyReactor()
    agent = RandomAgent(state_dim=10, action_min=action_min, action_max=action_max)

    state = env.reset()

    state_trajectory = []
    action_trajectory = []

    # while True:
    for _ in range(10000):
        action = agent.get_action(state)
        next_state, _, done, _ = env.step(action)

        state_trajectory.append(next_state)
        action_trajectory.append(action)

        if done:
            break

    # plotting state
    state_dim = env.observation_space.shape[0]

    full_trajectory = np.array(state_trajectory)
    fig, axs = plt.subplots(state_dim, 1, figsize=(5, 10))
    # fig, axs = plt.subplots(state_dim, 1, figsize=(10, 25))

    for i, ax in enumerate(axs):
        ax.plot(full_trajectory[:, i])

    plt.savefig('state.png')

    # plotting action
    action_dim = env.action_space.shape[0]
    action_trajectory = np.array(action_trajectory)

    fig, axs = plt.subplots(action_dim, 1, figsize=(5, 10))

    for i, ax in enumerate(axs):
        ax.plot(action_trajectory[:, i])

    plt.savefig('action.png')
