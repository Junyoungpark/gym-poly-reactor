import gym
import numpy as np
from gym import spaces

from gym_poly_reactor.envs.get_do_mpc_model import get_simulator

# TODO: Put original authors and link

ABS_ZERO = 273.15
T_SET = 90.0 + ABS_ZERO  # [Kelvin]
temp_range = 2.0

# State spaces and initial states
m_W_INIT, m_W_MIN, m_W_MAX = 10000, 0, np.inf  # [kg]
m_A_INIT, m_A_MIN, m_A_MAX = 853, 0, np.inf  # [kg]
m_P_INIT, m_P_MIN, m_P_MAX = 26.5, 0, np.inf  # [kg]

T_R_INIT, T_R_MIN, T_R_MAX = 90.0 + ABS_ZERO, T_SET - temp_range, T_SET + temp_range  # [Kelvin]
T_S_INIT, T_S_MIN, T_S_MAX = 90.0 + ABS_ZERO, ABS_ZERO, 100 + ABS_ZERO  # [Kelvin]
T_M_INIT, T_M_MIN, T_M_MAX = 90.0 + ABS_ZERO, ABS_ZERO, 100 + ABS_ZERO  # [Kelvin]
T_EK_INIT, T_EK_MIN, T_EK_MAX = 35.0 + ABS_ZERO, ABS_ZERO, 100 + ABS_ZERO  # [Kelvin]
T_AWT_INIT, T_AWT_MIN, T_AWT_MAX = 35.0 + ABS_ZERO, ABS_ZERO, 100 + ABS_ZERO  # [Kelvin]
T_adiab_INIT, T_adiab_MIN, T_adiab_MAX = 378.04682, ABS_ZERO, 109 + ABS_ZERO  # [Kelvin]
m_acc_F_INIT, m_acc_F_MIN, m_acc_F_MAX = 300, 0, 30000  # [kg]

# Action space
m_DOT_F_MIN, m_DOT_F_MAX = 0, 30000  # [kgh^-1]
T_IN_M_MIN, T_IN_M_MAX = 60 + ABS_ZERO, 100 + ABS_ZERO  # [Kelvin]
T_IN_AWT_MIN, T_IN_AWT_MAX = 60 + ABS_ZERO, 100 + ABS_ZERO  # [Kelvin]


class PolyReactor(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        obs_low = np.array([m_W_MIN, m_A_MIN, m_P_MIN,
                            T_R_MIN, T_S_MIN, T_M_MIN,
                            T_EK_MIN, T_AWT_MIN, T_adiab_MIN, m_acc_F_MIN])

        obs_high = np.array([m_W_MAX, m_A_MAX, m_P_MAX,
                             T_R_MAX, T_S_MAX, T_M_MAX,
                             T_EK_MAX, T_AWT_MAX, T_adiab_MAX, m_acc_F_MAX])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        act_low = np.array([m_DOT_F_MIN, T_IN_M_MIN, T_IN_AWT_MIN])
        act_high = np.array([m_DOT_F_MAX, T_IN_M_MAX, T_IN_AWT_MAX])

        self.action_space = spaces.Box(low=act_low, high=act_high)

        self.state = None
        self.simulator = None

        self.m_p_old = m_P_INIT
        self.m_p_new = None
        self.T_adiab = T_adiab_INIT

    def step(self, action: np.array):

        if action.ndim == 1:
            action = np.reshape(action, (-1, 1))

        state_new = self.simulator.make_step(action)
        self.state = state_new

        self.m_p_new = state_new[2]  # will be used for checking safety conditions
        self.T_adiab = state_new[-1]  # will be used for checking safety conditions

        done, done_reward = self.check_done()
        stage_reward = self.m_p_new - self.m_p_old
        reward = stage_reward + done_reward

        return np.array(state_new).squeeze(), reward, done, {}

    def reset(self, random_init=False):
        self.simulator = get_simulator()
        self.state = np.array(self.simulator.x0.master)
        return self.state

    def render(self, mode='human'):
        raise NotImplementedError("This environment does not support rendering")

    def check_done(self):
        done_state = np.abs(self.m_p_new - 20680) <= 1.0
        done_safety = self.T_adiab >= 109 + ABS_ZERO
        done = done_state or done_safety

        if done_safety:
            done_reward = -1000
        elif done_state:
            done_reward = 1000
        else:
            done_reward = 0

        return done, done_reward


if __name__ == '__main__':
    env = PolyReactor()
    s0 = env.reset()
    ns, r, done, _ = env.step(env.action_space.sample())
    print(ns)
