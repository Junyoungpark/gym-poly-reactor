from copy import deepcopy as dc

import gym
import numpy as np
from gym import spaces

from gym_poly_reactor.envs.params import *  # import parameters of the simulator

# TODO: Put original authors and link
T_SET = 90.0  # [celsius]

# State spaces and initial states
m_W_INIT, m_W_MIN, m_W_MAX = 10000, 0, np.inf  # [kg]
m_A_INIT, m_A_MIN, m_A_MAX = 853, 0, np.inf  # [kg]
m_P_INIT, m_P_MIN, m_P_MAX = 26.5, 0, np.inf  # [kg]

T_R_INIT, T_R_MIN, T_R_MAX = 90.0, T_SET - 2.0, T_SET + 2.0  # [celsius]
T_S_INIT, T_S_MIN, T_S_MAX = 90.0, 0, 100  # [celsius]
T_M_INIT, T_M_MIN, T_M_MAX = 90.0, 0, 100  # [celsius]
T_EK_INIT, T_EK_MIN, T_EK_MAX = 35.0, 0, 100  # [celsius]
T_AWT_INIT, T_AWT_MIN, T_AWT_MAX = 35.0, 0, 100  # [celsius]
T_adiab_INIT, T_adiab_MIN, T_adiab_MAX = 104.897, 0, 109  # [celsius]
m_acc_F_init, m_acc_F_MIN, m_acc_F_MAX = 0, 0, 30000  # [kg]

# Action space
m_DOT_F_MIN, m_DOT_F_MAX = 0, 30000  # [kgh^-1]
T_IN_M_MIN, T_IN_M_MAX = 60, 100  # [celsius]
T_IN_AWT_MIN, T_IN_AWT_MAX = 60, 100  # [celsius]


class PolyReactor(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        obs_low = np.array([m_W_MIN, m_A_MIN, m_P_MIN,
                            T_R_MIN, T_S_MIN, T_M_MIN,
                            T_EK_MIN, T_AWT_MIN, T_adiab_MIN, m_acc_F_MIN])

        obs_high = np.array([m_W_MAX, m_A_MAX, m_P_MAX,
                             T_R_MAX, T_S_MAX, T_M_MAX,
                             T_EK_MAX, T_AWT_MAX, T_adiab_MAX, m_acc_F_MAX])

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32)

        act_low = np.array([m_DOT_F_MIN, T_IN_M_MIN, T_IN_AWT_MIN])
        act_high = np.array([m_DOT_F_MAX, T_IN_M_MAX, T_IN_AWT_MAX])

        self.action_space = spaces.Box(low=act_low, high=act_high)

        # TODO: documentation about the states and action space
        # the state order
        self.state = None
        self._eps = 1e-20
        self.tau = 50.0 / 3600.0  # 0.1  # [sec] ; hopefully

    def _get_copied_state(self):
        state = dc(self.state)
        m_w, m_a, m_p, T_R, T_S, T_M, T_EK, T_AWT, T_adiab, m_acc = np.array_split(state, 10)
        return m_w, m_a, m_p, T_R, T_S, T_M, T_EK, T_AWT, T_adiab, m_acc

    def _get_aux_variable(self):
        abs_zero = 273.15
        m_w, m_a, m_p, T_R, T_S, _, T_EK, _, _, _ = self._get_copied_state()

        # TODO: 0 division edge case handling

        U = m_p / (m_a + m_p)
        m_ges = m_w + m_a + m_p
        k_r1 = k_0 * np.exp(-E_a / (R * (T_R + abs_zero))) * (k_U1 * (1 - U) + k_U2 * U)
        k_r2 = k_0 * np.exp(-E_a / (R * (T_EK + abs_zero))) * (k_U1 * (1 - U) + k_U2 * U)
        k_K = (m_w * k_WS + m_a * k_AS + m_p * k_PS) / (m_ges)
        m_a_r = m_a - m_a * m_AWT / m_ges

        return U, m_ges, k_r1, k_r2, k_K, m_a_r

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # Update 10 states ODEs in Euler method

        m_dot_f, t_m_in, t_awt_in = action[0], action[1], action[2]

        # copy the current state to prevent numerical problems from numpy referencing.
        m_w, m_a, m_p, T_R, T_S, T_M, T_EK, T_AWT, T_adiab, m_acc = self._get_copied_state()
        U, m_ges, k_r1, k_r2, k_K, m_a_r = self._get_aux_variable()

        # 1st ODE
        m_w_dot = m_dot_f * w_WF
        m_w_new = m_w + self.tau * m_w_dot

        # 2nd ODE
        m_a_dot = m_dot_f * w_WF - k_r1 * m_a_r - k_r2 * m_AWT * m_a / m_ges
        m_a_new = m_a + self.tau * m_a_dot

        # 3rd ODE
        m_p_dot = k_r2 * m_a_r + p_1 * k_r2 * m_AWT * m_a / m_ges
        m_p_new = m_p + self.tau * m_p_dot

        # 4th ODE
        _4th_ode_mul = (
                m_dot_f * c_pF * (T_F - T_R) + delH_R * k_r1 * m_a_r - k_K * A_tank * (T_R - T_S) - fm_AWT * c_pR * (
                T_R - T_EK))
        T_R_dot = 1 / (c_pR * m_ges) * _4th_ode_mul
        T_R_new = T_R + self.tau * T_R_dot

        # 5th ODE
        T_S_dot = 1 / (c_pS * m_S) * (k_K * A_tank * (T_R + 2 * T_S + T_M))
        T_S_new = T_S + self.tau * T_S_dot

        # 6th ODE
        # TODO: double check equation
        T_M_dot = 1 / (c_pW * m_M_KW) * (fm_M_KW * c_pW * (t_m_in - T_M) + k_K * A_tank * (T_S - T_M))
        T_M_new = T_M + self.tau * T_M_dot

        # 7th ODE
        _7th_ode_mul = (fm_AWT * c_pW * (T_R - T_EK) - alfa * (T_EK - T_AWT) + k_r2 * m_a * m_AWT * delH_R / m_ges)
        T_EK_dot = 1 / (c_pR * m_AWT) * _7th_ode_mul
        T_EK_new = T_EK + self.tau * T_EK_dot

        # 8th ODE
        _8th_ode_numer = (fm_AWT_KW * c_pW * (t_awt_in - T_AWT) - alfa * (T_AWT - T_EK))
        _8th_ode_denom = (c_pW * m_AWT_KW)
        T_AWT_dot = _8th_ode_numer / _8th_ode_denom
        T_AWT_new = T_AWT + self.tau * T_AWT_dot

        # SAFETY VARIABLES
        # 9th ODE
        T_adiab_dot = delH_R / (m_ges * c_pR) * m_a_dot - (m_w_dot + m_a_dot + m_p_dot) * (
                (m_a * delH_R) / (m_ges ** 2 * c_pR)) + T_R_dot
        T_adiab_new = T_adiab + self.tau * T_adiab_dot

        # 10th ODE
        m_acc_dot = m_dot_f
        m_acc_new = m_acc + self.tau * m_acc_dot

        state_new = [m_w_new, m_a_new, m_p_new, T_R_new, T_S_new, T_M_new, T_EK_new, T_AWT_new, T_adiab_new, m_acc_new]
        self.state = np.array(state_new)
        done = self.check_done()

        # TODO: Design reward depending on the goal
        if done:
            reward = 0
        else:
            reward = m_p_new - m_p
        return np.array(state_new).squeeze(), reward, done, {}

    def reset(self, random_init=False):
        x0 = [m_W_INIT, m_A_INIT, m_P_INIT,
              T_R_INIT, T_S_INIT, T_M_INIT, T_EK_INIT,
              T_AWT_INIT, T_adiab_INIT, m_acc_F_MIN]
        x0 = np.array(x0)
        if random_init:
            x0 = np.random.uniform(low=x0 * 0.9, high=x0 * 1.1)

        self.state = x0
        return self.state

    def render(self, mode='human'):
        raise NotImplementedError("This environment does not support rendering")

    def close(self):
        pass

    def check_done(self):
        done_state = np.abs(self.state[2] - 20680) <= 1.0
        done_safety = self.state[8] >= 109
        done = done_state or done_safety
        return done
