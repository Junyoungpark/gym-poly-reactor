import do_mpc
from casadi import exp

from gym_poly_reactor.envs.params import *


def get_simulator():
    model = do_mpc.model.Model('continuous')

    # set uncertain variables
    delH_R = model.set_variable('_p', 'delH_R')
    k_0 = model.set_variable('_p', 'k_0')

    # set state variables
    m_W = model.set_variable('_x', 'm_W')
    m_A = model.set_variable('_x', 'm_A')
    m_P = model.set_variable('_x', 'm_P')
    T_R = model.set_variable('_x', 'T_R')
    T_S = model.set_variable('_x', 'T_S')
    Tout_M = model.set_variable('_x', 'Tout_M')
    T_EK = model.set_variable('_x', 'T_EK')
    Tout_AWT = model.set_variable('_x', 'Tout_AWT')
    accum_monom = model.set_variable('_x', 'accum_monom')
    T_adiab = model.set_variable('_x', 'T_adiab')

    # set control variables
    m_dot_f = model.set_variable('_u', 'm_dot_f')
    T_in_M = model.set_variable('_u', 'T_in_M')
    T_in_EK = model.set_variable('_u', 'T_in_EK')

    # algebraic equations
    U_m = m_P / (m_A + m_P)
    m_ges = m_W + m_A + m_P
    k_R1 = k_0 * exp(- E_a / (R * T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
    k_R2 = k_0 * exp(- E_a / (R * T_EK)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
    k_K = ((m_W / m_ges) * k_WS) + ((m_A / m_ges) * k_AS) + ((m_P / m_ges) * k_PS)

    # Differential equations
    dot_m_W = m_dot_f * w_WF
    model.set_rhs('m_W', dot_m_W)
    dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A - ((m_A * m_AWT) / (m_W + m_A + m_P)))) - (
            p_1 * k_R2 * (m_A / m_ges) * m_AWT)
    model.set_rhs('m_A', dot_m_A)
    dot_m_P = (k_R1 * (m_A - ((m_A * m_AWT) / (m_W + m_A + m_P)))) + (p_1 * k_R2 * (m_A / m_ges) * m_AWT)
    model.set_rhs('m_P', dot_m_P)

    dot_T_R = 1. / (c_pR * m_ges) * (
            (m_dot_f * c_pF * (T_F - T_R)) - (k_K * A_tank * (T_R - T_S)) - (fm_AWT * c_pR * (T_R - T_EK)) + (
            delH_R * k_R1 * (m_A - ((m_A * m_AWT) / (m_W + m_A + m_P)))))
    model.set_rhs('T_R', dot_T_R)
    model.set_rhs('T_S', 1. / (c_pS * m_S) * ((k_K * A_tank * (T_R - T_S)) - (k_K * A_tank * (T_S - Tout_M))))
    model.set_rhs('Tout_M',
                  1. / (c_pW * m_M_KW) * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K * A_tank * (T_S - Tout_M))))
    model.set_rhs('T_EK', 1. / (c_pR * m_AWT) * ((fm_AWT * c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (
            p_1 * k_R2 * (m_A / m_ges) * m_AWT * delH_R)))
    model.set_rhs('Tout_AWT',
                  1. / (c_pW * m_AWT_KW) * ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK))))
    model.set_rhs('accum_monom', m_dot_f)
    model.set_rhs('T_adiab', delH_R / (m_ges * c_pR) * dot_m_A - (dot_m_A + dot_m_W + dot_m_P) * (
            m_A * delH_R / (m_ges * m_ges * c_pR)) + dot_T_R)

    model.setup()

    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': 50.0 / 36000.0
    }

    simulator.set_param(**params_simulator)

    p_num = simulator.get_p_template()
    tvp_num = simulator.get_tvp_template()

    # uncertain parameters
    p_num['delH_R'] = 950  # * np.random.uniform(1, 1)
    p_num['k_0'] = 7  # * np.random.uniform(1, 1)

    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)
    simulator.setup()

    # Set the initial state of the controller and simulator:
    # assume nominal values of uncertain parameters as initial guess
    delH_R_real = 950.0
    # c_pR = 5.0

    # x0 is a property of the simulator - we obtain it and set values.
    x0 = simulator.x0

    x0['m_W'] = 10000.0
    x0['m_A'] = 853.0
    x0['m_P'] = 26.5

    x0['T_R'] = 90.0 + 273.15
    x0['T_S'] = 90.0 + 273.15
    x0['Tout_M'] = 90.0 + 273.15
    x0['T_EK'] = 35.0 + 273.15
    x0['Tout_AWT'] = 35.0 + 273.15
    x0['accum_monom'] = 300.0
    x0['T_adiab'] = x0['m_A'] * delH_R_real / ((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']

    simulator.x0 = x0
    return simulator
