# Certain parameters
R = 8.314  # gas constant
T_F = 25 + 273.15  # feed temperature
E_a = 8500.0  # activation energy
delH_R = 950.0 * 1.00  # sp reaction enthalpy
A_tank = 65.0  # area heat exchanger surface jacket 65

k_0 = 7.0 * 1.00  # sp reaction rate
k_U2 = 32.0  # reaction parameter 1
k_U1 = 4.0  # reaction parameter 2
w_WF = .333  # mass fraction water in feed
w_AF = .667  # mass fraction of A in feed

m_M_KW = 5000.0  # mass of coolant in jacket
fm_M_KW = 300000.0  # coolant flow in jacket 300000;
m_AWT_KW = 1000.0  # mass of coolant in EHE
fm_AWT_KW = 100000.0  # coolant flow in EHE
m_AWT = 200.0  # mass of product in EHE
fm_AWT = 20000.0  # product flow in EHE
m_S = 39000.0  # mass of reactor steel

c_pW = 4.2  # sp heat cap coolant
c_pS = .47  # sp heat cap steel
c_pF = 3.0  # sp heat cap feed
c_pR = 5.0  # sp heat cap reactor contents

k_WS = 17280.0  # heat transfer coeff water-steel
k_AS = 3600.0  # heat transfer coeff monomer-steel
k_PS = 360.0  # heat transfer coeff product-steel

alfa = 5 * 20e4 * 3.6

p_1 = 1.0
