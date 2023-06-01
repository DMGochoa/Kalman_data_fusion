import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from numpy import clip

# Constante de tiempo del RTD.
rtd = {'tau': 60.0}

# pendientes ideales del proceso:
dT = lambda t: np.array([1/3, 0, 1/3, 0, 1/6, 0, -5/18, -5/18, -5/18, -5/18])[int((t-1)/600)]

# matriz A:
matrizA = lambda dt, t: np.array([[1, 0, dT(t)*dt], [dt/rtd['tau'], 1-dt/rtd['tau'], 0], [0, 0, 1]])

# Ruido de proceso: wk ~ N(0, Q)
sigma_wk = 500*0.0005  # 0.25 C, o el 0.05% de la plena escala.
wk = lambda desv: np.vstack([norm.rvs(0, desv, size=(2, 1)), 0])

# Ruidos de Medida: vk ~ N(0, R)
outlier = lambda T: T*1.1 if np.random.rand()<0.0005 else 0
v_rtd = lambda Ti: norm.rvs(0, 0.01) + norm.rvs(0, 0.005*abs(Ti)) + outlier(Ti)  # PT100
v_ntc = lambda Te: norm.rvs(0, 5.5) + outlier(Te)  # 5.5 C @ 500
v_par = lambda Te: norm.rvs(0, 1.5) + norm.rvs(0, 0.004*abs(Te)) + outlier(Te)  # J class 1.
vk = lambda Te, Ti: np.array([v_rtd(Ti), v_ntc(Te), v_par(Te)])

# Orden de los sensores: RTD, NTC, PAR.
H = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]])

xk = np.zeros((3, 5401))
xk[:, 0] = np.array([0, 0, 1])

zk = np.zeros((3, 5401))
zk[:, 0] = H @ xk[:, 0]

# Correr el proceso
for t in range(5400):
    # Ecuacion dinamica de Espacio de Estado: x(k+1) = Ax(k) + wk
    xk[:, t+1] = matrizA(1, t) @ xk[:, t] + wk(sigma_wk).flatten()
    # Ecuacion de salida para la simulacion:
    zk[:, t+1] = H @ xk[:, t+1] + vk(xk[1, t+1], xk[2, t+1])

# graficar T_rtd, T_ntc, T_par
plt.figure(2)
plt.plot(range(5401), zk.T, '.-')
plt.grid(True)
plt.title('Temperaturas Medidas por los Sensores con ATÍPICOS')

# simular el RTD
rtd.update({'Ro': 100, 'alpha': 0.0039})
R_rtd = rtd['Ro'] + rtd['alpha']*rtd['Ro']*zk[0, :]
# simular el acondicionamiento del RTD
rtd.update({'Vcc': 10, 'R_lim': 4000})    # voltaje de polarizacion, resistencia limitadora de corriente
Vm_rtd = (rtd['Vcc']*R_rtd)/(rtd['R_lim'] + R_rtd)  # divisor de tension
print(f'Vm_rtd: min= {min(Vm_rtd)} \tmax= {max(Vm_rtd)}')  # maximos y minimos
Vdaq_rtd = clip(-10 + 20*Vm_rtd, -10, 10)   # DAQ in [-10, +10]

# simular el NTC
ntc = {'Ro': 821970, 'To': 25 + 273.15, 'B': 4113}  # [K]
T_ntc = zk[1, :] + 273.15  # [K]
R_ntc = ntc['Ro'] * np.exp(ntc['B']*((1./T_ntc)-(1/ntc['To'])))
# Simular el acondicionamiento del NTC
ntc.update({'R_lim': 10000, 'Vcc': 10})
Vm_ntc = (ntc['Vcc']*R_ntc)/(ntc['R_lim'] + R_ntc)
print(f'Vm_ntc: min= {min(Vm_ntc)} \tmax= {max(Vm_ntc)}')
Vdaq_ntc = clip(-5 + 1.3*Vm_ntc, -10, 10)

# simular el TermoPar.  Coefficientes directos.
# https://srdata.nist.gov/its90/type_j/jcoefficients.html
tipoJ = np.array([0.503811878150E-1,
         0.304758369300E-4,
        -0.856810657200E-7,
         0.132281952950E-9,
        -0.170529583370E-12,
         0.209480906970E-15,
        -0.125383953360E-18,
         0.156317256970E-22
])
E = lambda T: tipoJ @ np.array([T, T**2, T**3, T**4, T**5, T**6, T**7, T**8])

# Temperatura simulada de la UNION FRIA.
T_amb = 25   # que calor!.  A quien se le ocurrió decir que 25 era Standard ?

# A la Temperatura del termopar se le agrega la union fria (que es como caliente !)
T_par = zk[2, :]  
Vm_par = (E(T_par) + E(T_amb)) * 1E-3
print(f'Vm_par: min= {min(Vm_par)} \tmax= {max(Vm_par)}')
Vdaq_par = clip(-5 + 1.3*Vm_par, -10, 10)

# Graficar las salidas de los acondicionadores
plt.figure(3)
plt.plot(range(5401), np.vstack([Vdaq_rtd, Vdaq_ntc, Vdaq_par]).T, '.-')
plt.grid(True)
plt.title('Salida de los acondicionadores')

# Simular el DAQ
ADC_bits = 16
ADC_full = 2**ADC_bits
ADC_LSB = 20. / ADC_full
ADC_rtd = (Vdaq_rtd + 10) / ADC_LSB
ADC_ntc = (Vdaq_ntc + 10) / ADC_LSB
ADC_par = (Vdaq_par + 10) / ADC_LSB

# Convierte los datos a un DataFrame de pandas
simulation_data_df = pd.DataFrame({
    'Vdaq_rtd': Vdaq_rtd,
    'Vdaq_ntc': Vdaq_ntc,
    'Vdaq_par': Vdaq_par,
    'ADC_rtd': ADC_rtd,
    'ADC_ntc': ADC_ntc,
    'ADC_par': ADC_par
})
# Guardar los datos en un archivo CSV
simulation_data_df.to_csv('simulation_data.csv', index=False)
plt.show()


