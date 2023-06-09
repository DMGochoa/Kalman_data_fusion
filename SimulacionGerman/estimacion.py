import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# Use json.load to load the dictionary from a file
with open('./SimulacionGerman/rtd_ntc.json', 'r') as f:
    rtd_ntc = json.load(f)

# Load the data from csv
data = pd.read_csv('./SimulacionGerman/simulation_data.csv')

Vdaq_rtd = data['Vdaq_rtd'].values
Vdaq_ntc = data['Vdaq_ntc'].values
Vdaq_par = data['Vdaq_par'].values
ADC_rtd = data['ADC_rtd'].values
ADC_ntc = data['ADC_ntc'].values
ADC_par = data['ADC_par'].values

# Defining the constants
ntc = rtd_ntc['ntc']
rtd = rtd_ntc['rtd']

T_amb = 25

# Reconstrucción del termopar
tipoJ = np.array([0.503811878150E-1,
                  0.304758369300E-4,
                  -0.856810657200E-7,
                  0.132281952950E-9,
                  -0.170529583370E-12,
                  0.209480906970E-15,
                  -0.125383953360E-18,
                  0.156317256970E-22])

E = lambda T: np.dot(tipoJ, [T**i for i in range(1, 9)])

invsJ = np.array([1.978425E1,
                  -2.001204E-1,
                  1.036969E-2,
                  -2.549687E-4,
                  3.585153E-6,
                  -5.344285E-8,
                  5.099890E-10])

T90 = lambda E: np.dot(invsJ, [E**i for i in range(1, 8)])

# Compensación de unión fria
Vm_par = (1/500)*(Vdaq_par + 9)*1E3 - E(T_amb)  # en milivoltios.
T_par = T90(Vm_par)

plt.figure(11)
plt.plot(T_par)

# Reconstrucción de la NTC
Vm_ntc = (1/1.3)*(Vdaq_ntc + 5)
R_ntc = (Vm_ntc * ntc['R_lim'])/(ntc['Vcc'] - Vm_ntc)
T_ntc = 1/((1/ntc['B'])*np.log(R_ntc/ntc['Ro'])+(1/ntc['To'])) - 273.15  # en Celsius.

plt.plot(T_ntc)

# Reconstrucción del RTD
Vm_rtd = (1/20)*(Vdaq_rtd + 10)
R_rtd = (Vm_rtd * rtd['R_lim'])/(rtd['Vcc'] - Vm_rtd)
T_rtd = (1/(rtd['Ro'] * rtd['alpha']))*(R_rtd - rtd['Ro'])

plt.plot(T_rtd)

# Actualizar la leyenda y el título
plt.legend(['T TERMOCUPLA','T NTC','T RTD'])
plt.title('Temperaturas Recuperadas de la Adquisición')
plt.grid(True)
plt.show()

# Inicialización
xk = np.zeros((3, 5401))
xk[:, 0] = [0, 0, 1]
Pk = np.zeros((3, 3, 5401))
Pk[:, :, 0] = np.eye(3)

# Covarianza de Proceso Q:
sigma_wk = 0.5  # desviación del proceso
Q = lambda dt: np.array([[dt**2, 0, 0], [0, dt**2, 0], [0, 0, 0]]) * sigma_wk**2

# Covarianza de Medida. Es un escalar. Se toma uno a la vez
R = np.array([3.0, 5.5, 3.0])**2  # desviaciones máximas esperadas en cada sensor.

# Matrices del modelo de espacio de estado:
# pendientes ideales del proceso:
dT = lambda t: [1/3, 0, 1/3, 0, 1/6, 0, -5/18, -5/18, -5/18, -5/18][t//600]
# matriz A:
matrizA = lambda dt, t: np.array([[1, 0, dT(t)*dt],
                                  [dt/rtd['tau'], 1-dt/rtd['tau'], 0], [0, 0, 1]])
# Ecuaciones de salida para cada sensor. Se toma solo una fila a la vez.
Hcomb = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]])

# Correr el proceso
last_t = 0
for t in range(1, 5400):
    
    # ETAPA DE ESTIMACION. A PRIORI.
    # calcular transicion para el dt = t - last_t:
    A = matrizA(t-last_t, t)
    # predecir estado siguiente
    xk[:, t] = np.dot(A, xk[:, last_t])  # Python es 0-indexado. ==> last_t.
    # proyectar el error al estado siguiente
    Pk[:, :, t] = np.dot(np.dot(A, Pk[:, :, last_t]), A.T) + Q(t-last_t)
    
    # ETAPA DE ACTUALIZACIÓN. A POSTERIORI.
    sensores = [T_rtd[t-1], T_ntc[t-1], T_par[t-1]]  # Python es 0-indexado
    for i in range(3):
        # llega el dato proveniente del sensor i.
        zk = sensores[i]
        # evaluar z-score:
        zscore = abs(zk - np.dot(Hcomb[i, :], xk[:, t]))/np.sqrt([Pk[2, 2, t], Pk[1, 1, t], Pk[1, 1, t]][i])
        
        if zscore < 3:    # 3 veces la desviación ya es atípico
            # que sensor es? H es una fila de 3 elementos.
            H = Hcomb[i, :]
            # Actualizar la ganancia de Kalman y el estado A POSTERIORI.
            Kk = np.dot(np.dot(Pk[:, :, t], H.T), 1 / (np.dot(np.dot(H, Pk[:, :, t]), H.T) + R[i]))
            xk[:, t] = xk[:, t] + np.dot(Kk, (zk - np.dot(H, xk[:, t])))
            Pk[:, :, t] = np.dot((np.eye(3) - np.dot(Kk, H)), Pk[:, :, t])
            # la ultima vez que se actualizo el estado fue en:
            last_t = t

# Resultado de la estimacion. para graficar todo al final.
fig, ax = plt.subplots()
ax.plot(xk[0:2, :].T, linewidth=1)
ax.grid()
ax.legend(['Te', 'Ti'])

# graficar intervalo de incertidumbre para Te.
sigma_te = np.sqrt(Pk[0, 0, :])
ax.plot(xk[0, :] + 4 * sigma_te, 'k--')
ax.plot(xk[0, :] - 4 * sigma_te, 'k--')
ax.set_title('Resultado de la Estimación')
plt.show()
