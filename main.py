import sys
sys.path.append("./")
from Filtro.kalman import Kalman
from SimulacionProceso.circuito import CircuitoPasaBajos
from SimulacionProceso.odessol import SistemaLineal
from Utils.custom_logger import CustomLogger
from Medida.intrumentationAmp import AD620
from Medida.sensor import Voltage_divider
from Medida.conditioning import diff_amp
import matplotlib.pyplot as plt
import numpy as np



class Simulacion:

    def __init__(self, R=100, L=375e-3, C=75e-6, dt=0.001) -> None:
        self.logger = CustomLogger(__name__)
        self.circuito = CircuitoPasaBajos(R, L, C)

    def simular_planta(self, entrada, y0, dt=0.1, metodo='rk4'):
        self.logger.info("Simulando Planta...")
        self.planta = SistemaLineal('ft', self.circuito.num, self.circuito.den)
        self.planta.parametros['C'] = np.array([1, 1])
        datos_simulacion = self.planta.simular(entrada, y0, dt, metodo)
        self.logger.info("Planta Simulada")
        print(datos_simulacion)
        return datos_simulacion

    def simular_planta2(self, Vin, T=1e-3, min=0, max=0.5, ruido=1e-2):
        t = np.arange(min, max + T, T)
        y = np.zeros((2, len(t)))
        rng = np.random.default_rng()
        for j in range(1, len(t)):
            y[:, j] = y[:, j-1] + T * self.circuito.model(y[:, j-1], Vin[j-1]) + \
                      rng.normal(0, ruido, (2,))
        return t, y

    def simular_proceso(self, entrada, y0, dt=0.1, metodo='rk4', ruido=0.1,
                        sensor_atributes=[{'R1': 1e5, 'R2': 1e5},
                                          {'R1': 1e5, 'R2': 9e5},
                                          {'R1': 1e5, 'R2': 1e5}]):
        self.logger.info("Simulando Proceso...")
        self.logger.debug("Generando datos reales...")
        datos_reales = self.simular_planta(entrada, y0, dt, metodo)
        self.logger.debug(
            f"Datos reales generados y ahora se le agregara ruido del proceso con std de {ruido}...")
        datos_reales_ruido = datos_reales + \
            np.random.normal(0, ruido, len(datos_reales)) * datos_reales
        maximo_esperado = np.max(datos_reales)
        minimo_esperado = np.min(datos_reales)
        # Ahora se debe pasar el proceso de medida
        self.logger.debug("Pasando por el proceso de medida...")
        self.logger.debug("Instanciando sensores...")
        sensors = [Voltage_divider(r1=espects['R1'], r2=espects['R2'])
                   for espects in sensor_atributes]
        self.logger.debug(f"Sensores instanciados {sensors}")
        self.logger.debug("Pasando por los sensores...")
        #[self.logger.debug(f"Sensor mean value: {sensor.mean_value(datos_reales_ruido)}") for sensor in sensors]
        datos_sensores = [sensor.noise(datos_reales_ruido)
                          for sensor in sensors]

        self.logger.debug("Instanciando amplificadores de instrumentacion...")
        def r(R1, R2): return R2/(R1+R2)
        def rg(r): return 49400/(1/r - 1)
        amplificadores_instrumentacion = [
            AD620(rg(r(espects['R1'], espects['R2']))) for espects in sensor_atributes]
        self.logger.debug("Amplificadores de instrumentacion instanciados")
        self.logger.debug(
            "Pasando por los amplificadores de instrumentacion...")
        datos_amplificados = [amp.output(np.array(datos_sensores[i])) for i, amp in enumerate(
            amplificadores_instrumentacion)]
        self.logger.debug("Instanciando amplificadores diferenciales...")
        amplificadores_diferenciales = [diff_amp(
            maximo_esperado, minimo_esperado) for _ in range(len(datos_amplificados))]
        self.logger.debug("Amplificadores diferenciales instanciados")
        self.logger.debug("Pasando por los amplificadores diferenciales...")
        datos_amplificados_diferenciales = [amp.output(datos_amplificados[i])
                                            for i, amp in enumerate(amplificadores_diferenciales)]
        self.logger.debug("Proceso de medida terminado")
        
        # DAQ
        datos_DAQ = [DAQ.daq(datos_amplificados_diferenciales[i]) 
                     for i, DAQ in enumerate(amplificadores_diferenciales)]

        # Aplicar kalman
        F = np.array([1])
        H = np.array([1])
        Q = np.array([8e-2])
        R = np.array([0.1])
        #F = np.array(self.planta.parametros['A'])
        #print(F)
        #H = np.array([[1, 0]])
        #Q = ((1e-5)**2)*np.array([[1, 0],[0, 1]])
        #R = ((1e-2)**2)
        filtro = Kalman(F, H, Q, R)
        filtro.primera_iter()
        salida_kalman = [[], []]
        for i in range(len(datos_DAQ[0])):
            #for medida in datos_DAQ:
            x, p = filtro.kalman(datos_DAQ[0][i], 0, 0)#, np.array([0, 0]), np.array([[1e-5, 0], [0, 1e-3]]))
            salida_kalman[0].append(x)
            salida_kalman[1].append(p)

        # Aplicar Metricas

        return {'datos_reales': datos_reales,
                'datos_reales_ruido': datos_reales_ruido,
                'datos_sensores': datos_sensores,
                'datos_amplificados': datos_amplificados,
                'datos_amplificados_diferenciales': datos_amplificados_diferenciales,
                'datos_DAQ': datos_DAQ,
                'datos_kalman_x': salida_kalman[0],
                'datos_kalman_p': salida_kalman[1]}
        # Graficar


if __name__ == "__main__":
    dt = 1e-4
    tiempo = np.arange(0, 0.05, dt)
    entrada = np.zeros_like(tiempo)
    entrada[0:2] = 5
    simulacion = Simulacion()
    datos_simulacion = simulacion.simular_proceso(entrada, 0, dt=dt, metodo='rk4', ruido=0.1)
    #simulacion.simular_planta(entrada, 0, dt, 'rk4')
    i = 1
    for datos in datos_simulacion['datos_DAQ']:
        plt.plot(tiempo, datos, '--', label=f'Sensor {i}', alpha=0.5)
        i += 1
    plt.plot (tiempo, datos_simulacion['datos_kalman_x'], '-*', label='Kalman', color='black')
    plt.plot (tiempo, datos_simulacion['datos_reales'], label='Real')
    plt.title('Respuesta al escalon')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid()
    plt.show()
