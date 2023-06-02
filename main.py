import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from Utils.custom_logger import CustomLogger
from SimulacionProceso.odessol import SistemaLineal
from SimulacionProceso.circuito import CircuitoPasaBajos

class Simulacion:
    
    def __init__(self, R=100, L=375e-3, C=75e-6, dt=0.001) -> None:
        self.logger = CustomLogger(__name__)
        self.circuito = CircuitoPasaBajos(R, L, C)

    def simular_planta(self, entrada, y0, dt=0.1, metodo='rk4'):
        self.logger.info("Simulando Planta...")
        self.planta = SistemaLineal('ft', self.circuito.num, self.circuito.den)
        datos_simulacion = self.planta.simular(entrada, y0, dt, metodo)
        self.logger.info("Planta Simulada")
        return datos_simulacion

    def simular_proceso(self, entrada, y0, dt=0.1, metodo='rk4', ruido=0.1):
        self.logger.info("Simulando Proceso...")
        self.logger.debug("Generando datos reales...")
        datos_reales = self.simular_planta(entrada, y0, dt, metodo)
        self.logger.debug(f"Datos reales generados y ahora se le agregara ruido del proceso con std de {ruido}...")
        datos_reales_ruido = datos_reales + np.random.normal(0, ruido, len(datos_reales))
        # Ahora se debe pasar el proceso de medida
        
        # Aplicar kalman
        
        # Aplicar Metricas
        
        # Graficar

if __name__ == "__main__":
    dt = 0.001
    tiempo = np.arange(0, 0.2, dt)
    entrada = np.zeros_like(tiempo)
    entrada[0:2] = 1
    simulacion = Simulacion()
    datos_simulacion = simulacion.simular_planta(entrada, 0, dt, 'rk4')
    print(datos_simulacion)
    plt.plot(tiempo, datos_simulacion)
    plt.title('Respuesta al escalon')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.show()