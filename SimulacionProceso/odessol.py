import sys
sys.path.append("./")
import numpy as np
import matplotlib.pyplot as plt
from Utils.custom_logger import CustomLogger


class SistemaLineal:
    """
    Esta clase permite declarar un sistema lineal o ecuación diferencial
    usando alguno de los tipos de representación:
    'ft' : función de transferencia
    'ee' : espacio de estados
    Permite visualizar la representación del sistema, encontrar la solución
    numérica y graficarla, así como observar la respuesta de un sistema
    gobernado por esta función de transferencia ante una entrada de algún
    tipo estandar:
    'escalon'
    'impulso'
    """

    def __init__(self, tiposistema='', *args):
        """
        Inicialización de la ecuación diferencial
        :param tiposis: 'ft': funcion de transferencia
        'ee': espacio de estados

        :param args: (num, den) para función de transferencia FORMATO LISTA
        (A, B, C, D) para modelo de estados (Forma canonica observable) FORMATO LISTA

        """
        self.logger = CustomLogger(__name__)
        self.tipo = tiposistema  # Representacion del tipo del sistema
        self.parametros = dict()
        if self.tipo == 'ft':
            self.logger.debug("Creating a transfer function")
            self.parametros['num'] = np.array(args[0])
            self.logger.debug(f"Numerator: {self.parametros['num']}")
            self.parametros['den'] = np.array(args[1])
            self.logger.debug(f"Denominator: {self.parametros['den']}")
            self.orden = len(args[0])
            self.logger.debug(f"Order: {self.orden}")
            self.ee(False)
        elif self.tipo == 'ee':
            self.logger.debug("Creating a state space model")
            self.parametros['A'] = np.array(args[0])
            self.logger.debug(f"A matrix: {self.parametros['A']}")
            self.parametros['B'] = np.array(args[1])
            self.logger.debug(f"B matrix: {self.parametros['B']}")
            self.parametros['C'] = np.array(args[2])
            self.logger.debug(f"C matrix: {self.parametros['C']}")
            #self.parametros['D'] = np.array(args[3])
            self.orden = self.parametros['A'].shape[0]
            self.logger.debug(f"Order: {self.orden}")
        else:
            raise NameError('El tipo de EDO no es valido')

    def __getInitial__(self, y0):
        """
        Es para verificar que haya un vector apropiado de estados 
        iniciales es decir que coincida con el orden del sistema

        :param y0: Valor de estados inicales
        :return: Vector numpy ya coincidiendo con orden
        """

        try:
            self.logger.debug(f"Initial state: {y0}")
            iter(y0)
            if len(y0) >= self.orden:
                self.logger.debug("Initial state vector is greater than or equal to the order of the system")
                x0 = np.array(y0[0:self.orden])
            else:
                self.logger.debug("Initial state vector is less than the order of the system")
                x0 = np.concatenate(np.array(y0), np.zeros(self.orden - len(y0)))
        except:
            self.logger.debug("Initial state is a scalar")
            x0 = np.zeros(self.orden)
            x0[0] = y0
        salida = x0.reshape((self.orden, 1))
        self.logger.debug(f"Initial state vector: {salida}")
        return salida

    def simular(self, entrada, y0, dt=0.1, metodo='rk4'):
        """
        Simulacion del comportamiento de un sistema usando las tecnicas mostradas en documento

        :param entrada: Entrada del sistema, este debe ser un vector
        :param y0: Valor inicial de la salida del sistema
        :param dt: pasos del tiempo
        :param metodo: Metodos que se encuentran disponibles
            'Feuler': Forward Euler
            'Beuler': Backward Euler
            'Tustin': Tustin
            'rk4': Runge Kutta
        :return: Vector de salida
        """

        y0 = self.__getInitial__(y0)  # Para definir el y0 adecuado
        self.logger.debug(f"Initial state vector: {y0}")
        # Se escoger el metodo requerido
        if metodo == 'Feuler':
            self.logger.debug("Using Forward Euler method")
            salida = self.forward_euler(entrada, y0, dt)
        elif metodo == 'Beuler':
            self.logger.debug("Using Backward Euler method")
            salida = self.backward_euler(entrada, y0, dt)
        elif metodo == 'Tustin':
            self.logger.debug("Using Tustin method")
            salida = self.tustin(entrada, y0, dt)
        elif metodo == 'rk4':
            self.logger.debug("Using Runge-Kutta method")
            salida = self.rk(entrada, y0, dt)
        else:
            self.logger.error("Invalid method")
            raise NameError('Método no valido')
        return salida

    def impulso(self, amp=1, dt=0.1, t_final=10.0, y0=0, metodo='rk4'):
        """
        Simulacion con impulso y devuelve gráfica resultante

        :param dt: pasos del tiempo
        :param t_final: Tiempo final
        :param y0: Valor inicial de la salida del sistema
        :param metodo: Metodos que se encuentran disponibles
            'Feuler': Forward Euler
            'Beuler': Backward Euler
            'Tustin': Tustin
            'rk4': Runge Kutta
        """
        tiempo = np.arange(0, t_final, dt)
        entrada = np.zeros_like(tiempo)
        entrada[0] = 1 * amp
        salida = self.simular(entrada, y0, dt, metodo)
        plt.plot(tiempo, salida)
        plt.title('Respuesta al impulso')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.grid()
        plt.show()

    def escalon(self, amp=1, dt=0.1, t_final=10.0, y0=0, metodo='rk4'):
        """
        Simulacion con escalon  y devuelve gráfica resultante

        :param dt: pasos del tiempo
        :param amp: amplitud del escalon
        :param t_final: Tiempo final
        :param y0: Valor inicial de la salida del sistema
        :param metodo: Metodos que se encuentran disponibles
            'Feuler': Forward Euler
            'Beuler': Backward Euler
            'Tustin': Tustin
            'rk4': Runge Kutta
        """
        tiempo = np.arange(0, t_final, dt)
        entrada = amp * np.ones_like(tiempo)
        entrada[0] = 0
        salida = self.simular(entrada, y0, dt, metodo)
        plt.plot(tiempo, salida)
        plt.title('Respuesta al escalon')
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.grid()
        plt.show()

    def ft(self, usuario=True):
        """
        Dado un espacio de estado cambiar a funcion de transferencia siempre que se ingrese de la forma canónica
        observable
        :return: Nada (Se hace de forma interna en el objeto)
        """
        list_num = []
        list_den = []

        if self.tipo == 'ee':
            if usuario:
                self.tipo = 'ft'
            for i in range(len(self.parametros['A'])):
                list_den.append(self.parametros['A'][i,0])
                list_num.append(self.parametros['B'][i, 0])
            self.parametros['num'] = np.array(list_num)
            self.parametros['den'] = -np.array(list_den)
        #print(self.parametros['num'], self.parametros['den'])

    def ee(self, usuario=True):
        """
        Este método convierte la representación del sistema a
        un modelo de espacio de estados.
        :return: Nada (Se hace de forma interna en el objeto)
        """
        if self.tipo == 'ft':
            if usuario:
                self.tipo = 'ee'
            # Corresponde a la conversión de la forma Observable
            self.parametros.update({'A': np.diag(np.ones(self.orden - 1), k=1)})
            self.parametros['A'][:, 0] = - self.parametros['den']
            self.parametros.update({'B': self.parametros['num'].reshape((self.orden, 1))})
            self.parametros.update({'C': np.zeros(self.orden)})
            self.parametros['C'][0] = 1
            #print(self.parametros['A'], '\n', self.parametros['B'], '\n', self.parametros['C'])

    def backward_euler(self, entrada, y0, dt):
        """
        Este método calcula la salida del sistema ante la entrada 'entrada'
        usando el método de diferencias hacia atras de Euler:
        :param entrada: la señal de entrada
        :param y0: estado inicial
        :param dt: delta de tiempo
        :return: sal_y: La salida del sistema en el dominio indicado por 'u' y 'h'
        """

        ma = self.parametros['A']
        mb = self.parametros['B']
        mc = self.parametros['C']
        vec_x = y0
        sal_y = np.zeros_like(entrada)
        for i in range(len(entrada)):
            sal_y[i] = mc @ vec_x
            vec_x = np.linalg.solve(np.eye(self.orden) - ma * dt,
                                    vec_x + mb * entrada[i] * dt)

        return sal_y

    def forward_euler(self, entrada, y0, dt):
        """
        Este método calcula la salida del sistema ante la entrada 'entrada'
        usando el método de diferencias hacia adelante de Euler:
        :param entrada: la señal de entrada
        :param y0: estado inicial
        :param dt: delta de tiempo
        :return: sal_y: La salida del sistema en el dominio indicado por 'u' y 'h'
        """

        ma = self.parametros['A']
        mb = self.parametros['B']
        mc = self.parametros['C']
        vec_x = y0
        sal_y = np.zeros_like(entrada)
        for i in range(len(entrada)):
            sal_y[i] = mc @ vec_x
            vec_x = vec_x + (ma @ vec_x + mb * entrada[i]) * dt
        return sal_y

    def tustin(self, entrada, y0, dt):
        """
        Este método calcula la salida del sistema ante la entrada 'entrada'
        usando el método de Tustin:
        :param entrada: la señal de entrada
        :param y0: estado inicial
        :param dt: delta de tiempo
        :return: sal_y: La salida del sistema en el dominio indicado por 'u' y 'h'
        """

        ma = self.parametros['A']
        mb = self.parametros['B']
        mc = self.parametros['C']
        vec_x = y0
        sal_y = np.zeros_like(entrada)
        for i in range(len(entrada)):
            sal_y[i] = mc @ vec_x
            vec_x = np.linalg.solve(np.eye(self.orden) - ma * dt / 2,
                                    vec_x + (ma @ vec_x + 2 * mb * entrada[i]) * dt / 2)
        return sal_y

    def rk(self, entrada, y0, dt):
        """
        Este método calcula la salida del sistema ante la entrada 'entrada'
        usando el método de Runge-Kutta:
        :param entrada: la señal de entrada
        :param y0: estado inicial
        :param dt: delta de tiempo
        :return: sal_y: La salida del sistema en el dominio indicado por 'u' y 'h'
        """

        ma = self.parametros['A']
        mb = self.parametros['B']
        mc = self.parametros['C']
        vec_x = y0
        sal_y = np.zeros_like(entrada)
        for i in range(len(entrada)):
            sal_y[i] = mc @ vec_x
            k1 = ma @ vec_x + mb * entrada[i]
            k2 = ma @ (vec_x + k1 * dt / 2) + mb * entrada[i]
            k3 = ma @ (vec_x + k2 * dt / 2) + mb * entrada[i]
            k4 = ma @ (vec_x + k3 * dt) + mb * entrada[i]
            vec_x = vec_x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
        return sal_y


if __name__ == '__main__':
    #sistema_RLC = SistemaLineal('ft', [0, 100], [100, 100])
    #sistema_RLC.impulso(dt=0.01,t_final=10,y0=0,metodo='rk4')


    # Pruebas para poder probar de que este funcionando IGNORAR LO DE ABAJO
    A = [[-3, 1],
         [-2, 0]]
    B = [[0],[1]]
    C = [1, 0]
    #sistema2 = SistemaLineal('ee', A, B, C)
    #sistema2.ft()
    #sistema2.impulso(dt=0.01,t_final=10,y0=0,metodo='rk4')


    sistema1 = SistemaLineal('ft', [1, 0], [1, 100])
    sistema1.escalon(dt=0.03, t_final=30, y0=0, metodo='Feuler')
