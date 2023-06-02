"""
Modulo en el que se van a crear tres  sensores
"""
import numpy as np

def rtd(R0=100, alpha=0.00385, T_i=np.array([])):
    """En esta funcion se simula una RTD pero sus paramaetros estan predeterminados para una 
    PT100 en donde R0 es 100 y alpha es una valor dado por la norma ANSI.

    Args:
        R0 (int, optional): Valor referencia de la resistencia. Defaults to 100.
        alpha (float, optional): Valor de alpha que esta determinado por la norma 
        ANSI. Defaults to 0.00385.
        T_i (np.array, optional): Arreglo de datos en donde se tiene la temperatura 
        para cada instante de tiempo. Defaults to np.array([]).
    Returns:
        np.array: Retorna un arreglo de datos de R.
    """
    print(R0, alpha)
    R = (alpha * R0 * T_i) + R0
    return R

def ntc(T_i=np.array([])):
    """En esta funcion se simula una NTC pero sus parametros esta predeterminados a 25°C y se hace una
    estimacion del parametro B por dentro de esta.

    Args:
        T_i (np.array, optional): Arreglo de datos de temperatura. Defaults to np.array([]).

    Returns:
        np.array: Retorna un arreglo de datos de R.
    """
    # Se pasa T_i a kelvin
    T_i = T_i + 273
    
    # Parametros de referencia
    R0 = 821970
    T0 = 25
    
    # Intervalos
    T_interval = [[-40,  100], 
                  [100, 200], 
                  [200, 300], 
                  [300, 400],
                  [400, 500],
                  [500, 620]]
    T_interval = np.array(T_interval) + 273
    R =[59902, 5000, 826.76, 207.35, 68.619, 28.051]
    
    # Estimación de los B con sus respectivos intervalos
    B = {i:[np.log(R[i]/R0)/(1/T_interval[i][1] - 1/T0), T_interval[i]] for i in range(len(T_interval))}
    
    # Declaro el arreglo que va contener los resultados de las resistencias
    R = np.zeros(len(T_i))
    for interval in B.keys():
        values = B[interval]
        # Se realiza el filtro a partir de los datos que estan guardados en B que estan de la forma
        # {num_inter: [B [lim_inf, lim_sup]]}
        filtro = (values[1][0] <= T_i) & (T_i < values[1][1])
        # R = R0 * exp(B(1/T - 1/T0))
        #print('\n', filtro , T_i[filtro], '\n')
        R[filtro] = R0 * np.exp(values[0] * (1/T_i[filtro] - 1/T0))  
    return R

def termocupla(T_i=np.array([])):
    """Como los coeficientes de la termocupla son muy pequeños, podemos tomar solo un parametro y
    como lineal.

    Args:
        T_i (np.array, optional): Arreglo de datos de temperatura. Defaults to np.array([]).
    """
    a = [5.04e1, 3.05e-2, -8.57e-5, 1.32e-7, -1.71e-10, 2.09e-13, -1.25e-16, 1.56e-20]
    V = a[0]*T_i**1 + a[1]*T_i**2 + a[2]*T_i**3 + a[3]*T_i**4 \
      + a[4]*T_i**5 + a[5]*T_i**6 + a[6]*T_i**7 + a[7]*T_i**8
    # V = T/5.04
    return V

if __name__ == '__main__':
    # Se define una prueba de temperatura siguiendo una linea.
    m = 20
    t = np.linspace(0, 10, 10)
    T = m * t
    print('='*32 + ' RTD ' + '='*32)
    print(rtd(T_i=T))
    print('='*32 + ' NTC ' + '='*32)
    print(ntc(T_i=T))
    print('='*32 + ' TERMOCUPLA ' + '='*32)
    print(termocupla(T_i=T))
    #
    #