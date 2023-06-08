"""_summary_

Raises:
    TypeError: _description_
    TypeError: _description_
    AttributeError: _description_

Returns:
    _type_: _description_
"""

import numpy as np


class Voltage_divider():
    """_summary_
    """

    def __init__(self, r1, r2, tolr1 = 10, tolr2 = 10):
        """_summary_

        Args:
            r1 (_type_): _description_
            r2 (_type_): _description_
            tolr1 (int, optional): _description_. Defaults to 10.
            tolr2 (int, optional): _description_. Defaults to 10.
        """
        self.r1 = r1
        self.r2 = r2
        self.__r_value = self.r2/(self.r1+self.r2)
        self.tolr1 = tolr1
        self.tolr2 = tolr2
        self.__deltar1 = self.__calculate_delta(r1, tolr1)
        self.__deltar2 = self.__calculate_delta(r2, tolr2)
        self.__delta_r = 0
        # Output
        self.uncertaintyV = 0
        self.meanV = 0

    def mean_value(self, vin):
        """_summary_

        Args:
            vin (_type_): _description_
        """
        if type(vin) == int or type(vin) == float or type(vin) == np.ndarray:
            self.meanV = vin*self.__r_value
        else: 
            raise TypeError("The type of argument accepted are float or numpy array")

    def noise(self, vin, mode = 'most probably'):
        """_summary_

        Args:
            vin (_type_): _description_
            mode (str, optional): _description_. Defaults to 'most probably'.
        """
        # The uncertainty is standard deviation
        if type(vin) == int or type(vin) == float:
            mu, sigma = 0, self.__uncertainty_r(mode) # mean and standard deviation
            self.uncertaintyV = vin*np.random.normal(mu, sigma, 1)[0]
        elif  type(vin) == np.ndarray:
            n = len(vin)
            mu, sigma = 0, self.__uncertainty_r(mode) # mean and standard deviation
            self.uncertaintyV = vin*np.random.normal(mu, sigma, n)
        else: 
            raise TypeError("The type of argument accepted are int, float or numpy array")
            
    def __calculate_delta(self, r, delta_r):
        """_summary_

        Args:
            r (_type_): _description_
            delta_r (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (delta_r/100)*r

    def __uncertainty_r(self, mode = 'most probably'):
        """_summary_

        Args:
            mode (str, optional): _description_. Defaults to 'most probably'.

        Raises:
            AttributeError: _description_

        Returns:
            _type_: _description_
        """
        terms = [self.__deltar2/self.r2, self.__deltar1/(self.r1+self.r2), self.__deltar2/(self.r1+self.r2)] 

        if mode == 'pesimist':
            solution = self.__r_value*sum(terms)
        elif mode == 'most probably':
            solution = self.__r_value*sum([v**2 for v in terms])**(1/2)
        else:
            raise AttributeError("The mode to calculate the uncertainty in not correctly defined")
        return solution
        


if __name__ == "__main__":

    from statistics import mean, stdev

    # Single Value
    vin = 12.0
    sens1 = Voltage_divider(200, 100)
    
    sens1.mean_value(vin)
    sens1.noise(vin, 'most probably')
    print(sens1.meanV, sens1.uncertaintyV)

    measures = list()
    for i in range(10000):
        sens1.noise(vin, 'pesimist')
        measures.append(sens1.uncertaintyV)

    print("Mu, std : ", mean(measures), stdev(measures))

    # Array of measures
    Vin = np.array([12, 11.3, 10.5])
    sens2 = Voltage_divider(200, 100)
    
    sens2.mean_value(Vin)
    sens2.noise(Vin, 'most probably')
    print(sens2.meanV, sens2.uncertaintyV)

