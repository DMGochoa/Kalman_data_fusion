"""_summary_

Raises:
    TypeError: _description_
"""

import numpy as np


class diff_amp():
    """_summary_
    """
    def __init__(self, max, min, pv=10, mv=-10):
        """_summary_

        Args:
            max (_type_): _description_
            min (_type_): _description_
            pv (int, optional): _description_. Defaults to 10.
            mv (int, optional): _description_. Defaults to -10.
        """
        self.max = max
        self.min = min
        self.pv = pv
        self.mv = mv
        self.m = (pv - mv)/(max - min)
        self.y = 0

    def output(self, input):
        """_summary_

        Args:
            input (_type_): _description_

        Raises:
            TypeError: _description_
        """
        self.y_daq = input
        if type(input) == int or type(input) == float or type(input) == np.ndarray:
            self.y = self.m*(input-self.min)+(self.mv)
        else:
            raise TypeError(
                "The type of argument accepted are int or numpy array")
        return self.y

    def daq(self, input):
        if type(input) == int or type(input) == float or type(input) == np.ndarray:
            return self.y_daq#self.m*(input-self.min)+(self.mv)
        else:
            raise TypeError(
                "The type of argument accepted are int or numpy array")


if __name__ == "__main__":
    # Single value
    acond = diff_amp(5, 0)
    acond.output(2.5)
    print(acond.y)
    # Array
    acond2 = diff_amp(-1,1)
    x = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    acond2.output(x)
    print(acond2.y)
    # Array
    acond3 = diff_amp(5, 0)
    x = np.array([0, 1, 2, 3, 4, 5])
    acond3.output(x)
    print(acond3.y)
    # Invalid value
    acond3 = diff_amp(5, 0)
    x = (1, 2, 3)
    acond3.output(x)
    print(acond3.y)
