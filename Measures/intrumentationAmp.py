"""_summary_

Raises:
    TypeError: _description_

Returns:
    _type_: _description_
"""

import numpy as np


class AD620():
    """_summary_
    """    

    def __init__(self, rg=49400):
        """_summary_

        Args:
            rg (int, optional): _description_. Defaults to 49400.
        """        
        self.rg = rg
        self.G = 49400/self.rg + 1

    def output(self, sample):
        """_summary_

        Args:
            sample (_type_): _description_

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """        
        if type(sample) == int or type(sample) == float or type(sample) == np.ndarray:
            return self.G*sample
        else:
            raise TypeError("The type of argument accepted are int or numpy array")


if __name__ == "__main__":
    # Single value
    amp = AD620()
    print(amp.output(2.0))
    # Sample array
    x = np.array([1, 0.5, 0.2, 2.0])
    print(type(x))
    print(amp.output(x))
