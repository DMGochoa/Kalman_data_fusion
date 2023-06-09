import numpy as np

class CircuitoPasaBajos:
    
    def __init__(self, R=100, L=375e-3, C=75e-6) -> None:
        self.R = R
        self.L = L
        self.C = C
        self.num = [0, 1/(L*C)]
        self.den = [R/L, 1/(L*C)]

    def A(self):
        A = np.array([[1 / self.C, 0],
                      [-(self.R / self.L), -(1 / self.L)]])
        return A

    def B(self):
        return np.array([[0],[1 / self.L]])

    def model(self, x, Vin):
        return np.array([x[0] / self.C,
                         -(self.R / self.L) * x[0] - (1 / (self.L)) * x[1] + Vin / self.L])