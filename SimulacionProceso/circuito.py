class CircuitoPasaBajos:
    
    def __init__(self, R=100, L=375e-3, C=75e-6) -> None:
        self.num = [0, 1/(L*C)]
        self.den = [R/L, 1/(L*C)]