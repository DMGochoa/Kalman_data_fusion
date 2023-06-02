import numpy as np

class RegressionMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.diff = y_true - y_pred
        self.mse = self.mean_squared_error()
        self.mae = self.mean_absolute_error()
        self.rmse = self.root_mean_squared_error()
        self.r2 = self.r2_score()
        self.mape = self.mean_absolute_percentage_error()
        self.mase = self.mean_absolute_scaled_error()
        self.smape = self.symmetric_mean_absolute_percentage_error()

    def mean_squared_error(self):
        return np.mean((self.diff) ** 2)

    def mean_absolute_error(self):
        return np.mean(np.abs(self.diff))

    def root_mean_squared_error(self):
        return np.sqrt(self.mse)

    def r2_score(self):
        return 1 - (np.sum((self.diff) ** 2) / np.sum((self.y_true - np.mean(self.y_true)) ** 2))

    def mean_absolute_percentage_error(self):
        return np.mean(np.abs((self.diff) / self.y_true)) * 100

    def mean_absolute_scaled_error(self):
        return np.mean(np.abs((self.diff) / (np.mean(np.abs(self.diff)))))

    def symmetric_mean_absolute_percentage_error(self):
        return 100 * np.mean(np.abs(-self.diff) / ((np.abs(self.y_true) + np.abs(self.y_pred)) / 2))