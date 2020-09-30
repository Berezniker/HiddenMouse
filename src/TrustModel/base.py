import numpy as np
from math import exp


class TrustModel:
    def __init__(self, clf,
                 A: float = 0.00,
                 B: float = 0.25,
                 C: float = 1.00,
                 D: float = 1.00,
                 lockout: float = 90.0):
        """
        Dynamic Trust Model (DTM)

        :param clf: Classifier
        :param A: Threshold for penalty or reward
        :param B: Width of the sigmoid, B > 0
        :param C: Maximum reward, C > 0
        :param D: Maximum penalty, D > 0
        :param lockout: Minimum T-value after which blocking occurs
        """
        self.A, self.B, self.C, self.D = A, B, C, D
        self.T_lockout = lockout
        self.T_value = 100
        self.clf = clf

    def predict(self, X: np.ndarray) -> float:
        """
        :param X: Feature vector
        :return: classifier trust value
        """
        return self.clf.decision_function(X.reshape(1, -1))

    def decision(self, X) -> bool:
        """
        Recalculates the T-value and decides to block the user

        :param X: Feature vector
        :return: True, if the user's trust falls below the threshold
        """
        up = self.D * (1.0 + (1.0 / self.C))
        down = (1.0 / self.C) + exp(-(self.predict(X) - self.A) / self.B)
        delta_T = min(-self.D + up / down, self.C)
        self.T_value = min(max(self.T_value + delta_T, 0), 100)
        return self.T_value < self.T_lockout

    def restart(self) -> None:
        """
        Sets the T-value to the maximum after re-authorization
        """
        self.T_value = 100
