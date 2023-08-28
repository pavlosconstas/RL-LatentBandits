import numpy as np

class LinUCB:
    def __init__(self, alpha, d):
        # alpha: exploration parameter
        # d: dimension of the feature vector
        self.alpha = alpha
        self.d = d
        self.A = np.identity(d)  # A matrix
        self.b = np.zeros(d)     # b vector

    def predict(self, x):
        # x: feature vector
        theta = np.linalg.inv(self.A).dot(self.b)
        p = theta.T.dot(x) + self.alpha * np.sqrt(x.T.dot(np.linalg.inv(self.A)).dot(x))
        return p

    def update(self, x, reward):
        # x: feature vector
        # self.A += np.outer(x, x)
        # self.b += reward * x
        for i in range(len(x)):
            self.A += np.outer(x[i], x[i])
            self.b += reward[i] * x[i]