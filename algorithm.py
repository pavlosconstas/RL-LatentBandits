import numpy as np

class LinUCB:
    def __init__(self, alpha, num_arms, d):
        self.alpha = alpha
        self.num_arms = num_arms
        self.d = d
        self.A = [np.eye(d) for _ in range(num_arms)]
        self.b = [np.zeros((d, 1)) for _ in range(num_arms)]
        self.theta = [np.zeros((d, 1)) for _ in range(num_arms)]

    def choose_action(self, context):
        x = context.reshape(-1, 1)
        p = np.zeros(self.num_arms)
        
        for action in range(self.num_arms):
            A_inv = np.linalg.inv(self.A[action])
            self.theta[action] = A_inv.dot(self.b[action])
            p[action] = self.theta[action].T.dot(x) + self.alpha * np.sqrt(x.T.dot(A_inv).dot(x))
            
        return np.argmax(p)

    def update(self, chosen_action, reward, context):
        x = context.reshape(-1, 1)
        self.A[chosen_action] += x.dot(x.T)
        self.b[chosen_action] += reward * x