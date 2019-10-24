import numpy as np
class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]
    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a

class LinearActionLinearPolicy(object):
    def __init__(self, theta):
        # theta: [(n * 3) * (n + 1) + (n+1)]
        assert len(theta) == 21 # only supports two products classes
        self.w = theta[:-3].reshape(6,3)
        self.b = theta[-3:]
    def act(self,ob):
        action = np.zeros(2, dtype=np.int64)
        y = ob.dot(self.w) + self.b
        action[0] = np.argmax(y[:2]) # classification: which product to produce
        action[1] = min(max(0, int(y[2])), 30) # amount of product to produce
        return action