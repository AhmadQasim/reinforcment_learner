import numpy as np
class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]
        # print(theta)
    def act(self, ob):
        try:
            y = ob.dot(self.w) + self.b
        except:
            ob = ob[1:]
            y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a

class BinaryActionNonLinearPolicy(object):
    def __init__(self, theta):
        assert len(theta)==13 , 'hard coded 2 layer NN'
        # self.w1 = np.zeros((3,2))
        # self.b1 = np.zeros((3,))
        # self.w2 = np.zeros(3)
        # self.b2= np.zeros(1)
        self.w1 = theta[:6].reshape(3,2)
        self.b1 = theta[6:9]
        self.w2=theta[9:-1]
        self.b2=theta[-1]
    def act(self, ob):
        x = np.dot(self.w1, ob) + self.b1
        # x = np.maximum(0, x)
        x = 1/(1+np.exp(-x))
        x = x.dot(self.w2) + self.b2
        a = int(x<0)
        return a
