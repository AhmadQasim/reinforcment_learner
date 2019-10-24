import numpy as np

"""
1.binary model
2.scalar model
model for y=f(t, eps)

eps ~ N(eps| mu(t), sigma(t))
eps ~ Uniform(eps | mu(t)- Bound, mu(t)+Bound)

mu = Regression from data
mu = sinus function

3.supply dependent model
model y=f(t, eps, Supply)
"""


class DemandDecision(object):
    def __init__(self):
        self.base_function = lambda x:x%24/24+0.4

    def binary_buy(self, t):
        if isinstance(t, int):
            eps = np.random.rand()
        elif isinstance(t, (np.generic, np.ndarray)):
            try:
                eps = np.random.rand(t.shape[0])
            except IndexError:
                eps = np.random.rand()
        else:
            raise RuntimeError('Input type is int or numpy array')
        # print(self.base_function(x))
        return eps > self.base_function(t)
        # return 1 > self.base_function(t)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    demand = DemandDecision()
    t = np.arange(24*10)
    out = demand.binary_buy(t)
    print(out)
    plt.plot(out)
    plt.show()
