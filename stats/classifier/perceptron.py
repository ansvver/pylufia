# -*- coding: utf-8 -*-

"""
perceptron.py
"""

import scipy as sp

class LinearClassifier():
    def __init__(self, alpha, A, E):
        self.w = None
        self.alpha = alpha
        self.A = A
        self.E = E

    def train(self, X, y):
        self.w = sp.zeros(X.shape[1])
        self.w[:] = 1.0/len(self.w)

        for x,_y in zip(X,y):
            f = _y*sp.dot(self.W.T,x)
            if f > E:
                pass
            else:
                self.w = self.w + _y*self.alpha*self.A*x

    def classify(self, x):
        f = self._sign(sp.dot(self.w.T,x))
        return f

    def _sign(self, x):
        y = 0
        if x >= 0:
            y = 1
        else:
            y = -1
        return y


class Perceptron():
    def __init__(self):
        self.alpha = 1.0
        self.A = None
        self.E = 0.0

    def train(self, X, y):
        self.w = sp.zeros(X.shape[1])
        self.w[:] = 1.0/len(self.w)

        for _x,_y in zip(X,y):
            f = self._sign(sp.dot(self.w.T,_x))
            if f == _y:
                pass
            else:
                self.w = self.w + y*_x

    def classify(self, x):
        f = self._sign(sp.dot(self.w.T,x))
        return f

    def _sign(self, x):
        y = 0
        if x >= 0:
            y = 1
        else:
            y = -1
        return y


def main():
    X = sp.array([[1,2,3,4,5],
                    [-1,-2,-3,-4,-5]])
    y = sp.array([1,-1])

    c_model = Perceptron()
    c_model.train(X,y)
    input_01 = sp.array([6,7,8,9,10])
    input_02 = sp.array([-4,-5,-6,-7,-8])
    f = c_model.classify(input_01)
    print(f)

if __name__ == '__main__':
    main()