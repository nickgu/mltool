#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import sys
import numpy
import theano
import random

from sklearn import preprocessing
from theano import tensor as T

class LogisticLayer:
    def __init__(self, dim, learning_rate=0.1, output_01=False, batch_size=512):
        self.__dim = dim
        self.__learning_rate = learning_rate
        self.__output_01 = output_01
        self.__batch_size = batch_size

        X = T.fmatrix('X')

        ones = numpy.ndarray(shape=512)
        ones.fill(1)
        ones = ones.astype(numpy.float32)

        w = theano.shared(
            value=numpy.random.rand(dim).astype(numpy.float32) - [numpy.float32(0.5)]*dim,
            name='w',
            borrow=True
        )

        b = theano.shared(
            value=random.random() - 0.5,
            name='b',
            borrow=True
        )

        self.__e_gw = theano.shared(
            value=numpy.zeros(dim).astype(numpy.float32),
            name='e_gw',
            borrow=True
        )

        self.__e_gb = theano.shared(
            value=0.,
            name='e_gb',
            borrow=True
        )

        self.__w = w
        self.__b = b

        y = T.fvector('y')

        # logit function.
        out = T.nnet.sigmoid( T.dot(X, w) + ones * b )
        # cost function
        cost = -T.mean(y * T.log(out) + (1-y) * T.log( (1-out) ))
        #cost = T.mean( (y-out) ** 2 )

        
        self.__gy_w = T.grad(cost=cost, wrt=w)
        self.__gy_b = T.grad(cost=cost, wrt=b)

        updates = [
                (w, w - 0.1 / T.sqrt(self.__e_gw + 0.1) * self.__gy_w),
                (b, b - 0.1 / T.sqrt(self.__e_gb + 0.1) * self.__gy_b),
                (self.__e_gw, 0.7 * self.__e_gw + 0.3 * self.__gy_w),
                (self.__e_gb, 0.7 * self.__e_gb + 0.3 * self.__gy_b)
                #(w, w - 0.1 / T.sqrt(self.__e_gw + 0.1) * self.__learning_rate * self.__gy_w),
                #(b, b - 0.1 / T.sqrt(self.__e_gb + 0.1) * self.__learning_rate * self.__gy_b),
                ]

        self.__logit = theano.function( [X], out)

        self.__training = theano.function( 
                [X, y], 
                cost,
                updates = updates
                )

    def predict(self, X):
        X = preprocessing.maxabs_scale(X)
        pred = self.__logit(X) 
        if self.__output_01:
            return numpy.array(map(lambda x:1. if x>=.5 else 0., pred))
        else:
            return pred

    def train(self, X, y):
        return self.__training(X, y)
        #print 'W: ' + str(self.__w.get_value())
        #print 'b: ' + str(self.__b.get_value())

    # same interface as sklearn-model.
    def fit(self, X, y):
        X = preprocessing.maxabs_scale(X)
        batch_num = (len(X) + self.__batch_size - 1) / self.__batch_size

        last_loss = None
        epoch = 0
        while 1:
            epoch += 1

            epoch_cost = 0
            for batch in range(batch_num):
                beg = batch * self.__batch_size
                end = beg + self.__batch_size
                epoch_cost += self.train(X[beg:end], y[beg:end])

            loss = epoch_cost / batch_num
            print >> sys.stderr, 'Epoch[%d] loss : %f' % (epoch, loss)
            if last_loss is not None:
                if last_loss - loss < 1e-5:
                    print >> sys.stderr, 'Early stop'
                    break
                if self.__learning_rate>=1e-3 and last_loss - loss < 1e-3:
                    self.__learning_rate = self.__learning_rate * 0.5
                    print >> sys.stderr, 'Change learning rate : %f (%f)' % (self.__learning_rate,
                            last_loss - loss)
            last_loss = loss

def Test_SingleTrain():
    # test code.
    TestDim = 5
    EpochTimes = 200
    LearningRate = 0.2
    layer = LogisticLayer(dim = TestDim, learning_rate=LearningRate)
    single_x = numpy.random.rand(TestDim).astype(numpy.float32) - [numpy.float32(0.5)] * TestDim
    single_y = 0.
    print layer.predict(single_x)
    for i in range(EpochTimes):
        layer.train(single_x, single_y)
    print layer.predict(single_x)

if __name__=='__main__':
    Test_SingleTrain()

