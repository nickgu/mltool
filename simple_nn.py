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

'''
Layer
members:
    y: get output variable.
    W: param matrix
    b: bias vector

    active: active function.
'''
class SimpleLayer:
    def __init__(self, input, n_in, n_out, batch_size=512):
        #X = T.fmatrix('X')
        X = input
        W = theano.shared(value=numpy.random.rand(n_in, n_out) - 0.5, borrow=True)
        b = theano.shared(value=numpy.random.rand(n_out), borrow=True)

        self.W = W
        self.b = b

        #### Active function ####
        # logit function.
        self.Y = T.nnet.sigmoid( T.dot(X, W) + numpy.ones( (batch_size, n_out) ) * b )

        # Function Definition.
        self.active = theano.function([X], self.Y)

    def make_updates(self, updates, cost, learning_rate):
        gy_w = T.grad(cost=cost, wrt=self.W)
        gy_b = T.grad(cost=cost, wrt=self.b)
        updates.append( (self.W, self.W - learning_rate * gy_w) )
        updates.append( (self.b, self.b - learning_rate * gy_b) )

class SimpleNetwork:
    def __init__(self, n_in, hidden_layers_width, batch_size=512, learning_rate=0.1, output_01=False):
        self.__learning_rate = learning_rate

        self.layers = []
        self.X = T.fmatrix()
        
        # build network.
        last_layer_output = self.X
        width = [n_in] + hidden_layers_width + [1]
        for i in range(len(width)-1):
            print >> sys.stderr, 'Build network: %dx%d' % (width[i], width[i+1])
            l = SimpleLayer(
                    input=last_layer_output,
                    n_in=width[i], 
                    n_out=width[i+1], 
                    batch_size=batch_size)
            last_layer = l
            last_layer_output = l.Y
            self.layers.append(l)
        
        # network output.
        self.Y = last_layer.Y
        self.active = theano.function([self.X], self.Y)

        #### cost function ####
        # label.
        label = T.fmatrix() 
        # LogLikelihood
        cost = -T.mean(label * T.log(self.Y) + (1-label) * T.log( (1-self.Y) ))
        # RMSE
        #cost = T.mean( (label - self.Y) ** 2 )

        updates = []
        for l in self.layers:
            l.make_updates(updates, cost, self.__learning_rate)
        self.train = theano.function( 
                [self.X, label], 
                cost,
                updates = updates
                )


    def predict(self, X, do_scale=True):
        if do_scale:
            X = preprocessing.maxabs_scale(X)
        pred = self.active(X) 
        if self.__output_01:
            return numpy.array(map(lambda x:1. if x>=.5 else 0., pred))
        else:
            return pred

    def fit(self, X, y, do_scale=True):
        if do_scale:
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

class LogisticClassifier:
    def __init__(self, dim, learning_rate=0.1, output_01=False, batch_size=512):
        self.__dim = dim
        self.__learning_rate = learning_rate
        self.__output_01 = output_01
        self.__batch_size = batch_size

        X = T.fmatrix('X')
        w = theano.shared(value=numpy.random.rand(dim).astype(numpy.float32) - 0.5, borrow=True)
        b = theano.shared(value=random.random() - 0.5, borrow=True)
        y = T.fvector('y')

        self.__w = w
        self.__b = b

        #### active function ####
        # logit function.
        out = T.nnet.sigmoid( T.dot(X, w) + numpy.ones(shape=batch_size) * b )

        #### cost function ####
        # LogLikelihood
        cost = -T.mean(y * T.log(out) + (1-y) * T.log( (1-out) ))
        # RMSE
        #cost = T.mean( (y-out) ** 2 )
        
        self.__gy_w = T.grad(cost=cost, wrt=w)
        self.__gy_b = T.grad(cost=cost, wrt=b)

        # Ada-delta
        # Learning method: ada-delta 
        E_gw = theano.shared(value=numpy.zeros(dim).astype(numpy.float32), borrow=True)
        E_gb = theano.shared(value=0., borrow=True)

        updates = [
                (w, w - 0.1 / T.sqrt(E_gw + 0.1) * self.__gy_w),
                (b, b - 0.1 / T.sqrt(E_gb + 0.1) * self.__gy_b),
                (E_gw, 0.7 * E_gw + 0.3 * self.__gy_w),
                (E_gb, 0.7 * E_gb + 0.3 * self.__gy_b)
                ]

        # SGD.
        '''
        updates = [
                (w, w - self.__learning_rate * self.__gy_w),
                (b, b - self.__learning_rate * self.__gy_b),
                ]
        '''

        # Function Definition.
        self.__logit = theano.function( [X], out)

        self.train = theano.function( 
                [X, y], 
                cost,
                updates = updates
                )

    def predict(self, X, do_scale=True):
        if do_scale:
            X = preprocessing.maxabs_scale(X)
        pred = self.__logit(X) 
        if self.__output_01:
            return numpy.array(map(lambda x:1. if x>=.5 else 0., pred))
        else:
            return pred

    # same interface as sklearn-model.
    def fit(self, X, y, do_scale=True):
        if do_scale:
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
    layer = LogisticClassifier(dim = TestDim, learning_rate=LearningRate)
    single_x = numpy.random.rand(TestDim).astype(numpy.float32) - [numpy.float32(0.5)] * TestDim
    single_y = 0.
    print layer.predict(single_x)
    for i in range(EpochTimes):
        layer.train(single_x, single_y)
    print layer.predict(single_x)

if __name__=='__main__':
    Test_SingleTrain()

