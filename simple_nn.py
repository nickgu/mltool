#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import sys
import random
import time
from abc import ABCMeta, abstractmethod

import numpy
import theano
from sklearn import preprocessing
from theano import tensor as T


class ILayer:
    '''
    ILayer: layers' interface.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def make_updates(self, updates, cost, learning_rate, use_sgd = True):
        ''' Making updates to parameters. '''
        pass


class SimpleLayer(ILayer):
    '''
    SimpleLayer
    members:
        y: get output variable.
        W: param matrix
        b: bias vector

        active: active function.
    '''

    def __init__(self, input, n_in, n_out, tanh=False):
        self.n_in = n_in
        self.n_out = n_out

        X = input
        W = theano.shared(value=(numpy.random.rand(n_in, n_out)-0.5), borrow=True)
        b = theano.shared(value=numpy.random.rand(n_out), borrow=True)

        self.W = W
        self.b = b

        #### Active function ####
        # logit function.
        self.Y = T.nnet.sigmoid( T.dot(X, W) + b )
        # tanh function.
        if tanh:
            self.Y = T.tanh( T.dot(X, W) + b )

        # Function Definition.
        self.active = theano.function([X], self.Y)

        # Regularization.
        self.reg = T.sum(W ** 2) + T.sum(b ** 2)

    def make_updates(self, updates, cost, learning_rate, use_sgd=True):
        gy_w = T.grad(cost=cost, wrt=self.W)
        gy_b = T.grad(cost=cost, wrt=self.b)

        # SGD
        if use_sgd:
            updates.append( (self.W, self.W - learning_rate * gy_w) )
            updates.append( (self.b, self.b - learning_rate * gy_b) )
        else:
            # Ada-delta
            # Learning method: ada-delta 
            E_gw = theano.shared(value=numpy.zeros(shape=(self.n_in, self.n_out)), borrow=True)
            E_gb = theano.shared(value=numpy.zeros(self.n_out), borrow=True)

            updates.append( (self.W, self.W - 0.1 / T.sqrt(E_gw + 0.1) * gy_w) )
            updates.append( (self.b, self.b - 0.1 / T.sqrt(E_gb + 0.1) * gy_b) )
            updates.append( (E_gw, 0.7 * E_gw + 0.3 * gy_w) )
            updates.append( (E_gb, 0.7 * E_gb + 0.3 * gy_b) )

class EmbeddingLayer(ILayer):
    '''
    EmbeddingLayer:
        dot(id_a * embeddings_a, id_b * embeddings_b)
    '''
    def __init__(self, input_group, n_in_list, emb_dim):
        print input_group
        self.n_in_list = n_in_list
        self.n_out = emb_dim

        Xs = []
        Ws = []
        bs = []
        outs = []

        self.Ws = Ws
        self.bs = bs
        self.Xs = Xs
        self.outs = outs

        for i, input in enumerate(input_group):
            x = input
            w = theano.shared(value=(numpy.random.rand(n_in_list[i], emb_dim)-0.5), borrow=True) 
            b = theano.shared(value=numpy.random.rand(emb_dim), borrow=True)
            Xs.append( x )
            Ws.append( w )
            bs.append( b )
            outs.append( T.dot(x, w) + b )

        #### Active function ####
        # TODO: just support dot(Xs[0], Xs[1]) now.
        if len(Xs)!=2:
            raise Exception('Just support 2 input group now.')
        self.Y = T.batched_dot( outs[0], outs[1] )

        # Function Definition.
        self.active = theano.function(Xs, self.Y)

        # Regularization.
        #self.reg = T.sum(W ** 2) + T.sum(b ** 2)

    def make_updates(self, updates, cost, learning_rate, use_sgd=True):
        for i in range(len(self.Ws)):
            w = self.Ws[i]
            b = self.bs[i]
            gy_w = T.grad(cost=cost, wrt=w)
            gy_b = T.grad(cost=cost, wrt=b)

            # SGD
            if use_sgd:
                updates.append( (self.Ws[i], self.Ws[i] - learning_rate * gy_w) )
                updates.append( (self.bs[i], self.bs[i] - learning_rate * gy_b) )
            else:
                # Ada-delta
                # Learning method: ada-delta 
                E_gw = theano.shared(value=numpy.zeros(shape=(self.n_in_list[i], self.n_out)), borrow=True)
                E_gb = theano.shared(value=numpy.zeros(self.n_out), borrow=True)

                updates.append( (self.Ws[i], self.Ws[i] - 0.1 / T.sqrt(E_gw + 0.1) * gy_w) )
                updates.append( (self.bs[i], self.bs[i] - 0.1 / T.sqrt(E_gb + 0.1) * gy_b) )
                updates.append( (E_gw, 0.7 * E_gw + 0.3 * gy_w) )
                updates.append( (E_gb, 0.7 * E_gb + 0.3 * gy_b) )


class SimpleNetwork:
    def __init__(self, n_in, n_out, hidden_layers_width, batch_size=512, learning_rate=0.1, output_01=False):
        self.__learning_rate = learning_rate
        self.__output_01 = output_01

        self.layers = []
        self.X = T.fmatrix()
        
        # build network.
        last_layer_output = self.X
        width = [n_in] + hidden_layers_width + [n_out]
        for i in range(len(width)-1):
            use_tanh=True
            if i == len(width)-2:
                use_tanh=False

            print >> sys.stderr, 'Build network: %dx%d %s' % (width[i], width[i+1], 'tanh' if use_tanh else 'sigmoid')
            l = SimpleLayer(
                    input=last_layer_output,
                    n_in=width[i], 
                    n_out=width[i+1],
                    tanh=use_tanh)
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
        #cost = -T.mean(label * T.log(self.Y) + (1-label) * T.log( (1-self.Y) ))
        # RMSE
        cost = T.mean( (label - self.Y) ** 2 )
        
        # regulazation.
        reg = 0
        '''
        for layer in self.layers:
            reg = reg + layer.reg
        '''

        # Train function.
        # NOTICE: train input:
        #   X : matrix with rows: batch_size, columns: features num.
        #   Y : matrix with rows: batch_size, columns: 0/1 (one-hot not supported).
        updates = []
        for l in self.layers:
            l.make_updates(updates, cost + reg, self.__learning_rate, use_sgd=True)
        self.train = theano.function( 
                [self.X, label], 
                cost,
                updates = updates
                )


    def predict(self, X, do_scale=False):
        if do_scale:
            X = preprocessing.maxabs_scale(X)
        pred = self.active(X) 

        if self.__output_01:
            for x in numpy.nditer(pred, op_flags=['readwrite']):
                x[...] = 1. if x[...]>=.5 else 0.
        return pred

    def fit(self, X, y, do_scale=False, batch_size=512):
        if do_scale:
            X = preprocessing.maxabs_scale(X)
        batch_num = (len(X) + batch_size - 1) / batch_size

        last_loss = None
        epoch = 0
        while 1:
            st = time.time()

            epoch += 1
            epoch_cost = 0
            for batch in range(batch_num):
                beg = batch * batch_size
                end = beg + batch_size
                # try to cast label.
                label = numpy.array(y[beg:end])#.astype(numpy.float32)
                if len(label.shape) == 1:
                    label.shape = (label.shape[0], 1)
                epoch_cost += self.train(X[beg:end], label)

            dt = time.time() - st

            loss = epoch_cost / batch_num
            diff_loss = 0
            if last_loss is not None:
                diff_loss = last_loss - loss
            print >> sys.stderr, 'Epoch[%d] loss : %f (diff_loss:%f, diff_time:%.3f(s))' % (
                    epoch, loss, diff_loss, dt)
            if last_loss is not None:
                if last_loss - loss < 1e-5:
                    print >> sys.stderr, 'Early stop'
                    break

                '''
                if self.__learning_rate>=1e-3 and last_loss - loss < 1e-3:
                    self.__learning_rate = self.__learning_rate * 0.5
                    print >> sys.stderr, 'Change learning rate : %f (%f)' % (self.__learning_rate,
                            last_loss - loss)
                '''
            last_loss = loss

    '''
    '''
    #def interactive_fit(self, X, y):


class EmbeddingNetwork:
    '''
    Trainging Embedding network.
    Input type:
        Xs = [X[0], X[1], ...]
    '''
    def __init__(self, n_in_list, embedding_size, batch_size=512, learning_rate=0.1, output_01=False):
        self.__learning_rate = learning_rate
        self.__output_01 = output_01

        self.layers = []

        self.Xs = []
        for x in n_in_list:
            self.Xs.append(T.fmatrix())

        # first layer: 
        self.layers.append(EmbeddingLayer(self.Xs, n_in_list, embedding_size))
       
        # network output.
        self.Y = self.layers[0].Y
        self.active = theano.function(self.Xs, self.Y)

        #### cost function ####
        # label.
        label = T.fmatrix() 
        # LogLikelihood
        #cost = -T.mean(label * T.log(self.Y) + (1-label) * T.log( (1-self.Y) ))
        # RMSE
        cost = T.mean( (label.T - self.Y) ** 2 )
        self.cost = cost
       
        # Train function.
        # NOTICE: train input:
        #   X : matrix with rows: batch_size, columns: features num.
        #   Y : matrix with rows: batch_size, columns: 0/1 (one-hot not supported).
        updates = []
        for l in self.layers:
            l.make_updates(updates, cost, self.__learning_rate, use_sgd=True)
        training_input = self.Xs + [label]
        self.train = theano.function( 
                training_input, 
                cost,
                updates = updates
                )

    def predict(self, X1, X2):
        pred = self.active(X1, X2) 
        return pred

    def fit(self, X1, X2, y, batch_size=512):
        batch_num = (len(X1) + batch_size - 1) / batch_size

        last_loss = None
        epoch = 0
        while 1:
            st = time.time()

            epoch += 1
            epoch_cost = 0
            for batch in range(batch_num):
                beg = batch * batch_size
                end = beg + batch_size
                # try to cast label.
                label = numpy.array(y[beg:end])#.astype(numpy.float32)
                if len(label.shape) == 1:
                    label.shape = (label.shape[0], 1)
                epoch_cost += self.train(X1[beg:end], X2[beg:end], label)

            dt = time.time() - st

            loss = epoch_cost / batch_num
            diff_loss = 0
            if last_loss is not None:
                diff_loss = last_loss - loss
            print >> sys.stderr, 'Epoch[%d] loss : %f (diff_loss:%f, diff_time:%.3f(s))' % (
                    epoch, loss, diff_loss, dt)
            if last_loss is not None:
                if last_loss - loss < 1e-5:
                    print >> sys.stderr, 'Early stop'
                    break
            last_loss = loss

if __name__=='__main__':
    pass

