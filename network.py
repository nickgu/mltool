#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 


import sys
from abc import ABCMeta, abstractmethod

import numpy
import theano
from sklearn import preprocessing
from theano import tensor as T

import ConfigParser


class LearningConfig:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.symbol_learning_rate = T.scalar()

class ILayer:
    '''
    ILayer: layers' interface.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, inputs, config_reader):
        pass

    @abstractmethod
    def make_updates(self, updates, cost, learning_config):
        ''' Making updates to parameters. '''
        pass

class Layer_FullConnect(ILayer):
    def __init__(self, inputs, config_reader=None):
        n_in = int( config_reader('n_in') )
        n_out = int( config_reader('n_out') )

        self.x = inputs[0]

        self.w = theano.shared(value=(numpy.random.rand(n_in, n_out)-0.5), borrow=True)
        self.b = theano.shared(value=numpy.random.rand(n_out), borrow=True)

        self.y = self.x.dot(self.w) + self.b
        self.active = theano.function([self.x], self.y)

    def make_updates(self, updates, cost, learning_config):
        symbol_learning_rate = learning_config.symbol_learning_rate

        gy_w = T.grad(cost=cost, wrt=self.w)
        gy_b = T.grad(cost=cost, wrt=self.b)

        # SGD
        updates.append( (self.w, self.w - gy_w * symbol_learning_rate) )
        updates.append( (self.b, self.b - gy_b * symbol_learning_rate) )

class Layer_Dot(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x1 = inputs[0]
        self.x2 = inputs[1]
        self.y = T.batched_dot(self.x1, self.x2)
        self.active = theano.function([self.x1, self.x2], self.y)

    def make_updates(self, updates, cost, learning_config):
        ''' no updates '''
        pass

class Layer_Sigmoid(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # calc sigmoid for each value in matrix.
        self.y = T.nnet.sigmoid(x)
        self.active = theano.function([self.x], self.y)

    def make_updates(self, updates, cost, learning_config):
        ''' no updates '''
        pass

class Layer_Tanh(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # calc tanh for each value in matrix.
        self.y = T.tanh(x)
        self.active = theano.function([self.x], self.y)

    def make_updates(self, updates, cost, learning_config):
        ''' no updates '''
        pass

class Layer_Norm2Cost(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        self.label = inputs[1]
        self.y = T.mean( (self.x - self.label) ** 2 )
        self.active = theano.function([self.x, self.label], self.y)

    def make_updates(self, updates, cost, learning_config):
        ''' no updates '''
        pass

class ConfigNetwork:
    def __init__(self, config_file, network_name):
        self.learning_config = LearningConfig(0.1)
        self.learning_config.symbol_learning_rate = T.scalar()

        self.__inputs = []
        self.__label = T.fmatrix()
        self.__layers = []
        self.__layers_info = {}

        cp = ConfigParser.ConfigParser()
        self.__config_parser = cp
        self.__network_name = network_name
        cp.read(config_file)
        
        input_count = int(cp.get(network_name, 'input_count'))
        print >> sys.stderr, 'input_count = %d' % input_count
        for i in range(input_count):
            self.__inputs.append( T.fmatrix() )

        layer_names = cp.get(network_name, 'layers').split(',')
        active_name = cp.get(network_name, 'active').strip()
        cost_name = cp.get(network_name, 'cost').strip()
   
        for layer_name in layer_names:
            layer_type = cp.get(network_name, '%s.type' % layer_name)
            input_names = cp.get(network_name, '%s.input' % layer_name).split(',')

            # type, inputs, layer_refer.
            self.__layers_info[layer_name] = [layer_type, input_names, None]
        
        for name in layer_names:
            self.__init_layer(name)

        # make active function.
        self.active = theano.function(self.__inputs, self.__get_layer(active_name).y)
        
        # make training function.
        updates = []
        print >> sys.stderr, 'Get cost = %s' % cost_name
        cost_y = self.__get_layer(cost_name).y
        for layer in self.__layers:
            layer.make_updates(updates, cost_y, self.learning_config)
        print >> sys.stderr, 'updates=%d' % len(updates)

        inputs = []
        for item in self.__inputs:
            inputs.append(item)
        inputs.append(self.__label)
        inputs.append(self.learning_config.symbol_learning_rate)

        self.train = theano.function(
                    inputs,
                    cost_y, 
                    updates=updates)

    def predict(self, *args):
        return self.active(*args)

    def fit(self, *args):
        # simple N epoch train.
        param = list(args)
        param.append( self.learning_config.learning_rate )
        for i in range(1000):
            final_cost = self.train(*param)
        return final_cost

    def __init_layer(self, name):
        print >> sys.stderr, 'Try to init layer [%s]' % name
        ltype, inames, layer = self.__layers_info.get(name, ['', [], None])

        if layer is not None:
            # layer has already inited.
            print >> sys.stderr, '[%s] is inited.' % name
            return layer

        inputs = []
        for sub_name in inames:
            if sub_name.startswith('input[') and sub_name[-1] == ']':
                # is from input
                iid = int(sub_name.replace('input[','').replace(']', ''))
                inputs.append(self.__inputs[iid])

            elif sub_name == '__label__':
                inputs.append(self.__label)

            else:
                l = self.__init_layer(sub_name)
                inputs.append(l.y)

        config_reader = self.__layer_config_reader(name)
        if ltype == 'full_connect':
            layer = Layer_FullConnect(inputs, config_reader)
        elif ltype == 'dot':
            layer = Layer_Dot(inputs, config_reader)
        elif ltype == 'norm2':
            layer = Layer_Norm2Cost(inputs, config_reader)
        elif ltype == 'sigmoid':
            layer = Layer_Sigmoid(inputs, config_reader)

        # for random access.
        self.__layers_info[name][2] = layer
        # for iteration.
        self.__layers.append(layer)
        return layer

    def __get_layer(self, name):
        ltype, inames, layer = self.__layers_info.get(name, ['', [], None])
        return layer

    def __layer_config_reader(self, layer_name):
        return lambda opt: self.__config_parser.get(self.__network_name, ('%s.'%layer_name) + opt)

if __name__=='__main__':
    pass







