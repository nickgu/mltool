#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 


import sys
from abc import ABCMeta, abstractmethod


import tensorflow as tf

'''
import numpy
import theano
from sklearn import preprocessing
from theano import tensor as T
'''

import ConfigParser

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

class LearningConfig:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        #self.symbol_learning_rate = T.scalar()

class ILayer:
    '''
    ILayer: layers' interface.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, inputs, config_reader):
        '''
            need to initialize self.y : output symbol.
        '''
        pass

class Layer_FullConnect(ILayer):
    def __init__(self, inputs, config_reader=None):
        n_in = int( config_reader('n_in') )
        n_out = int( config_reader('n_out') )

        self.x = inputs[0]
        self.w = weight_variable([n_in, n_out])
        self.b = weight_variable([n_out])

        # active function.
        self.y = tf.matmul(self.x, self.w) + self.b

class Layer_Dot(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x1 = inputs[0]
        self.x2 = inputs[1]

        # active function.
        # x1 dot x2.
        pack = tf.pack([self.x1, self.x2])
        self.y = tf.reduce_sum( tf.reduce_prod(pack, [0]), [1], keep_dims=True )

class Layer_Sigmoid(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # calc sigmoid for each value in matrix.
        self.y = tf.sigmoid(x)

class Layer_Tanh(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # calc tanh for each value in matrix.
        self.y = tf.tanh(x)

class Layer_Norm2Cost(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        self.label = inputs[1]
        self.y = tf.reduce_mean( (self.x - self.label) ** 2 )

class ConfigNetwork:
    def __init__(self, config_file, network_name):
        self.learning_config = LearningConfig(0.1)
        #self.learning_config.symbol_learning_rate = T.scalar()

        self.__inputs = []
        self.__label = tf.placeholder(tf.float32, shape=[None, 1])
        self.__layers = []
        self.__layers_info = {}

        cp = ConfigParser.ConfigParser()
        self.__config_parser = cp
        self.__network_name = network_name
        cp.read(config_file)

        input_count = int(cp.get(network_name, 'input_count'))
        print >> sys.stderr, 'input_count = %d' % input_count
        for i in range(input_count):
            self.__inputs.append( tf.placeholder(tf.float32, shape=[None, None]) )

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

        # make network-active function.
        self.active = self.__get_layer(active_name).y
        # cost function.
        self.cost = self.__get_layer(cost_name).y
        # training function.
        self.train = tf.train.AdamOptimizer(1e-3).minimize(self.cost)

        self.session = tf.Session()
        self.session.run( tf.initialize_all_variables() )

    def predict(self, *args):
        # simple N epoch train.
        feed_dict = {}
        for idx, item in enumerate(args):
            feed_dict[ self.__inputs[idx] ] = item

        ret = self.active.eval(feed_dict=feed_dict, session=self.session)
        return ret

    def fit(self, *args):
        # simple N epoch train.
        feed_dict = {}

        # last one is label.
        for idx, item in enumerate(args):
            if idx == len(args)-1:
                feed_dict[ self.__label ] = item
            else:
                feed_dict[ self.__inputs[idx] ] = item

        print self.cost.eval(feed_dict=feed_dict, session=self.session)
            
        self.train.run(session=self.session,
                feed_dict=feed_dict)

        final_cost = self.cost.eval(feed_dict=feed_dict, session=self.session)
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







