#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 


import sys
from abc import ABCMeta, abstractmethod
import time


import pydev
import tensorflow as tf
import numpy
import numpy.random

'''
import numpy
import theano
from sklearn import preprocessing
from theano import tensor as T
'''

import ConfigParser

def precision_01(label, pred):
    total_count = len(label)
    if isinstance(pred[0], float):
        correct_count = len(filter(lambda x:x<.5, pred - label))
    else:
        a = [x.argmax() for x in label]
        b = [x.argmax() for x in pred]
        correct_count = len( filter(lambda x:x[0]==x[1], zip(a,b))) 
    precision = correct_count * 1. / total_count
    return precision, correct_count, total_count

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

'''
class LearningConfig:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        #self.symbol_learning_rate = T.scalar()
'''

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

class Layer_OpFullConnect(ILayer):
    ''' 
        Y = Op(W*X + b) 
        param:
            n_in    : input_count
            n_out   : output_count
            op      : operator: sigmoid|tanh|softmax|relu
    '''
    def __init__(self, inputs, config_reader=None):
        n_in = int( config_reader('n_in') )
        n_out = int( config_reader('n_out') )
        op = config_reader('op')

        Fdict = {
            'sigmoid'   : tf.sigmoid,
            'tanh'      : tf.tanh,
            'softmax'   : tf.nn.softmax,
            'relu'      : tf.nn.relu
                }
        F = Fdict.get(op, None)

        self.x = inputs[0]
        self.w = weight_variable([n_in, n_out])
        self.b = bias_variable([n_out])

        # active function.
        if F is None:
            print >> sys.stderr, 'Warning: FullConnectOp with no OP. [%s]' % op
            self.y = tf.matmul(self.x, self.w) + self.b
        else:
            self.y = F( tf.matmul(self.x, self.w) + self.b )


class Layer_FullConnect(ILayer):
    ''' 
        Y = W*X + b 
        param:
            n_in    : input_count
            n_out   : output_count
    '''
    def __init__(self, inputs, config_reader=None):
        n_in = int( config_reader('n_in') )
        n_out = int( config_reader('n_out') )

        self.x = inputs[0]
        self.w = weight_variable([n_in, n_out])
        self.b = bias_variable([n_out])

        # active function.
        self.y = tf.matmul(self.x, self.w) + self.b

class Layer_Dot(ILayer):
    '''
        Y = X1 * X2
    '''
    def __init__(self, inputs, config_reader=None):

        # active function.
        # x1 dot x2.
        pack = tf.pack([self.x1, self.x2])
        self.y = tf.reduce_sum( tf.reduce_prod(pack, [0]), [1], keep_dims=True )

class Layer_Sigmoid(ILayer):
    ''' Y = sigmoid(X) '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # calc sigmoid for each value in matrix.
        self.y = tf.sigmoid(self.x)

class Layer_Tanh(ILayer):
    ''' Y = tanh(X) '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # calc tanh for each value in matrix.
        self.y = tf.tanh(self.x)

class Layer_Relu(ILayer):
    ''' Y = relu(X) '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # calc relu for each value in matrix.
        self.y = tf.nn.relu(self.x)

class Layer_Norm2Cost(ILayer):
    ''' Y = |x1 - x2|_2  ''' 
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        self.label = inputs[1]
        self.y = tf.reduce_mean( (self.x - self.label) ** 2 )

class Layer_DropOut(ILayer):
    ''' 
        Y = drop_out(X) 
        param:
            prob    : selective ratio.
    '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        prob = float( config_reader('prob') )
        self.y = tf.nn.dropout(self.x, keep_prob=prob)

class Layer_Conv2D(ILayer):
    ''' 
        Y = conv2d(X)
        param:
            shape = window_x, window_y, in_chan, out_chan
    '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # x, y, in_chan, out_chan
        self.shape = map(int, config_reader('shape').split(','))
        self.W = weight_variable(self.shape)
        self.b = bias_variable([self.shape[3]])
        print >> sys.stderr, 'Conv-shape : %s' % (self.shape)

        # out_chan
        self.y = tf.nn.relu( 
                    tf.nn.conv2d(
                        self.x, 
                        self.W, 
                        strides=[1, 1, 1, 1], 
                        padding='SAME'
                    ) 
                    + self.b 
                )

class Layer_PoolingMax(ILayer):
    '''
        Y = pooling_max(X)
        param:
            size    :  window of [size x size]
    '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        self.pooling_size = int( config_reader('size') )
        print 'PoolingSize=%d' % self.pooling_size
        self.y = tf.nn.max_pool(self.x, 
                    ksize=[1, self.pooling_size, self.pooling_size, 1],
                    strides=[1, self.pooling_size, self.pooling_size, 1], 
                    padding='SAME')

class Layer_Conv2DPooling(ILayer):
    ''' 
        Y = pooling( conv2d(X) )
        param:
            shape       : window_x, window_y, in_chan, out_chan
            pool_type   : [max]
            pool_size   : window of [size x size] in pooling.
    '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # x, y, in_chan, out_chan
        self.shape = map(int, config_reader('shape').split(','))
        self.W = weight_variable(self.shape)
        self.b = bias_variable([self.shape[3]])

        self.pooling_size = int( config_reader('pool_size') )
        self.pooling_type = config_reader('pool_type')
        print >> sys.stderr, 'Conv-shape : %s' % (self.shape)

        # out_chan
        conv_out = tf.nn.relu( 
                    tf.nn.conv2d(
                        self.x, 
                        self.W, 
                        strides=[1, 1, 1, 1], 
                        padding='SAME'
                    ) 
                    + self.b 
                )

        print 'PoolingSize=%d' % self.pooling_size
        if self.pooling_type == 'max':
            self.y = tf.nn.max_pool(conv_out, 
                        ksize=[1, self.pooling_size, self.pooling_size, 1],
                        strides=[1, self.pooling_size, self.pooling_size, 1], 
                        padding='SAME')
        else:
            print >> sys.stderr, 'Bad pooling type: %s' % self.pooling_type

class Layer_Reshape(ILayer):
    '''
        Y = reshape(X)
        param:
            shape : new shape.
    '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        self.shape = map(int, config_reader('shape').split(','))
        print >> sys.stderr, 'Reshape : %s' % (self.shape)
        self.y = tf.reshape(self.x, self.shape)

class Layer_Softmax(ILayer):
    ''' Y = softmax(X)  '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        self.y = tf.nn.softmax(self.x)

class ConfigNetwork:
    def __init__(self, config_file, network_name, output_01=False):
        self.__LayerCreator__ = {
                'full_connect'      : Layer_FullConnect,
                'full_connect_op'   : Layer_OpFullConnect,
                'dot'               : Layer_Dot,
                'norm2'             : Layer_Norm2Cost,
                'sigmoid'           : Layer_Sigmoid,
                'softmax'           : Layer_Softmax,
                'tanh'              : Layer_Tanh,
                'relu'              : Layer_Relu,
                'conv2d'            : Layer_Conv2D,
                'maxpool'           : Layer_PoolingMax,
                'conv2d_pool'       : Layer_Conv2DPooling,
                'reshape'           : Layer_Reshape,
                'dropout'           : Layer_DropOut,
            }

        self.__output_01 = output_01

        self.__inputs = []
        self.__label = tf.placeholder(tf.float32, shape=[None, None])
        self.__layers = []
        self.__layers_info = {}

        cp = ConfigParser.ConfigParser()
        self.__config_parser = cp
        self.__network_name = network_name
        cp.read(config_file)

        # batch_size.
        self.__batch_size = int(pydev.config_default_get(cp, network_name, 'batch_size', 50))

        input_count = int(cp.get(network_name, 'input_count'))
        print >> sys.stderr, 'input_count = %d' % input_count
        for i in range(input_count):
            self.__inputs.append( tf.placeholder(tf.float32, shape=[None, None]) )

        layer_names = cp.get(network_name, 'layers').split(',')
        active_name = cp.get(network_name, 'active').strip()
        cost_name = cp.get(network_name, 'cost').strip()

        self.__batch_size = int( pydev.config_default_get(cp, network_name, 'batch_size', 50) )
        self.__learning_rate = float( pydev.config_default_get(cp, network_name, 'learning_rate', 1e-3) )
        print >> sys.stderr, 'LearningRate : %.5f' % self.__learning_rate
        self.__epoch = int( pydev.config_default_get(cp, network_name, 'epoch', 1000) )
        print >> sys.stderr, 'Epoch : %d' % self.__epoch
   
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
        self.train = tf.train.AdamOptimizer( self.__learning_rate ).minimize(self.cost)

        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session.run( tf.initialize_all_variables() )

    def predict(self, X):
        ret = None
        batch_size = self.__batch_size
        for b in range(0, len(X), batch_size):
            feed_dict = {}
            feed_dict[ self.__inputs[0] ] = X[b:b+batch_size]

            partial_ret = self.active.eval(feed_dict=feed_dict, session=self.session)
            if self.__output_01:
                for x in numpy.nditer(partial_ret, op_flags=['readwrite']):
                    x[...] = 1. if x[...]>=.5 else 0.
            if ret is None:
                ret = partial_ret
            else:
                ret = numpy.append(ret, partial_ret, axis=0)
        return ret

    '''
    def predict(self, *args):
        feed_dict = {}
        for idx, item in enumerate(args):
            feed_dict[ self.__inputs[idx] ] = item

        ret = self.active.eval(feed_dict=feed_dict, session=self.session)
        if self.__output_01:
            for x in numpy.nditer(ret, op_flags=['readwrite']):
                x[...] = 1. if x[...]>=.5 else 0.
        return ret
    '''

    def fit(self, X, Y):
        # simple train.
        tm = time.time()
        for it in range(self.__epoch):
            idx_list = []
            for i in range(self.__batch_size):
                idx_list.append( numpy.random.choice(range(len(X))) )
            sub_X = numpy.array( map(lambda i:X[i], idx_list) )
            sub_Y = numpy.array( map(lambda i:Y[i], idx_list) )

            self.fit_one_batch(sub_X, sub_Y)
            if it % 100 == 0:
                cost = self.calc_cost(sub_X, sub_Y)
                diff_tm = time.time() - tm
                print >> sys.stderr, 'iter=%d, cost=%.5f, tm=%.3f' % (it, cost, diff_tm)
                tm = time.time()

    def calc_cost(self, *args):
        # simple N epoch train.
        feed_dict = {}
        # last one is label.
        for idx, item in enumerate(args):
            if idx == len(args)-1:
                if len(item.shape) == 1:
                    item.shape = (item.shape[0], 1)
                feed_dict[ self.__label ] = item
            else:
                feed_dict[ self.__inputs[idx] ] = item

        # debug cost.
        y = self.active.eval(feed_dict=feed_dict, session=self.session)
        #print 'y[0]:' +  str(y[0]) 
        #print 'label:' + str(feed_dict[ self.__label ][0])

        final_cost = self.cost.eval(feed_dict=feed_dict, session=self.session)
        return final_cost

    def fit_one_batch(self, *args):
        # simple N epoch train.
        feed_dict = {}
        # last one is label.
        for idx, item in enumerate(args):
            if idx == len(args)-1:
                if len(item.shape) == 1:
                    item.shape = (item.shape[0], 1)
                feed_dict[ self.__label ] = item
            else:
                feed_dict[ self.__inputs[idx] ] = item

        self.train.run(session=self.session,
                feed_dict=feed_dict)

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
        creator = self.__LayerCreator__[ltype]
        layer = creator(inputs, config_reader)

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







