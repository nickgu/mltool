#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 


import sys
from abc import ABCMeta, abstractmethod
import time

import numpy
import numpy.random
import ConfigParser
import random

import pydev
import tensorflow as tf

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

def weight_variable(shape, l2_weight=0.000, stddev=0.01):
    initial = tf.truncated_normal(shape, stddev=stddev)
    weight = tf.Variable(initial)

    if l2_weight > 0:
        losses = tf.mul(tf.nn.l2_loss(weight), l2_weight, 'weighted_loss')
        tf.add_to_collection('losses', losses)

    return weight

def bias_variable(shape, init=0.0):
    initial = tf.constant(init, shape=shape)
    bias = tf.Variable(initial)
    return bias

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
            l2_wd   : L2 weight.
    '''
    def __init__(self, inputs, config_reader=None):
        n_in = int( config_reader('n_in') )
        n_out = int( config_reader('n_out') )
        op = config_reader('op')
        self.l2wd = float(config_reader('l2wd', 0.0))
        self.bias_init = float(config_reader('bias_init', 0.0))
        self.weight_stddev = float(config_reader('weight_stddev', 0.01))
        
        pydev.log('l2 weight : %f' % self.l2wd )
        pydev.log('weight_stddev : %f' % self.weight_stddev)
        pydev.log('bias_init : %f' % self.bias_init)

        Fdict = {
            'sigmoid'   : tf.sigmoid,
            'tanh'      : tf.tanh,
            'softmax'   : tf.nn.softmax,
            'relu'      : tf.nn.relu
                }
        F = Fdict.get(op, None)

        self.x = inputs[0]
        self.w = weight_variable([n_in, n_out], l2_weight=self.l2wd, stddev=self.weight_stddev)
        self.b = bias_variable([n_out], init=self.bias_init)

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
            l2wd    : l2 weight.
    '''
    def __init__(self, inputs, config_reader=None):
        n_in = int( config_reader('n_in') )
        n_out = int( config_reader('n_out') )
        self.l2wd = float(config_reader('l2wd', 0.0))
        self.bias_init = float(config_reader('bias_init', 0.0))
        self.weight_stddev = float(config_reader('weight_stddev', 0.01))
        
        pydev.log('l2 weight : %f' % self.l2wd )
        pydev.log('weight_stddev : %f' % self.weight_stddev)
        pydev.log('bias_init : %f' % self.bias_init)

        self.x = inputs[0]
        self.w = weight_variable([n_in, n_out], l2_weight=self.l2wd, stddev=self.weight_stddev)
        self.b = bias_variable([n_out], init=self.bias_init)

        # active function.
        self.y = tf.matmul(self.x, self.w) + self.b

class Layer_LocalResponseNormalization(ILayer):
    ''' 
    '''
    #TODO make config.
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # active function.
        #self.y = tf.nn.local_response_normalization(self.x)
        self.y = tf.nn.local_response_normalization(self.x,
                4, 
                bias=1.0, 
                alpha=0.001 / 9.0, 
                beta=0.75)

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

class Layer_SoftmaxEntropyCost(ILayer):
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        self.label = inputs[1]
        self.y = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(self.x, self.label) )

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
        self.shape = map(int, config_reader('shape').split(','))
        self.l2wd = float(config_reader('l2wd', 0.0))

        # x, y, in_chan, out_chan
        self.W = weight_variable(self.shape, l2_weight=self.l2wd)
        self.b = bias_variable([self.shape[3]])

        print >> sys.stderr, 'l2 weight : %f' % self.l2wd
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

class Layer_Pooling(ILayer):
    '''
        Y = pooling(X)
        param:
            pool_type   : [max, avg]
            pool_size   : window of [size x size] in pooling.
            pool_strides: strides of [size x size] in pooling.
    '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        self.pooling_size = int( config_reader('pool_size') )
        self.pooling_strides = int( config_reader('pool_strides') )
        self.pooling_type = config_reader('pool_type')

        print >> sys.stderr, 'PoolingSize=%d' % self.pooling_size
        if self.pooling_type == 'max':
            self.y = tf.nn.max_pool(self.x, 
                        ksize=[1, self.pooling_size, self.pooling_size, 1],
                        strides=[1, self.pooling_strides, self.pooling_strides, 1], 
                        padding='SAME')
        elif self.pooling_type == 'avg':
            self.y = tf.nn.avg_pool(self.x, 
                        ksize=[1, self.pooling_size, self.pooling_size, 1],
                        strides=[1, self.pooling_strides, self.pooling_strides, 1], 
                        padding='SAME')
        else:
            print >> sys.stderr, 'Bad pooling type: %s' % self.pooling_type

class Layer_Conv2DPooling(ILayer):
    ''' 
        Y = pooling( relu(conv2d(X)) )
        param:
            shape       : window_x, window_y, in_chan, out_chan
            pool_type   : [max, avg]
            pool_size   : window of [size x size] in pooling.
            pool_strides: strides of [size x size] in pooling.
    '''
    def __init__(self, inputs, config_reader=None):
        self.x = inputs[0]
        # x, y, in_chan, out_chan
        self.shape = map(int, config_reader('shape').split(','))
        self.l2wd = float(config_reader('l2wd', 0.0))

        print >> sys.stderr, 'l2 weight : %f' % self.l2wd

        self.W = weight_variable(self.shape, l2_weight=self.l2wd)
        self.b = bias_variable([self.shape[3]])

        self.pooling_size = int( config_reader('pool_size') )
        self.pooling_strides = int( config_reader('pool_strides') )
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
                        strides=[1, self.pooling_strides, self.pooling_strides, 1], 
                        padding='SAME')
        elif self.pooling_type == 'avg':
            self.y = tf.nn.avg_pool(conv_out, 
                        ksize=[1, self.pooling_size, self.pooling_size, 1],
                        strides=[1, self.pooling_strides, self.pooling_strides, 1], 
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


class MovingGradientDescentOptimizer:
    def __init__(self, lr, decay=0.9999):
        self.__lr = lr
        self.__decay = decay

    def minimize(self, cost, global_step):
        # Compute gradients.
        opt = tf.train.GradientDescentOptimizer(self.__lr)
        grads = opt.compute_gradients(cost)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(self.__decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            self.train = tf.no_op(name='train')
        return self.train

class ConfigNetwork:
    def __init__(self, config_file, network_name, output_01=False):
        self.__LayerCreator__ = {
                'full_connect'      : Layer_FullConnect,
                'full_connect_op'   : Layer_OpFullConnect,
                'dot'               : Layer_Dot,
                'norm2'             : Layer_Norm2Cost,
                'softmax_entropy'   : Layer_SoftmaxEntropyCost,
                'sigmoid'           : Layer_Sigmoid,
                'softmax'           : Layer_Softmax,
                'tanh'              : Layer_Tanh,
                'relu'              : Layer_Relu,
                'conv2d'            : Layer_Conv2D,
                'pooling'           : Layer_Pooling,
                'conv2d_pool'       : Layer_Conv2DPooling,
                'reshape'           : Layer_Reshape,
                'dropout'           : Layer_DropOut,
                'local_norm'        : Layer_LocalResponseNormalization,
            }

        self.__output_01 = output_01

        self.__layers = []
        self.__layers_info = {}

        cp = ConfigParser.ConfigParser()
        self.__config_parser = cp
        self.__network_name = network_name
        cp.read(config_file)

        # read inputs and label.
        self.__inputs = self.__placeholders_read( 
                pydev.config_default_get(cp, network_name, 'input_def', 'f:2') )
        self.__label = self.__single_placeholder_read( 
                pydev.config_default_get(cp, network_name, 'label_def', 'f:2') )
        pydev.log('Inputs: %s' % self.__inputs)
        pydev.log('Labels  %s' % self.__label)

        # batch_size.
        self.__batch_size = int(pydev.config_default_get(cp, network_name, 'batch_size', 50))

        layer_names = cp.get(network_name, 'layers').split(',')
        active_name = cp.get(network_name, 'active').strip()
        cost_name = cp.get(network_name, 'cost').strip()

        global_step = tf.Variable(0, trainable=False)

        self.__batch_size = int( pydev.config_default_get(cp, network_name, 'batch_size', 50) )
        self.__epoch = int( pydev.config_default_get(cp, network_name, 'epoch', 10) )
        print >> sys.stderr, 'Epoch     : %d' % self.__epoch
        print >> sys.stderr, 'BatchSize : %d' % self.__batch_size
   
        for layer_name in layer_names:
            layer_type = cp.get(network_name, '%s.type' % layer_name)
            input_names = cp.get(network_name, '%s.input' % layer_name).split(',')

            # type, inputs, layer_refer.
            self.__layers_info[layer_name] = [layer_type, input_names, None]
        
        self.__layer_has_receiver = set()
        for name in layer_names:
            self.__init_layer(name)

        # check receivers.
        warning_layers = []
        for name in layer_names:
            if name not in self.__layer_has_receiver:
                warning_layers.append(name)
        if len(warning_layers)>0:
            pydev.err('Layers of no receivers, check it : %s' % ','.join(warning_layers))

        # make network-active function.
        self.active = self.__get_layer(active_name).y
        # cost function.
        self.cost = self.__get_layer(cost_name).y

        tf.add_to_collection('losses', self.__get_layer(cost_name).y)
        self.cost = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # learning method and learning rate.
        self.__learner = pydev.config_dict_get(cp, network_name, 'learner', 
                {
                    'gradient'          : tf.train.GradientDescentOptimizer,
                    'movingGradient'    : MovingGradientDescentOptimizer,
                    'adam'              : tf.train.AdamOptimizer,
                }, default_key='gradient'
                )
        
        self.__lr_value = float( pydev.config_default_get(cp, network_name, 'learning_rate', 1e-3) )
        self.__lr_decay_ratio = float( pydev.config_default_get(cp, network_name, 'learning_decay_ratio', 0.96) )
        self.__lr_decay_step = float( pydev.config_default_get(cp, network_name, 'learning_decay_step', 300) )
        self.__lr_tensor = pydev.config_dict_get(cp, network_name, 'learning_rate_type', 
                {
                    'fixed'             : tf.Variable(self.__lr_value),
                    'exponential_decay' : tf.train.exponential_decay(self.__lr_value, global_step, 
                        self.__lr_decay_step, self.__lr_decay_ratio, staircase=True)
                }, default_key = 'fixed'
            )

        pydev.log('learner : %s' % self.__learner)
        pydev.log('lr_type : %s' % self.__lr_tensor)
        pydev.log('lr_value: %s' % self.__lr_value)
        pydev.log('lr_step : %s' % self.__lr_decay_step)
        pydev.log('lr_ratio: %s' % self.__lr_decay_ratio)

        tf.scalar_summary('train/learning_rate', self.__lr_tensor)

        # generate training function.
        self.train = self.__learner( self.__lr_tensor ).minimize(self.cost, global_step=global_step)

        # Generate moving averages of all losses and associated summaries.
        self.__add_loss_summaries(self.cost)
        self.__train_summary_merged = tf.merge_all_summaries()

        self.session = tf.Session()
        #self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    def save(self, model):
        saver = tf.train.Saver()
        saver.save(self.session, model)

    def load(self, model):
        saver = tf.train.Saver()
        saver.restore(self.session, model)

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

    def fit(self, X, Y, callback=None, callback_iteration=100, preprocessor=None):
        '''
            X : training X
            Y : training Y
            callback : callback when training. callback type: callback(predict_function)
            callback_interval : interval to call back (total 1.0)
            callback_iteration : callback each N iterations.
            preprocess : preprocess tensor for each (x, y)
        '''

        # make shuffle and preprocess graph.
        holder_x = tf.constant(X)
        holder_y = tf.constant(Y)

        queue_x, queue_y = tf.train.slice_input_producer([holder_x, holder_y])
        if preprocessor is not None:
            queue_x, queue_y = preprocessor(queue_x, queue_y)
        batch_x, batch_y = tf.train.batch([queue_x, queue_y], batch_size=self.__batch_size, num_threads=4)

        # initialize the training summary.
        ts = time.asctime().replace(' ', '_')
        self.__train_writer = tf.train.SummaryWriter(
                            './tensorboard_logs/%s/%s' % (self.__network_name, ts),
                            self.session.graph)

        # init all variables.
        self.session.run( tf.initialize_all_variables() )

        # simple train.
        data_size = len(X)
        iteration_count = (self.__epoch * data_size) // self.__batch_size
        print >> sys.stderr, 'Iteration=%d (batchsize=%d, epoch=%d, datasize=%d)' % (
                iteration_count, self.__batch_size, self.__epoch, data_size)
        self.__current_iteration = 0

        last_percentage = 0
        last_callback_iteration = 0
        begin_time = time.time()
        with self.session.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord = coord)

            for it in xrange( 1, iteration_count+1 ):
                # training code.
                self.__current_iteration = it
                
                # run back data and fit one batch.
                sub_x, sub_y = self.session.run([batch_x, batch_y])

                cost, summary_info, lr = self.fit_one_batch(sub_x, sub_y)
                self.__train_writer.add_summary(summary_info, self.__current_iteration)

                # Report code.
                percentage = it * 1. / iteration_count
                cost_time = time.time() - begin_time
                remain_time = cost_time / percentage - cost_time

                sys.stderr.write('%cProgress: %3.1f%% [%s/%s] [iter=%7d loss=%.4f lr=%f ips=%.2f]' % (
                    13, 
                    percentage * 100., 
                    pydev.format_time(cost_time),
                    pydev.format_time(remain_time),
                    it, cost, lr, 
                    it / (time.time()-begin_time) 
                    ))

                # call back the reporter and tester.
                if callback:
                    if it - last_callback_iteration >= callback_iteration:
                        callback(self.predict, self.__train_writer, self.__current_iteration)
                        last_callback_iteration = it

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
        final_cost = self.cost.eval(feed_dict=feed_dict, session=self.session)
        return final_cost

    def fit_one_batch(self, *args):
        # simple N epoch train.
        feed_dict = {}
        # last one is label.
        for idx, item in enumerate(args):
            if idx == len(args)-1:
                feed_dict[ self.__label ] = item
            else:
                feed_dict[ self.__inputs[idx] ] = item

        summary_info, cost, lr, _ = self.session.run(
                [self.__train_summary_merged, self.cost, self.__lr_tensor, self.train], 
                feed_dict=feed_dict)
        return cost, summary_info, lr

    def __init_layer(self, name):
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

                # sub layer has receiver
                self.__layer_has_receiver.add(sub_name)

        config_reader = self.__layer_config_reader(name)
        creator = self.__LayerCreator__[ltype]
        print >> sys.stderr, '---- Init layer [%s] of type %s ----' % (name, ltype)
        layer = creator(inputs, config_reader)

        # for random access.
        self.__layers_info[name][2] = layer
        # for iteration.
        self.__layers.append(layer)
        return layer

    def __placeholders_read(self, config):
        ret = []
        for elem in config.split(','):
            ret.append(self.__single_placeholder_read(elem))
        return ret

    def __single_placeholder_read(self, elem):
        type_s, shape_s = elem.split(':')
        t = {
            'i': tf.int32,
            'f': tf.float32,
                }.get(type_s, tf.float32)
        shape = [None] * int(shape_s)
        return tf.placeholder(t, shape)

    def __get_layer(self, name):
        ltype, inames, layer = self.__layers_info.get(name, ['', [], None])
        return layer

    def __layer_config_reader(self, layer_name):
        return lambda opt,default_value=None: pydev.config_default_get(self.__config_parser, self.__network_name, ('%s.'%layer_name) + opt, default_value)

    def __add_loss_summaries(self, total_loss):
        tf.scalar_summary('train/total_loss', total_loss)

if __name__=='__main__':
    pass







