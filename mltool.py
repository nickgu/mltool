#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import sys

import numpy
import sklearn
from sklearn import cross_validation

import pydev
import mlreader

if __name__=='__main__':
    arg = pydev.Arg('A simple tool for machine learning based ont scikit-learn')
    arg.str_opt('filename', 'f', 'input file of data')
    arg.str_opt('test', 't', 'data file for test')
    arg.str_opt('reader_config', 'r', 'config of reader, format: filename#section')
    arg.bool_opt('auc', 'a', 'output auc score of features in training data.')
    opt = arg.init_arg()

    train_reader = mlreader.DataReader()
    if opt.reader_config:
        train_reader.config(opt.reader_config)
    train_reader.read(opt.filename)

    train_X = train_reader.data
    train_Y = train_reader.label
    test_X = None
    test_Y = None

    test_reader = None
    if opt.test:
        test_reader = mlreader.DataReader()
        if opt.reader_config:
            test_reader.config(opt.reader_config)
        test_reader.read(opt.test)
        test_X = test_reader.data
        test_Y = test_reader.label
    else:
        train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(
                train_X, train_Y, test_size=0.3, random_state=0)

    if opt.auc:
        print >> sys.stderr, 'Calculate AUC score'
        train_reader.auc(train_X, train_Y, file('auc.txt', 'w'))
   
    from sklearn import svm
    from sklearn import linear_model
    from sklearn import neighbors
    from sklearn import naive_bayes
    from sklearn import tree
    from sklearn import ensemble
    from simple_nn import SimpleNetwork
    import nnet_tf

    models = [
        #('lr', linear_model.LogisticRegression() ),
        #('knn', neighbors.KNeighborsClassifier() ),
        #('gnb', naive_bayes.GaussianNB() ),
        #('tree', tree.DecisionTreeClassifier() ),
        #('simple_gbdt', ensemble.GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=0) ),
        #('best_gbdt', ensemble.GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=0) ),
        #('ada', ensemble.AdaBoostClassifier(n_estimators=100) ),
        #('rf', ensemble.RandomForestClassifier(n_estimators=100) ),
        #('svm', svm.SVC() ),
        #('simp_nn', SimpleNetwork(len(train_X[0]), 1, [256, 256, 128], output_01=True)),
        #('simp_nn', SimpleNetwork( len(train_X[0]), 1, [12, 12], output_01=True)),
        #('simp_nn_lr', SimpleNetwork(len(train_X[0]), [], output_01=True)),
        #('fc3_nnet', nnet_tf.ConfigNetwork('conf/net.conf', 'fc3_net', output_01=True)),
        ('mnist_fc', nnet_tf.ConfigNetwork('conf/net.conf', 'mnist_fc', output_01=True)),
        #('mnist_conv2d', nnet_tf.ConfigNetwork('conf/net.conf', 'mnist_conv2d')),
        ]

    def report(pred, label, X, reader, error_writer, out_stream):

        if len(pred.shape) > len(label.shape):
            # try to flatten.
            print >> sys.stderr, 'Force flatten! [%s] => [%s]' % (pred.shape, label.shape)
            pred.shape = label.shape

        print label[:10]
        print pred[:10]
        same_count = 0
        for p, l in zip(pred, label):
            if (l-p).dot(l-p)<1e-4:
                same_count += 1
        print same_count, len(label)

        true_negative = len(filter(lambda x:x==0, pred + label))
        true_positive = len(filter(lambda x:x==2, pred + label))
        false_positive = len(filter(lambda x:x==1, pred - label))
        false_negative = len(filter(lambda x:x==-1, pred - label))

        for i in range(len(pred)):
            if pred[i] != label[i]:
                if reader is None:
                    print >> error_writer, '%d\tP=%.1f\tT=%.1f\t%s' % (
                            i, pred[i], label[i], X[i])
                else:
                    print >> error_writer, '%d\tP=%.1f\tT=%.1f\t%s\t%s' % (
                            i, pred[i], label[i], reader.info[i], reader.text_format(X[i]))

        diff_count = len(filter(lambda x:x!=0, pred - label))
        accuracy = (len(label) - diff_count) * 100. / len(label)

        print >> out_stream, 'Accuracy : ### %.2f%% (%d/%d) ###' % (accuracy, len(label)-diff_count, len(label))
        print >> out_stream, '     P: %.3f%%   R: %.3f%%' % ( 
                                100. * true_positive / (true_positive + false_positive),
                                100. * true_positive / (true_positive + false_negative)
                                )

        print >> out_stream, ' Positive: true:%d false:%d' % (true_positive, false_positive)
        print >> out_stream, ' Negative: true:%d false:%d' % (true_negative, false_negative)

    for name, model in models:
        print >> sys.stderr, '====> model [%s] <====' % name

        model.fit(train_X, train_Y)
        print >> sys.stderr, 'Training over.'

        # TEST: training-set
        # test for best performance(ignore overfitting.).
        pred = model.predict(train_X)
        print >> sys.stderr, 'Predict [Training-set] over.'
        report(pred, train_Y, train_X, train_reader, file('error_of_train.txt', 'w'), sys.stderr)

        # TEST: test-set.
        pred = model.predict(test_X)
        print >> sys.stderr, 'Predict [Test-set] over.'
        report(pred, test_Y, test_X, test_reader, file('error_of_test.txt', 'w'), sys.stderr)


    
