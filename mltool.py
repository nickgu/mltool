#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import sys

import numpy
import sklearn
from sklearn import cross_validation

import pydev
import ml_reader

if __name__=='__main__':
    arg = pydev.Arg('A simple tool for machine learning based ont scikit-learn')
    arg.str_opt('filename', 'f', 'input file of data')
    arg.str_opt('test', 't', 'data file for test')
    arg.str_opt('reader_config', 'r', 'config of reader, format: filename#section')
    arg.bool_opt('auc', 'a', 'output auc score of features in training data.')
    opt = arg.init_arg()

    reader = ml_reader.DataReader()
    if opt.reader_config:
        reader.config(opt.reader_config)
    reader.read(opt.filename)

    train_X = reader.data
    train_Y = reader.target
    test_X = None
    test_Y = None

    if opt.test:
        reader.read(opt.test)
        test_X = reader.data
        test_Y = reader.target
    else:
        train_X, test_X, train_Y, test_Y = cross_validation.train_test_split(
                train_X, train_Y, test_size=0.3, random_state=0)

    if opt.auc:
        print >> sys.stderr, 'Calculate AUC score'
        reader.auc(train_X, train_Y, file('auc.txt', 'w'))
   
    from sklearn import svm
    from sklearn import linear_model
    from sklearn import neighbors
    from sklearn import naive_bayes
    from sklearn import tree
    from sklearn import ensemble
    from simple_nn import LogisticClassifier
    from simple_nn import SimpleNetwork

    models = [
        #('lr', linear_model.LogisticRegression() ),
        #('knn', neighbors.KNeighborsClassifier() ),
        #('gnb', naive_bayes.GaussianNB() ),
        #('tree', tree.DecisionTreeClassifier() ),
        #('simple_gbdt', ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0) ),
        #('best_gbdt', ensemble.GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=0) ),
        #('ada', ensemble.AdaBoostClassifier(n_estimators=100) ),
        ('rf', ensemble.RandomForestClassifier(n_estimators=100) ),
        #('svm', svm.SVC() ),

        #('mylog', LogisticClassifier(dim=len(train_X[0]), output_01=True) ),
        #('simp_nn', SimpleNetwork(len(train_X[0]), [256, 256, 128], output_01=True)),
        #('simp_nn_lr', SimpleNetwork(len(train_X[0]), [], output_01=True)),
        ]

    def report(pred, target, X, reader, error_writer, out_stream):
        true_negative = len(filter(lambda x:x==0, pred + target))
        true_positive = len(filter(lambda x:x==2, pred + target))
        false_positive = len(filter(lambda x:x==1, pred - target))
        false_negative = len(filter(lambda x:x==-1, pred - target))

        for i in range(len(pred)):
            if pred[i] != target[i]:
                print >> error_writer, '%d\tP=%.1f\tT=%.1f\t%s' % (
                        i, pred[i], target[i], reader.text_format(X[i]))

        diff_count = len(filter(lambda x:x!=0, pred - target))
        precision = (len(target) - diff_count) * 100. / len(target)

        print >> out_stream, 'Accuracy : ### %.2f%% (%d/%d) ###' % (precision, len(target)-diff_count, len(target))
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
        report(pred, train_Y, train_X, reader, file('error_of_train.txt', 'w'), sys.stderr)

        # TEST: test-set.
        pred = model.predict(test_X)
        print >> sys.stderr, 'Predict [Test-set] over.'
        report(pred, test_Y, test_X, reader, file('error_of_test.txt', 'w'), sys.stderr)


    
