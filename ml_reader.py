#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import sys

import numpy
import pydev
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import metrics

class NameIDTransformer:
    def __init__(self):
        self.__mapping = {}
        self.__names = []

    def read(self, name):
        if name not in self.__mapping:
            self.__mapping[name] = len(self.__names)
            self.__names.append(name)
        return self.__mapping[name]

    def id(self, name):
        return self.__mapping.get(name, None)

    def name(self, id):
        return self.__names[id]

    def debug(self):
        print >> sys.stderr, self.__names

class DataReader(object):
    def __init__(self):
        self.__seperator = ','
        self.__expect_column_count = -1
        self.__target_column = -1
        self.__name_to_id_dict = {}
        self.__ignore_first_row = False
        self.__ignore_columns = set()
        self.__target_trans = NameIDTransformer()
        self.__dv = None

        self.__X = []
        self.__Y = []

    def config(self, config_name):
        if '#' not in config_name:
            raise Exception('bad config format: [ config#secion ] is needed.')

        config_file, section = config_name.split('#')
        config = pydev.VarConfig()
        config.read(config_file)

        self.__seperator = config.get(section, 'seperator', default=',')
        self.__expect_column_count = int( config.get(section, 'fix_columns', default='-1') )
        self.__target_column = int( config.get(section, 'target_column', default='0') ) - 1
        self.__ignore_first_row = int( config.get(section, 'ignore_first_row', default='0') )

        self.__maxabs_scale = int( config.get(section, 'maxabs_scale', default='0') )

        s = config.get(section, 'name_to_id', default='')
        if s:
            self.__name_to_id_dict = dict( map(lambda x:(int(x)-1, NameIDTransformer()), s.split(',')) )

        s = config.get(section, 'ignore_columns', default='')
        if s:
            self.__ignore_columns = set( map(lambda x:int(x)-1, s.split(',')) )

        if self.__expect_column_count >= 0 and self.__target_column < 0:
            self.__target_column = self.__expect_column_count - 1

        print >> sys.stderr, 'Load config over'
        print >> sys.stderr, 'seperator : [%s]' % self.__seperator
        print >> sys.stderr, 'columns : %d (-1 : indicates from first useable row)' % self.__expect_column_count
        print >> sys.stderr, 'target  : %d (0 based, -1 indicates last column)' % self.__target_column
        print >> sys.stderr, 'name_to_id : %s' % (','.join(map(str, self.__name_to_id_dict.keys())))
        print >> sys.stderr, 'ignore_columns : %s' % (','.join(map(str, self.__ignore_columns)))
        print >> sys.stderr, 'ignore_first_row : %d' % self.__ignore_first_row

    def read(self, filename):
        self.__X = []
        self.__Y = []
        first_row = True

        dict_list = []
        for row in pydev.foreach_row(
                file(filename), 
                seperator=self.__seperator):

            if first_row and self.__ignore_first_row:
                first_row = False
                continue

            if self.__expect_column_count < 0:
                self.__expect_column_count = len(row)
                if self.__target_column < 0:
                    self.__target_column = self.__expect_column_count - 1
                print >> sys.stderr, 'columns set to %d, target:%d' % (self.__expect_column_count, self.__target_column)
            elif len(row) != self.__expect_column_count:
                    continue

            row = map(lambda x:x.strip(), row)


            # get x dict.
            x_dict = {}
            for cid, value in enumerate(row):
                if cid == self.__target_column:
                    continue
                if cid in self.__name_to_id_dict:
                    x_dict[ '#%03d' % cid ] = value
                else:
                    x_dict[ '#%03d' % cid ] = float(value)
            dict_list.append(x_dict)

            # get Y
            row[self.__target_column] = self.__target_trans.read( row[self.__target_column] )
            y = row[self.__target_column]
            self.__Y.append(y)

            '''
            for cid, trans in self.__name_to_id_dict.iteritems():
                row[cid] = trans.read(row[cid])

            row[self.__target_column] = self.__target_trans.read( row[self.__target_column] )
            y = row[self.__target_column]

            filter_row = map(
                            lambda (rid, value): float(value),
                            filter(
                                lambda (rid, value):rid not in self.__ignore_columns and rid!=self.__target_column, 
                                enumerate(row))
                            )
            x = numpy.array( filter_row )
            x = x.astype(numpy.float32)

            self.__X.append(x)
            self.__Y.append(y)
            '''
        if self.__dv is None:
            self.__dv = DictVectorizer()
            self.__X = self.__dv.fit_transform(dict_list).toarray().astype(numpy.float32)
            print >> sys.stderr, self.__dv.get_feature_names()
        else:
            self.__X = self.__dv.transform(dict_list).toarray().astype(numpy.float32)
            print >> sys.stderr, 'Use old DictVectorizer'

        if self.__maxabs_scale:
            print >> sys.stderr, 'Do maxabs_scale'
            self.__X = preprocessing.maxabs_scale(self.__X)

        #self.__target_trans.debug()
        '''
        debug = ''
        for idx in self.__X[0].nonzero()[0]:
            debug += '%d:%.1f, ' % (idx, self.__X[0][idx])
        print >> sys.stderr, debug
        '''
        print >> sys.stderr, 'Data load [ %d(records) x %d(features) ]' % (len(self.__X), len(self.__X[0]))

    @property
    def data(self):
        return self.__X

    @property
    def target(self):
        return self.__Y

    def text_format(self, v):
        if self.__dv is None:
            raise Exception('Bad text_format call() because the dict_victorizer is invalid.')
        s = []
        names = self.__dv.get_feature_names()
        for id, value in enumerate(v):
            if value != 0:
                if value == 1.0:
                    s.append(names[id])
                else:
                    s.append('%s:%.2f' % (names[id], value))
        return ', '.join(map(lambda x:x.replace('#0', '#').replace('#0', '#'), sorted(s)))

    def auc(self, X, Y, out_stream):
        if self.__dv is None:
            raise Exception('Bad text_format call() because the dict_victorizer is invalid.')
        names = self.__dv.get_feature_names()
        for id in range(len(X[0])):
            feature_name = names[id]
            pred_Y = map(lambda x:x[id], X)

            auc = metrics.roc_auc_score(Y, pred_Y)
            precision = metrics.precision_score(Y, map(lambda x:1 if x>0.5 else 0, pred_Y))
            recall = metrics.recall_score(Y, map(lambda x:1 if x>0.5 else 0, pred_Y))

            print >> out_stream, '%s\t%.3f\t%.3f\t%.3f' % (
                    feature_name.replace('#0', '#').replace('#0', '#'),
                    auc, 
                    precision,
                    recall,
                    )

if __name__=='__main__':
    pass
