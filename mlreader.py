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
    '''
    allocate continuous ID to concrete-value.
    Usage:
        t = NameIDTransformer()
        # first allocate.
        id = t.allocate_id(name)
        # search:
        id = t.id(name)
        name = t.name(id)
    '''

    def __init__(self):
        self.__mapping = {}
        self.__names = []

    def allocate_id(self, name):
        if name not in self.__mapping:
            self.__mapping[name] = len(self.__names)
            self.__names.append(name)
        return self.__mapping[name]

    def id(self, name):
        return self.__mapping.get(name, None)

    def name(self, id):
        return self.__names[id]

    def size(self):
        return len(self.__names)

    def debug(self):
        print >> sys.stderr, self.__names

class DataReader(object):
    '''
    Read data from files.
    Usage:
        reader = DataReader()
        # if needed.
        reader.config(config_filename)
        reader.read(data_file)
        x, y = reader.data, reader.target
    '''
    (DenseValue, IVSparse) = range(2)

    def __init__(self):
        self.__seperator = ','
        self.__expect_column_count = -1
        self.__target_column = -1
        self.__ignore_first_row = False

        # TODO: this is not work for iv-sparse format.
        self.__ignore_columns = set()

        # feature_id need to be concrete.
        self.__concrete_ids = set()
        self.__concrete_target = False
        self.__feature_trans = NameIDTransformer()
        self.__target_trans = NameIDTransformer()
        self.__maxabs_scale = 0

        # default mode 
        self.__row_mode = DataReader.DenseValue
        self.__use_id_mapping = False

        self.__X = []
        self.__Y = []
        self.__info = []

    def config(self, config_name):
        if '#' not in config_name:
            raise Exception('bad config format: [ config#secion ] is needed.')

        config_file, section = config_name.split('#')
        config = pydev.VarConfig()
        config.read(config_file)

        self.__seperator = config.get(section, 'seperator', default=',')
        if self.__seperator == 'tab':
            self.__seperator = '\t'
        self.__expect_column_count = int( config.get(section, 'fix_columns', default='-1') )
        self.__target_column = int( config.get(section, 'target_column', default='0') ) - 1
        self.__ignore_first_row = int( config.get(section, 'ignore_first_row', default='0') )

        self.__maxabs_scale = int( config.get(section, 'maxabs_scale', default='0') )
        self.__concrete_target = int( config.get(section, 'concrete_target', default='0') )

        s = config.get(section, 'concrete_ids', default='')
        if s:
            self.__concrete_ids = set( map(lambda x:int(x)-1, s.split(',')) )


        s = config.get(section, 'ignore_columns', default='')
        if s:
            self.__ignore_columns = set( map(lambda x:int(x)-1, s.split(',')) )

        if self.__expect_column_count >= 0 and self.__target_column < 0:
            self.__target_column = self.__expect_column_count - 1

        print >> sys.stderr, 'Load config over'
        print >> sys.stderr, 'seperator : [%s]' % self.__seperator
        print >> sys.stderr, 'columns : %d (-1 : indicates from first useable row)' % self.__expect_column_count
        print >> sys.stderr, 'target  : %d (0 based, -1 indicates last column)' % self.__target_column
        print >> sys.stderr, 'concrete_ids : %s' % (','.join(map(str, self.__concrete_ids)))
        print >> sys.stderr, 'concrete_target : %d' % (self.__concrete_target)
        print >> sys.stderr, 'ignore_columns : %s' % (','.join(map(str, self.__ignore_columns)))
        print >> sys.stderr, 'ignore_first_row : %d' % self.__ignore_first_row

    def read(self, filename):
        self.__X = []
        self.__Y = []
        self.__info = []
        first_row = True

        fd = file(filename)
        progress = pydev.FileProgress(fd, filename)
        raw_X = []
        for row in pydev.foreach_row(fd, seperator=self.__seperator):
            progress.check_progress()

            # whether to ignore first row.
            if first_row and self.__ignore_first_row:
                first_row = False
                continue

            # check column count.
            if self.__expect_column_count < 0:
                self.__expect_column_count = len(row)
                if self.__target_column < 0:
                    self.__target_column = self.__expect_column_count - 1
                print >> sys.stderr, 'columns set to %d, target:%d' % (self.__expect_column_count, self.__target_column)
            elif len(row) != self.__expect_column_count:
                    continue

            # strip each columns.
            row = map(lambda x:x.strip(), row)

            # get x dict.
            id_value = []
            v_size = 0
            ignored_info = []
            for rid, value in enumerate(row):
                # continue if target columns.
                if rid == self.__target_column:
                    continue
                # continue if filter columns.
                if rid in self.__ignore_columns:
                    ignored_info.append(value)
                    continue

                # dense and id-value-sparse
                if self.__row_mode == DataReader.DenseValue:
                    cid = rid
                elif self.__row_mode == DataReader.IVSparse:
                    cid, value = value.split(':')
                    cid = int(cid)

                if cid in self.__concrete_ids:
                    # one-hot representation for key.
                    # feature = id-value : 1
                    fid, value = self.__feature_trans.allocate_id('#%03d:%s' % (cid, value)), 1
                else:
                    # feature = id : value
                    fid, value = self.__feature_trans.allocate_id('#%03d' % (cid)), float(value)

                id_value.append( (fid, value) )
                if v_size < fid+1:
                    v_size = fid+1 

            x = numpy.ndarray(shape=(v_size,))
            x.fill(0)
            for fid, value in id_value:
                x[fid] = float(value)

            raw_X.append(x)

            # get Y
            if self.__concrete_target:
                row[self.__target_column] = self.__target_trans.allocate_id( row[self.__target_column] )
            y = row[self.__target_column]
            self.__Y.append(y)

            self.__info.append( self.__seperator.join(ignored_info) )

        progress.end_progress()
        
        # resize for each X.
        x_size = self.__feature_trans.size()
        for x in raw_X:
            new_x = numpy.ndarray(shape=(x_size,), dtype=numpy.float32)
            new_x.fill(0)
            new_x[:x.shape[0]] = x
            self.__X.append( new_x )

        # transform X to numpy.ndarray
        self.__X = numpy.array(self.__X)

        # preprocessing.
        if self.__maxabs_scale:
            print >> sys.stderr, 'Do maxabs_scale'
            self.__X = preprocessing.maxabs_scale(self.__X)

        # make Y as ndarray
        self.__Y = numpy.array(self.__Y).astype(numpy.float32)

        #self.__feature_trans.debug()
        #self.__target_trans.debug()

        print >> sys.stderr, 'Data load [ %d(records) x %d(features) ]' % (len(self.__X), len(self.__X[0]))

    @property
    def data(self):
        return self.__X

    @property
    def label(self):
        return self.__Y

    @property
    def info(self):
        return self.__info

    def text_format(self, v):
        s = []
        for id, value in enumerate(v):
            name =self.__feature_trans.name(id)
            if value != 0:
                if value == 1.0:
                    s.append( name )
                else:
                    s.append('%s:%.2f' % (name, value))
        return ', '.join(sorted(s))

    def auc(self, X, Y, out_stream):
        ''' TODO: move it to tools.
        '''
        for id in range(len(X[0])):
            feature_name =self.__feature_trans.name(id)
            pred_Y = map(lambda x:x[id], X)

            auc = metrics.roc_auc_score(Y, pred_Y)
            precision = metrics.precision_score(Y, map(lambda x:1 if x>0.5 else 0, pred_Y))
            recall = metrics.recall_score(Y, map(lambda x:1 if x>0.5 else 0, pred_Y))

            print >> out_stream, '%s\t%.3f\t%.3f\t%.3f' % (
                    feature_name,
                    auc, 
                    precision,
                    recall,
                    )

if __name__=='__main__':
    # test for reader.
    # mlreader.py <filename> <config> <section>
    if len(sys.argv)==4:
        filename, config, section = sys.argv[1:]
        reader = DataReader()
        reader.config(config + '#' + section)
        reader.read(filename)

    else:
        filename = sys.argv[1]
        reader = DataReader()
        reader.read(filename)




