#! /bin/env python
# encoding=utf-8
# gusimiu@baidu.com
# 

import network_tensorflow
from ml_reader import DataReader

def test_Network():
    R = DataReader()
    R.read('matrix.txt')

    A = R.data[ : , :5]
    B = R.data[ : ,5: ]
    Y = R.label
    Y.shape = (25, 1)

    N = network_tensorflow.ConfigNetwork('conf/net.conf', 'embeddings')
    for i in range(500):
        cost = N.fit(A, B, Y)
        print cost

if __name__=='__main__':
    test_Network()





